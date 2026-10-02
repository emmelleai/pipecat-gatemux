"""HTTP streaming TTS: raw PCM frames, GateMux voices and cancellation."""

from collections.abc import AsyncGenerator
import asyncio
import aiohttp
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import TTSSettings
from .config import GateMuxConfig


class GateMuxTTSService(TTSService):
    """Generate PCM audio without buffering an entire synthesis response.

    Client cancellation stops delivery; the gateway can still charge already
    submitted text. Keep sentence aggregation enabled to limit that exposure.
    """

    Settings = TTSSettings

    def __init__(
        self,
        *,
        config: GateMuxConfig | None = None,
        model: str = "gatemux/qwen3-tts-flash",
        voice: str = "Cherry",
        sample_rate: int = 24000,
        settings: TTSSettings | None = None,
        session: aiohttp.ClientSession | None = None,
        max_characters: int = 600,
        **kwargs,
    ) -> None:
        self._config = config or GateMuxConfig.from_env()
        self._session = session
        self._owns_session = session is None
        self._max_characters = max_characters
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        if sample_rate not in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
            raise ValueError("Unsupported TTS sample_rate")
        defaults = TTSSettings(model=model, voice=voice, language=None)
        if settings is not None:
            defaults.apply_update(settings)
        # Keep Pipecat audio contexts alive until the bounded HTTP request
        # finishes. Its 3s default can expire before a real first audio chunk.
        kwargs.setdefault("stop_frame_timeout_s", self._config.timeout + 1)
        super().__init__(
            settings=defaults,
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.timeout)
            )
        return self._session

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream one bounded text segment as aligned PCM16 mono frames."""
        if not text.strip():
            return
        if len(text) > self._max_characters:
            yield ErrorFrame(error="GateMux TTS input exceeds max_characters")
            return
        session = await self._get_session()
        pending = b""
        try:
            async with session.post(
                self._config.url("audio/speech"),
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                json={
                    "model": self._settings.model,
                    "input": text,
                    "voice": self._settings.voice,
                    "stream": True,
                    "response_format": "pcm",
                    "sample_rate": self.sample_rate,
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    yield ErrorFrame(error=f"GateMux TTS HTTP {response.status}")
                    return
                if (
                    response.headers.get("Content-Type", "").split(";")[0]
                    != "audio/pcm"
                ):
                    yield ErrorFrame(error="GateMux TTS expected PCM audio")
                    return
                rate = response.headers.get("X-Sample-Rate")
                if rate is not None and rate != str(self.sample_rate):
                    yield ErrorFrame(error="GateMux TTS sample rate mismatch")
                    return
                await self.start_tts_usage_metrics(text)
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    pending += chunk
                    size = len(pending) // 2 * 2
                    if size:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            pending[:size], self.sample_rate, 1, context_id=context_id
                        )
                        pending = pending[size:]
                if pending:
                    yield ErrorFrame(error="GateMux TTS truncated PCM sample")
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError):
            yield ErrorFrame(error="GateMux TTS connection failed")

    async def cleanup(self) -> None:
        """Close only the HTTP session owned by this service."""
        try:
            await super().cleanup()
        finally:
            if self._owns_session and self._session is not None:
                await self._session.close()
                self._session = None
