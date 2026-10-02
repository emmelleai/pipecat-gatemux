"""GateMux realtime recognition with PCM conversion and explicit turn boundaries."""

import asyncio
import base64
import json
from urllib.parse import urlencode
import aiohttp
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from .config import GateMuxConfig


class GateMuxSTTService(STTService):
    """Stream to GateMux ASR, using Pipecat local VAD to commit each utterance.

    No automatic replay/retry of audio. A failed session reports an error;
    the application may start a new pipeline once the cause is resolved.
    """

    Settings = STTSettings

    def __init__(
        self,
        *,
        config: GateMuxConfig | None = None,
        model: str = "gatemux/qwen3-asr-flash",
        language: str = "zh",
        settings: STTSettings | None = None,
        session: aiohttp.ClientSession | None = None,
        ttfs_p99_latency: float = 3.0,
        **kwargs,
    ) -> None:
        self._config = config or GateMuxConfig.from_env()
        self._session = session
        self._owns_session = session is None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receiver: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._finished = asyncio.Event()
        self._failure: str | None = None
        self._closing = False
        self._resampler = create_stream_resampler()
        self._pending_pcm = b""
        self._preroll_pcm = b""
        self._has_audio = False
        self._vad_seen = False
        self._speech_since_commit = False
        self._last_item: str | None = None
        defaults = STTSettings(model=model, language=language)
        if settings is not None:
            defaults.apply_update(settings)
        # Conservative bootstrap wait, not a measured percentile. Calibrate
        # this window with your own STT latency benchmark before optimizing it.
        super().__init__(settings=defaults, ttfs_p99_latency=ttfs_p99_latency, **kwargs)

    def can_generate_metrics(self) -> bool:
        return True

    async def _update_settings(self, delta: STTSettings) -> dict:
        if self._ws is not None and not self._ws.closed:
            await self.push_error("GateMux STT settings require a new session")
            return {}
        return await super()._update_settings(delta)

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._closing = False
        self._failure = None
        self._ready.clear()
        self._finished.clear()
        self._last_item = None
        self._preroll_pcm = b""
        self._vad_seen = False
        self._speech_since_commit = False
        await self._resampler.reset()
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.timeout)
            )
        try:
            self._ws = await self._session.ws_connect(
                self._config.websocket_url("realtime/transcription")
                + "?"
                + urlencode({"model": self._settings.model}),
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                heartbeat=20,
                max_msg_size=1024 * 1024,
            )
            self._receiver = asyncio.create_task(self._receive())
            await self._send(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": 16000,
                        "language": getattr(
                            self._settings.language, "value", self._settings.language
                        ),
                        "turn_detection": None,
                    },
                }
            )
            await asyncio.wait_for(self._ready.wait(), self._config.timeout)
            if self._failure:
                await self._disconnect(graceful=False)
        except (aiohttp.ClientError, TimeoutError) as exc:
            status = getattr(exc, "status", None)
            detail = type(exc).__name__
            if isinstance(status, int):
                detail += f" HTTP {status}"
            self._failure = f"GateMux STT connection failed: {detail}"
            await self.push_error(self._failure)
            await self._disconnect(graceful=False)

    async def _send(self, event: dict) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_json(event)

    async def _receive(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        event = json.loads(message.data)
                    except (ValueError, TypeError):
                        self._failure = "GateMux STT invalid event"
                        await self.push_error(self._failure)
                        break
                    if not isinstance(event, dict):
                        self._failure = "GateMux STT invalid event"
                        await self.push_error(self._failure)
                        break
                    await self._handle_event(event)
                    if self._failure or self._finished.is_set():
                        break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    break
        except (aiohttp.ClientError, ConnectionError):
            self._failure = "GateMux STT connection lost"
            await self.push_error(self._failure)
        finally:
            if not self._closing and not self._failure and not self._finished.is_set():
                self._failure = "GateMux STT connection closed"
                await self.push_error(self._failure)
            self._ready.set()
            if self._failure and self._ws is not None:
                await self._ws.close()

    async def _handle_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "session.updated":
            self._ready.set()
        elif kind == "conversation.item.input_audio_transcription.text":
            text = event.get("text") or ""
            if text:
                await self.push_frame(
                    InterimTranscriptionFrame(text, self._user_id, time_now_iso8601())
                )
        elif kind == "conversation.item.input_audio_transcription.completed":
            if (event.get("moderation") or {}).get("blocked"):
                await self.push_error("GateMux STT moderation_blocked")
                return
            item = event.get("item_id")
            if item is not None and item == self._last_item:
                return
            self._last_item = item
            text = event.get("transcript") or ""
            if text:
                await self.stop_ttfb_metrics()
                await self.push_frame(
                    TranscriptionFrame(text, self._user_id, time_now_iso8601())
                )
        elif kind == "error":
            code = str((event.get("error") or {}).get("code") or "upstream_error")
            # Do not copy arbitrary server messages that can contain credentials.
            safe = (
                code
                if code.replace("_", "").isalnum() and len(code) < 64
                else "upstream_error"
            )
            self._failure = f"GateMux STT {safe}"
            await self.push_error(self._failure)
            self._ready.set()
        elif kind == "session.finished":
            self._finished.set()

    async def run_stt(self, audio: bytes):
        """Send aligned PCM16; receive task supplies transcription frames."""
        if self._failure or self._closing or self._ws is None or self._ws.closed:
            yield None
            return
        audio = self._pending_pcm + audio
        size = len(audio) // 2 * 2
        self._pending_pcm = audio[size:]
        if size and self._vad_seen and not self._speech_since_commit:
            # VAD sits downstream of STT. Retain 0.5s locally while quiet
            # so its next start event can restore speech onset without
            # submitting unbounded silence or relying on upstream clear.
            limit = int(self.sample_rate * 0.5) * 2
            self._preroll_pcm = (self._preroll_pcm + audio[:size])[-limit:]
            yield None
            return
        if size:
            self._has_audio = True
            converted = await self._resampler.resample(
                audio[:size], self.sample_rate, 16000
            )
            if converted:
                try:
                    await self._send(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(converted).decode(),
                        }
                    )
                except (aiohttp.ClientError, ConnectionError):
                    self._failure = "GateMux STT send failed"
                    await self.push_error(self._failure)
        yield None

    async def _commit_audio(self) -> None:
        tail = await self._resampler.flush()
        if tail:
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(tail).decode(),
                }
            )
        await self._send({"type": "input_audio_buffer.commit"})
        self._has_audio = False
        self._speech_since_commit = False
        await self._resampler.reset()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_seen = True
            self._speech_since_commit = True
            if self._preroll_pcm:
                preroll = self._preroll_pcm
                self._preroll_pcm = b""
                async for _ in self.run_stt(preroll):
                    pass
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_seen = True
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self._has_audio:
            await self.start_ttfb_metrics()
            try:
                await self._commit_audio()
            except (aiohttp.ClientError, ConnectionError):
                self._failure = "GateMux STT send failed"
                await self.push_error(self._failure)

    async def _disconnect(self, *, graceful: bool) -> None:
        self._closing = True
        try:
            if self._ws is not None and not self._ws.closed:
                if (
                    graceful
                    and self._has_audio
                    and (not self._vad_seen or self._speech_since_commit)
                ):
                    await self._commit_audio()
                await self._send({"type": "session.finish"})
                if graceful and self._receiver is not None:
                    try:
                        await asyncio.wait_for(self._finished.wait(), 2)
                    except TimeoutError:
                        pass
        except (aiohttp.ClientError, ConnectionError):
            pass
        finally:
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
            if self._receiver is not None:
                self._receiver.cancel()
                try:
                    await self._receiver
                except asyncio.CancelledError:
                    pass
                self._receiver = None
            self._has_audio = False
            self._pending_pcm = b""
            self._preroll_pcm = b""
            await self._resampler.reset()

    async def stop(self, frame: EndFrame) -> None:
        await self._disconnect(graceful=True)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        await self._disconnect(graceful=False)
        await super().cancel(frame)

    async def cleanup(self) -> None:
        try:
            await self._disconnect(graceful=False)
            await super().cleanup()
        finally:
            if self._owns_session and self._session is not None:
                await self._session.close()
                self._session = None
