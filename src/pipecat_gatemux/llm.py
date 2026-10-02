"""GateMux LLM service using Pipecat's maintained OpenAI streaming implementation."""

from pipecat.services.openai.llm import OpenAILLMService
from openai import AsyncOpenAI
from .config import GateMuxConfig


class GateMuxLLMService(OpenAILLMService):
    """Chat streaming, tool calls and metrics with GateMux endpoint defaults.

    Select a GateMux model advertising openai-chat-completions. Retries are
    disabled by default so uncertain responses do not cause duplicate charges.
    """

    # Pipecat rewrites its recovery instructions to user messages without
    # mutating context. The default Ali channel rejects developer roles.
    supports_developer_role = False

    def __init__(
        self,
        *,
        config: GateMuxConfig | None = None,
        model: str = "gatemux/deepseek-v4-flash-ali",
        settings: OpenAILLMService.Settings | None = None,
        http_client=None,
        **kwargs,
    ) -> None:
        cfg = config or GateMuxConfig.from_env()
        self._config = cfg
        self._http_client = http_client
        defaults = self.Settings(model=model)
        if settings is not None:
            defaults.apply_update(settings)
        super().__init__(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            settings=defaults,
            **kwargs,
        )

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            http_client=self._http_client,
            max_retries=0,
            timeout=self._config.timeout,
        )

    async def cleanup(self) -> None:
        try:
            await super().cleanup()
        finally:
            await self._client.close()
