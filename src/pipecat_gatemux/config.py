"""Shared endpoint and credential configuration; credentials never appear in repr."""

from dataclasses import dataclass, field
import os
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class GateMuxConfig:
    """Configuration shared by the three service adapters."""

    api_key: str = field(repr=False)
    base_url: str = "https://rest.gatemux.ai/v1"
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("Set GATEMUX_API_KEY or provide api_key")
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("base_url must be an HTTP(S) API URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must not contain credentials, query or fragment")
        if url.scheme == "http" and url.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise ValueError("Use HTTPS outside localhost")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_env(cls) -> "GateMuxConfig":
        """Read explicit GateMux variables without loading credential files."""
        return cls(
            api_key=os.environ.get("GATEMUX_API_KEY", ""),
            base_url=os.environ.get("GATEMUX_BASE_URL", "https://rest.gatemux.ai/v1"),
        )

    def url(self, path: str) -> str:
        """Join an API-relative path with the configured /v1 base."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def websocket_url(self, path: str) -> str:
        """Use the same host and API prefix for WebSocket endpoints."""
        u = urlsplit(self.url(path))
        return urlunsplit(
            ("wss" if u.scheme == "https" else "ws", u.netloc, u.path, "", "")
        )
