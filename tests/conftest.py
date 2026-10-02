"""Local-only servers: never use project .env or a real GateMux endpoint."""

import socket
import pytest
import pytest_asyncio
from aiohttp import web
from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest.fixture(autouse=True)
def no_external_connections(monkeypatch):
    original = socket.socket.connect

    def guarded(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError("Tests cannot connect to external services")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.delenv("GATEMUX_API_KEY", raising=False)


@pytest_asyncio.fixture
async def processor_setup():
    return FrameProcessorSetup(
        clock=SystemClock(),
        task_manager=TaskManager(),
        pipeline_worker=None,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
    )


@pytest_asyncio.fixture
async def server():
    runners = []

    async def launch(app):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    yield launch
    for runner in runners:
        await runner.cleanup()
