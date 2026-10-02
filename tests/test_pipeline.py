"""Real Silero/Smart Turn/Pipeline/output with localhost protocol peers."""

import asyncio
import base64
import importlib.util
import json
from pathlib import Path

from aiohttp import web
import pytest
from pipecat_gatemux import GateMuxConfig

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "tests/audio/synthetic-zh.wav"
spec = importlib.util.spec_from_file_location(
    "pipeline_check", ROOT / "examples/live_pipeline.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["playback", "llm"])
async def test_vad_turns_barge_in_context_and_cleanup(server, tmp_path, phase):
    commits = []
    requests = []
    closed_streams = []
    cancelled_llm = []
    tts_requests = []
    cleared = []
    final_tasks = []

    async def stt(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        buffered = bytearray()
        async for msg in ws:
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                await ws.send_json({"type": "session.updated"})
            elif event["type"] == "input_audio_buffer.append":
                buffered.extend(base64.b64decode(event["audio"]))
            elif event["type"] == "input_audio_buffer.commit":
                commits.append(event)
                voiced = any(buffered)
                buffered.clear()
                if not voiced:
                    continue

                async def send_final(item):
                    await asyncio.sleep(0.9)
                    if not ws.closed:
                        await ws.send_json(
                            {
                                "type": "conversation.item.input_audio_transcription.completed",
                                "item_id": item,
                                "transcript": "测试成功。",
                            }
                        )

                final_tasks.append(asyncio.create_task(send_final(str(len(commits)))))
            elif event["type"] == "input_audio_buffer.clear":
                cleared.append(True)
                buffered.clear()
            elif event["type"] == "session.finish":
                if buffered:
                    await ws.send_json(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "late",
                            "transcript": "静音收尾不应形成新轮次。",
                        }
                    )
                await ws.send_json({"type": "session.finished"})
                break
        await asyncio.gather(*final_tasks, return_exceptions=True)
        await ws.close()
        return ws

    async def llm(request):
        requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for text in [
            (
                ("正在生成尚未结束的文本" if phase == "llm" else "这是第一段测试内容。")
                if len(requests) == 1
                else "测试成功。"
            )
        ]:
            body = {
                "id": "offline",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "offline",
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
            await response.write(("data: " + json.dumps(body) + "\n\n").encode())
        if phase == "llm" and len(requests) == 1:
            try:
                for _ in range(200):
                    await asyncio.sleep(0.05)
                    await response.write(b": keepalive\n\n")
            except ConnectionError:
                cancelled_llm.append(True)
                return response
        await response.write(b"data: [DONE]\n\n")
        return response

    async def tts(request):
        tts_requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await asyncio.sleep(3.2)  # real TTS can exceed Pipecat default 3s idle window
        try:
            marker = 1111 if phase == "playback" and len(tts_requests) == 1 else 2222
            chunks = 400 if marker == 1111 else 50
            for _ in range(chunks):
                await response.write(marker.to_bytes(2, "little", signed=True) * 480)
                await asyncio.sleep(0.02)
        except (ConnectionError, asyncio.CancelledError):
            closed_streams.append(True)
        return response

    app = web.Application()
    app.router.add_get("/v1/realtime/transcription", stt)
    app.router.add_post("/v1/chat/completions", llm)
    app.router.add_post("/v1/audio/speech", tts)
    base = await server(app)
    report = await module.run(
        AUDIO,
        tmp_path / "report.json",
        config=GateMuxConfig(api_key="offline", base_url=base),
        barge_in_phase=phase,
    )
    assert report["status"] == "passed", report
    assert len(requests) == 2
    if phase == "playback":
        assert any(
            closed_streams
        ), "Interrupted TTS response must close the HTTP stream"
        assert (
            report["stale_output_chunks"] == 0
        ), "Old playback queue must be discarded"
    else:
        assert cancelled_llm, "Interruption must close the active LLM stream"
    assert len(commits) >= 2
    assert not cleared, "Upstream ASR does not support buffer clear"


async def test_local_voice_stops_on_stt_startup_failure_without_hardware(
    server, monkeypatch, capsys
):
    import sys
    import types
    from pipecat.processors.frame_processor import FrameProcessor

    app = web.Application()

    async def reject(request):
        return web.Response(status=403)

    app.router.add_get("/v1/realtime/transcription", reject)
    config = GateMuxConfig("offline", await server(app))
    voice_spec = importlib.util.spec_from_file_location(
        "local_voice_check", ROOT / "examples/local_voice.py"
    )
    voice = importlib.util.module_from_spec(voice_spec)
    voice_spec.loader.exec_module(voice)
    monkeypatch.setattr(voice.GateMuxConfig, "from_env", staticmethod(lambda: config))

    class PassThrough(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    class NoHardwareTransport:
        def __init__(self, params):
            self._input = PassThrough()
            self._output = PassThrough()

        def input(self):
            return self._input

        def output(self):
            return self._output

    fake_module = types.ModuleType("pipecat.transports.local.audio")
    fake_module.LocalAudioTransport = NoHardwareTransport
    fake_module.LocalAudioTransportParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "pipecat.transports.local.audio", fake_module)
    await asyncio.wait_for(voice.main(), 10)
    output = capsys.readouterr().out
    assert "HTTP 403" in output
    assert "已停止" in output
    assert "语音对话已就绪" not in output
