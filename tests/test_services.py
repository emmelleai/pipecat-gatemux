import asyncio
import base64
import json
from unittest.mock import AsyncMock
import aiohttp
from aiohttp import web

try:
    import httpx2 as httpx  # OpenAI SDK 3.x
except ImportError:
    import httpx  # OpenAI SDK 1.x/2.x
import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    StartFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat_gatemux import (
    GateMuxConfig,
    GateMuxLLMService,
    GateMuxSTTService,
    GateMuxTTSService,
)


def test_config_does_not_expose_credentials():
    cfg = GateMuxConfig("secret-for-test")
    assert "secret-for-test" not in repr(cfg)
    assert (
        cfg.websocket_url("realtime/transcription")
        == "wss://rest.gatemux.ai/v1/realtime/transcription"
    )
    with pytest.raises(ValueError):
        GateMuxConfig("secret", "https://user:password@example.com/v1")
    with pytest.raises(ValueError):
        GateMuxConfig("secret", "http://example.com/v1")
    with pytest.raises(ValueError):
        GateMuxConfig.from_env()


async def test_tts_streams_before_end_and_aligns_samples(server, processor_setup):
    received = []
    first = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        received.append(await request.json())
        assert request.headers["Authorization"] == "Bearer test-secret"
        r = web.StreamResponse(
            headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "24000"}
        )
        await r.prepare(request)
        await r.write(b"\x01\x02\x03")
        first.set()
        await release.wait()
        await r.write(b"\x04\x05\x06")
        return r

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    service = GateMuxTTSService(config=GateMuxConfig("test-secret", await server(app)))
    await service.setup(processor_setup)
    service.settings.validate_complete()
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    stream = service.run_tts("你好", "context-one")
    frame = await asyncio.wait_for(anext(stream), 2)
    assert isinstance(frame, TTSAudioRawFrame)
    assert frame.audio == b"\x01\x02"
    assert frame.context_id == "context-one"
    assert first.is_set() and not release.is_set()
    release.set()
    rest = [x async for x in stream]
    assert b"".join(x.audio for x in [frame] + rest) == b"\x01\x02\x03\x04\x05\x06"
    assert received[0]["stream"] is True and received[0]["voice"] == "Cherry"
    assert received[0]["response_format"] == "pcm"
    await service.cleanup()
    assert service._session is None


async def test_tts_errors_are_safe_and_not_retried(server, processor_setup):
    calls = []

    async def handler(request):
        calls.append(1)
        return web.Response(status=402, text="secret-response-test")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    service = GateMuxTTSService(config=GateMuxConfig("test", await server(app)))
    await service.setup(processor_setup)
    frames = [x async for x in service.run_tts("hello", "c")]
    assert len(calls) == 1
    assert isinstance(frames[0], ErrorFrame) and "402" in frames[0].error
    assert "secret-response-test" not in frames[0].error
    frames = [x async for x in service.run_tts("x" * 601, "c")]
    assert isinstance(frames[0], ErrorFrame) and len(calls) == 1
    await service.cleanup()


async def test_tts_cancellation_releases_http_response(server, processor_setup):
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        r = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await r.prepare(request)
        waiting.set()
        await release.wait()
        return r

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handler)
    service = GateMuxTTSService(config=GateMuxConfig("test", await server(app)))
    await service.setup(processor_setup)
    stream = service.run_tts("hello", "c")
    task = asyncio.create_task(anext(stream))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await service.cleanup()
    await service.cleanup()
    release.set()


async def test_stt_events_deduplicate_final_and_hide_blocked_text():
    service = GateMuxSTTService(config=GateMuxConfig("test"))
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    await service._handle_event(
        {"type": "conversation.item.input_audio_transcription.text", "text": "你"}
    )
    final = {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "你好",
        "item_id": "one",
    }
    await service._handle_event(final)
    await service._handle_event(final)
    frames = [x.args[0] for x in service.push_frame.call_args_list]
    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert isinstance(frames[1], TranscriptionFrame) and len(frames) == 2
    await service._handle_event(
        {**final, "item_id": "two", "moderation": {"blocked": True}}
    )
    assert service.push_frame.call_count == 2
    service.push_error.assert_awaited_once_with("GateMux STT moderation_blocked")
    await service._handle_event(
        {"type": "error", "error": {"code": "scope_denied", "message": "secret"}}
    )
    assert service._failure == "GateMux STT scope_denied"
    await service.cleanup()


@pytest.mark.parametrize("input_rate", [16000, 48000])
async def test_stt_websocket_contract_and_finish(server, processor_setup, input_rate):
    events = []
    final_received = asyncio.Event()

    async def handler(request):
        assert request.query["model"] == "gatemux/qwen3-asr-flash"
        assert request.headers["Authorization"] == "Bearer test"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            ev = json.loads(message.data)
            events.append(ev)
            if ev["type"] == "session.update":
                assert ev["session"]["sample_rate"] == 16000
                assert ev["session"]["turn_detection"] is None
                await ws.send_json({"type": "session.updated"})
            elif ev["type"] == "input_audio_buffer.commit":
                await ws.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "你好",
                        "item_id": "one",
                    }
                )
                final_received.set()
            elif ev["type"] == "session.finish":
                await ws.send_json({"type": "session.finished"})
                await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime/transcription", handler)
    service = GateMuxSTTService(config=GateMuxConfig("test", await server(app)))
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    processor_setup.audio_in_sample_rate = input_rate
    await service.setup(processor_setup)
    await service.start(StartFrame())
    assert not service._failure
    audio = b"\x01\x02" * input_rate
    assert [x async for x in service.run_stt(audio)] == [None]
    await service.process_frame(
        VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
    )
    await asyncio.wait_for(final_received.wait(), 2)
    await service._disconnect(graceful=True)
    await service.cleanup()
    sent = b"".join(
        base64.b64decode(e["audio"])
        for e in events
        if e["type"] == "input_audio_buffer.append"
    )
    if input_rate == 16000:
        assert sent == audio
    else:
        assert len(sent) == 16000 * 2
    assert any(e["type"] == "session.finish" for e in events)
    assert service._receiver is None and service._ws is None
    service.push_error.assert_not_awaited()


async def test_llm_streaming_and_tool_calls_use_gatemux():
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test"
        chunks = [
            {
                "id": "chat-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gatemux/test",
                "choices": [
                    {"index": 0, "delta": {"content": "你好"}, "finish_reason": None}
                ],
            },
            {
                "id": "chat-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gatemux/test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_one",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        body = (
            "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    service = GateMuxLLMService(
        config=GateMuxConfig("test", "http://localhost/v1"),
        model="gatemux/test",
        settings=GateMuxLLMService.Settings(
            extra={"extra_body": {"enable_thinking": False}}
        ),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert service._client.max_retries == 0
    stream = await service._client.chat.completions.create(
        model="gatemux/test",
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
    )
    chunks = [c async for c in stream]
    assert chunks[0].choices[0].delta.content == "你好"
    assert chunks[1].choices[0].delta.tool_calls[0].function.name == "weather"
    assert seen[0]["model"] == "gatemux/test"
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    from pipecat.processors.aggregators.llm_context import LLMContext

    await service._process_context(LLMContext([{"role": "user", "content": "你好"}]))
    assert seen[-1]["model"] == "gatemux/test"
    assert seen[-1]["enable_thinking"] is False
    assert any(
        getattr(call.args[0], "text", None) == "你好"
        for call in service.push_frame.call_args_list
    )
    call = service.run_function_calls.call_args.args[0][0]
    assert call.function_name == "weather" and call.arguments == {}
    await service.cleanup()
    assert service._client.is_closed()


async def test_services_preserve_injected_http_session():
    async with aiohttp.ClientSession() as session:
        services = [
            GateMuxSTTService(config=GateMuxConfig("test"), session=session),
            GateMuxTTSService(config=GateMuxConfig("test"), session=session),
        ]
        for service in services:
            await service.cleanup()
            await service.cleanup()
        assert not session.closed


async def test_stt_disconnect_reports_error_without_replay(server, processor_setup):
    calls = []

    async def handler(request):
        calls.append(1)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json({"type": "session.updated"})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime/transcription", handler)
    service = GateMuxSTTService(config=GateMuxConfig("test", await server(app)))
    service.push_error = AsyncMock()
    await service.setup(processor_setup)
    await service.start(StartFrame())
    receiver = service._receiver
    if receiver is not None:
        await asyncio.wait_for(receiver, 2)
    assert service._failure and calls == [1]
    assert [x async for x in service.run_stt(b"\x01\x02" * 160)] == [None]
    await service.cleanup()
    assert service._receiver is None and service._ws is None
    assert service.push_error.await_count == 1


@pytest.mark.parametrize("input_rate", [16000, 24000])
async def test_stt_two_turns_bound_preroll_and_drop_end_silence(
    server, processor_setup, input_rate
):
    from pipecat.frames.frames import EndFrame, VADUserStartedSpeakingFrame

    events = []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            event = json.loads(msg.data)
            events.append(event)
            if event["type"] == "session.update":
                await ws.send_json({"type": "session.updated"})
            elif event["type"] == "session.finish":
                await ws.send_json({"type": "session.finished"})
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime/transcription", handler)
    service = GateMuxSTTService(
        config=GateMuxConfig("test", await server(app)), sample_rate=input_rate
    )
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    await service.setup(processor_setup)
    await service.start(StartFrame())
    voice = (1000).to_bytes(2, "little", signed=True) * (input_rate // 10)
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    async for _ in service.run_stt(voice):
        pass
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    for _ in range(10):
        async for _ in service.run_stt(bytes(input_rate * 2 // 10)):
            pass
    assert len(service._preroll_pcm) == input_rate  # at most 0.5s PCM16
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    async for _ in service.run_stt(voice):
        pass
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    async for _ in service.run_stt(bytes(input_rate * 2)):
        pass
    await service.stop(EndFrame())
    await service.cleanup()
    buffers = [bytearray()]
    for event in events:
        if event["type"] == "input_audio_buffer.append":
            buffers[-1].extend(base64.b64decode(event["audio"]))
        elif event["type"] == "input_audio_buffer.commit":
            buffers.append(bytearray())
    assert [len(b) for b in buffers] == [3200, 19200, 0]
    assert not any(e["type"] == "input_audio_buffer.clear" for e in events)
    service.push_error.assert_not_awaited()


async def test_llm_recovery_developer_role_is_converted_without_mutating_context():
    from pipecat.processors.aggregators.llm_context import LLMContext

    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    service = GateMuxLLMService(
        config=GateMuxConfig("test", "http://localhost/v1"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    context = LLMContext(
        [
            {"role": "developer", "content": "Please repeat."},
            {"role": "user", "content": "hello"},
        ]
    )
    stream = await service.get_chat_completions(context)
    async with stream:
        async for _ in stream:
            pass
    assert [m["role"] for m in seen[0]["messages"]] == ["user", "user"]
    assert context.get_messages()[0]["role"] == "developer"
    await service.cleanup()


async def test_stt_handshake_error_reports_status_without_server_body(
    server, processor_setup
):
    app = web.Application()

    async def reject(request):
        return web.Response(status=403, text="private-server-token")

    app.router.add_get("/v1/realtime/transcription", reject)
    service = GateMuxSTTService(config=GateMuxConfig("test", await server(app)))
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    await service.setup(processor_setup)
    await service.start(StartFrame())
    assert (
        service._failure
        == "GateMux STT connection failed: WSServerHandshakeError HTTP 403"
    )
    assert "private-server-token" not in service._failure
    assert service._ws is None and service._receiver is None
    await service.cleanup()
