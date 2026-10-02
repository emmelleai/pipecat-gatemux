"""Microphone -> GateMux STT -> GateMux LLM -> GateMux TTS -> speakers.

Requires the local extra, PortAudio and environment credentials. Running this
example sends audio to GateMux and incurs usage charges; offline tests do not.
"""

# ruff: noqa: E402 -- configure logging before Pipecat imports emit data.

import asyncio
import argparse
import json
import sys

import aiohttp
from loguru import logger
from pipecat.frames.frames import CancelFrame

logger.remove()
logger.add(
    sys.stderr, level="INFO", filter=lambda record: record["level"].name == "INFO"
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat_gatemux import (
    GateMuxConfig,
    GateMuxLLMService,
    GateMuxSTTService,
    GateMuxTTSService,
)


async def main() -> None:
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    config = GateMuxConfig.from_env()
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = GateMuxSTTService(config=config, ttfs_p99_latency=3.0)
    llm = GateMuxLLMService(
        config=config,
        settings=GateMuxLLMService.Settings(
            system_instruction="你是语音助手。用简短、自然的中文回答，不使用 Markdown。",
            extra={"extra_body": {"enable_thinking": False}},
        ),
    )
    tts = GateMuxTTSService(config=config)
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    task = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
    )

    @task.event_handler("on_pipeline_started")
    async def ready(_, frame):
        if stt._failure:
            return
        print("语音对话已就绪：请说话，停顿后会回复；Ctrl+C 退出。", flush=True)

    @task.event_handler("on_pipeline_error")
    async def failed(worker, frame):
        status = getattr(frame.exception, "status_code", None)
        detail = (
            frame.error
            if str(frame.error).startswith("GateMux STT connection failed:")
            else type(frame.exception).__name__ if frame.exception else "service_error"
        )
        print(
            f"服务错误：{type(frame.processor).__name__} {detail} HTTP={status}；已停止，请先运行 --check。",
            flush=True,
        )
        await worker.queue_frame(CancelFrame(reason="service_error"))

    @aggregators.user().event_handler("on_user_turn_stopped")
    async def recognized(_, strategy, message):
        print(f"本轮识别 {len(message.content or '')} 字。", flush=True)

    runner = WorkerRunner()
    await runner.add_workers(task)
    await runner.run()


async def check() -> None:
    config = GateMuxConfig.from_env()
    report = {"endpoint": config.base_url}
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            async with session.ws_connect(
                config.websocket_url("realtime/transcription")
                + "?model=gatemux%2Fqwen3-asr-flash",
                headers={"Authorization": f"Bearer {config.api_key}"},
            ) as ws:
                await ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "input_audio_format": "pcm",
                            "sample_rate": 16000,
                            "language": "zh",
                            "turn_detection": None,
                        },
                    }
                )
                while True:
                    event = await ws.receive_json(timeout=10)
                    if event.get("type") == "session.updated":
                        report["stt"] = "ready"
                        break
                    if event.get("type") == "error":
                        report["stt"] = "failed"
                        code = str(
                            (event.get("error") or {}).get("code") or "upstream_error"
                        )
                        report["error_code"] = (
                            code
                            if code.replace("_", "").isalnum() and len(code) < 64
                            else "upstream_error"
                        )
                        break
                await ws.send_json({"type": "session.finish"})
    except (aiohttp.ClientError, TimeoutError) as exc:
        report.update(
            stt="failed",
            exception_type=type(exc).__name__,
            http_status=getattr(exc, "status", None),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Only check STT connection; no recording"
    )
    args = parser.parse_args()
    asyncio.run(check() if args.check else main())
