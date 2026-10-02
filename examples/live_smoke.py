"""Explicitly opt-in, paid smoke check; never collected by pytest.

Reads GATEMUX_API_KEY from the environment. Makes two short TTS requests,
one STT session and up to two bounded LLM requests. No automatic retries.
Reports only synthetic-data summaries, timings and usage quantities.
"""

# ruff: noqa: E402 -- disable Pipecat logging before its imports can emit text.

import argparse
import asyncio
from decimal import Decimal
import json
from pathlib import Path
import time
import wave

from loguru import logger

logger.remove()

import aiohttp
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ErrorFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat_gatemux import (
    GateMuxConfig,
    GateMuxLLMService,
    GateMuxSTTService,
    GateMuxTTSService,
)


async def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cfg = GateMuxConfig.from_env()
    report = {"endpoint": cfg.base_url, "automatic_retries": False}
    setup = FrameProcessorSetup(
        clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=None
    )
    tts = GateMuxTTSService(config=cfg, sample_rate=16000)
    stt = GateMuxSTTService(config=cfg)
    llm = GateMuxLLMService(
        config=cfg,
        model="gatemux/deepseek-v4-flash-ali",
        settings=GateMuxLLMService.Settings(
            max_tokens=64, extra={"extra_body": {"enable_thinking": False}}
        ),
    )
    errors = []

    async def push_error(error, **kwargs):
        errors.append(str(error))

    async def billing(session):
        async with session.get(
            cfg.url("billing/usage"),
            params={"limit": 100},
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                return response.status, []
            body = await response.json()
            return response.status, body.get("data", [])

    async def synthesize(text, name):
        started = time.monotonic()
        first = None
        chunks = []
        async for frame in tts.run_tts(text, name):
            if isinstance(frame, ErrorFrame):
                raise RuntimeError(frame.error)
            if isinstance(frame, TTSAudioRawFrame):
                if first is None:
                    first = time.monotonic() - started
                chunks.append(frame.audio)
        audio = b"".join(chunks)
        if not audio:
            raise RuntimeError("empty_tts_audio")
        with wave.open(str(output / f"{name}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio)
        report[name] = {
            "characters": len(text),
            "audio_seconds": round(len(audio) / 32000, 3),
            "chunks": len(chunks),
            "first_audio_seconds": round(first, 3),
            "total_seconds": round(time.monotonic() - started, 3),
        }
        return audio

    try:
        for service in (tts, stt, llm):
            await service.setup(setup)
            service.settings.validate_complete()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            before_status, before = await billing(session)
            report["billing_before_status"] = before_status
            audio = await synthesize("你好，请回复测试成功。", "input")
            final = asyncio.Event()
            recognized = []
            usage = []
            original_event = stt._handle_event

            async def handle_event(event):
                if event.get("usage"):
                    usage.append(
                        {k: v for k, v in event["usage"].items() if k in ("duration",)}
                    )
                await original_event(event)

            async def stt_frame(frame, direction=FrameDirection.DOWNSTREAM):
                if isinstance(frame, TranscriptionFrame):
                    recognized.append(frame.text)
                    final.set()

            stt._handle_event = handle_event
            stt.push_frame = stt_frame
            stt.push_error = push_error
            started = time.monotonic()
            await stt.start(StartFrame())
            if stt._failure:
                raise RuntimeError(stt._failure)
            for pos in range(0, len(audio), 640):
                async for _ in stt.run_stt(audio[pos : pos + 640]):
                    pass
                await asyncio.sleep(0.02)
            commit_time = time.monotonic()
            await stt.process_frame(
                VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await asyncio.wait_for(final.wait(), cfg.timeout)
            text = "".join(recognized)
            report["stt"] = {
                "characters": len(text),
                "expected_phrase_detected": "测试成功" in text,
                "final_after_commit_seconds": round(time.monotonic() - commit_time, 3),
                "session_seconds": round(time.monotonic() - started, 3),
                "usage": usage,
            }
            await stt._disconnect(graceful=True)
            if errors:
                raise RuntimeError(errors[0])

            answer = []
            llm_usage = []
            first_text = []

            async def llm_frame(frame, direction=FrameDirection.DOWNSTREAM):
                if isinstance(frame, ErrorFrame):
                    raise RuntimeError("llm_error_frame")
                if getattr(frame, "text", None):
                    if not first_text:
                        first_text.append(time.monotonic() - started)
                    answer.append(frame.text)

            async def record_usage(value):
                llm_usage.append(value.model_dump())

            llm.push_frame = llm_frame
            llm.start_llm_usage_metrics = record_usage
            started = time.monotonic()
            await llm._process_context(
                LLMContext(
                    [
                        {
                            "role": "system",
                            "content": "只回复四个汉字：测试成功。不要思考或解释。",
                        },
                        {"role": "user", "content": text},
                    ]
                )
            )
            response_text = "".join(answer)
            if not response_text:
                raise RuntimeError("empty_llm_text")
            report["llm"] = {
                "characters": len(response_text),
                "expected_reply_detected": "测试成功" in response_text,
                "first_text_seconds": round(first_text[0], 3),
                "total_seconds": round(time.monotonic() - started, 3),
                "usage": llm_usage,
            }
            await synthesize(response_text[:60], "reply")

            started = time.monotonic()
            tool_stream = await llm._client.chat.completions.create(
                model=llm.settings.model,
                messages=[
                    {
                        "role": "user",
                        "content": "调用 smoke_check 工具，参数 value 为 ok。",
                    }
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "smoke_check",
                            "description": "Local smoke check, no external actions",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "smoke_check"}},
                max_tokens=64,
                stream=True,
                stream_options={"include_usage": True},
            )
            names = []
            arguments = []
            async with tool_stream:
                async for chunk in tool_stream:
                    if chunk.choices:
                        for call in chunk.choices[0].delta.tool_calls or []:
                            if call.function:
                                names.append(call.function.name or "")
                                arguments.append(call.function.arguments or "")
            report["tools"] = {
                "function_detected": "".join(names) == "smoke_check",
                "arguments_valid": json.loads("".join(arguments) or "{}")
                == {"value": "ok"},
                "total_seconds": round(time.monotonic() - started, 3),
            }
            after_status, after = await billing(session)
            known = {row["request_id"] for row in before}
            new = (
                [row for row in after if row["request_id"] not in known]
                if before_status == 200
                else []
            )
            report["billing_after_status"] = after_status
            report["new_usage"] = [
                {
                    k: row.get(k)
                    for k in ("model", "unit", "quantity", "amount", "currency")
                }
                for row in new
            ]
            totals = {}
            for row in new:
                currency = row.get("currency", "unknown")
                totals[currency] = totals.get(currency, Decimal(0)) + Decimal(
                    row["amount"]
                )
            report["new_usage_totals"] = {k: str(v) for k, v in totals.items()}
            report["status"] = (
                "passed"
                if report["stt"]["expected_phrase_detected"]
                and report["llm"]["expected_reply_detected"]
                and report["tools"]["function_detected"]
                and report["tools"]["arguments_valid"]
                else "needs_review"
            )
    except Exception as exc:
        # Exception text may include provider response bodies. Never print it.
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        if isinstance(exc, RuntimeError) and str(exc).startswith("GateMux"):
            report["adapter_error"] = str(exc)
        report["http_status"] = getattr(exc, "status_code", None)
    finally:
        for service in (stt, tts, llm):
            await service.cleanup()
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Explicit consent to paid API calls",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
