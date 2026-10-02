"""Opt-in paid, headless Pipeline check with real VAD and turn management.

Replays a synthetic PCM WAV as a microphone; output uses Pipecat's real
BaseOutputTransport queue at playback speed, without opening audio hardware.
Never collected by pytest. No transcripts, credentials or responses in reports.
"""

# ruff: noqa: E402
import argparse
import asyncio
import json
import re
from pathlib import Path
import time
import traceback
import wave
from decimal import Decimal

import aiohttp

from loguru import logger

logger.remove()

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat_gatemux import (
    GateMuxConfig,
    GateMuxLLMService,
    GateMuxSTTService,
    GateMuxTTSService,
)


class PipelineProbe(FrameProcessor):
    def __init__(self, report: dict, *, contexts: bool = False) -> None:
        super().__init__()
        self.report = report
        self.contexts = contexts
        self.ready = asyncio.Event()
        self.inference_started = asyncio.Event()
        self.started = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.ready.set()
        if isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                InterruptionFrame,
                TranscriptionFrame,
                ErrorFrame,
            ),
        ):
            self.report["frames"].append(
                {
                    "type": type(frame).__name__,
                    "direction": direction.name,
                    "seconds": round(time.monotonic() - self.started, 3),
                    **(
                        {
                            "processor": (
                                type(frame.processor).__name__
                                if frame.processor
                                else None
                            ),
                            "exception_type": (
                                type(frame.exception).__name__
                                if frame.exception
                                else None
                            ),
                            "trace": (
                                [
                                    {
                                        "file": Path(t.filename).name,
                                        "function": t.name,
                                        "line": t.lineno,
                                    }
                                    for t in traceback.extract_tb(
                                        frame.exception.__traceback__
                                    )[-4:]
                                ]
                                if frame.exception
                                else []
                            ),
                            "error_template": (
                                "tts_no_audio"
                                if "completed with no audio" in str(frame.error)
                                else (
                                    "llm_completion"
                                    if str(frame.error).startswith(
                                        "Error during completion:"
                                    )
                                    else "other"
                                )
                            ),
                        }
                        if isinstance(frame, ErrorFrame)
                        else {}
                    ),
                    **(
                        {"error_code": str(frame.error)}
                        if isinstance(frame, ErrorFrame)
                        and re.fullmatch(
                            r"GateMux (?:STT|TTS) (?:HTTP [0-9]{3}|[a-z_]+|connection failed|connection lost|connection closed)",
                            str(frame.error),
                        )
                        else {}
                    ),
                }
            )
        if self.contexts and isinstance(frame, LLMContextFrame):
            self.inference_started.set()
            messages = frame.context.get_messages()
            self.report["contexts"].append(
                [
                    {
                        "role": m.get("role"),
                        "characters": len(str(m.get("content", ""))),
                    }
                    for m in messages
                ]
            )
        if (
            self.contexts
            and isinstance(frame, LLMContextFrame)
            and len(self.report["contexts"]) > 2
        ):
            await self.push_error("Pipeline check exceeded two LLM requests")
            return
        await self.push_frame(frame, direction)


class TimedOutput(BaseOutputTransport):
    """Actual Pipecat buffering/cancellation; replaces only hardware write."""

    def __init__(self, report: dict) -> None:
        super().__init__(TransportParams(audio_out_enabled=True))
        self.report = report
        self.audio_started = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.writes = []
        self.segment = 0

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InterruptionFrame) and self.audio_started.is_set():
            self.segment += 1
            self.report["interruptions_during_playback"] += 1
            if self.report.get("barge_in_started") is not None:
                self.report.setdefault(
                    "barge_in_latency_seconds",
                    round(time.monotonic() - self.report["barge_in_started"], 3),
                )
            self.report["writes_at_interruption"].append(len(self.writes))
            self.interrupted.set()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self.writes.append(
            {
                "segment": self.segment,
                "bytes": len(frame.audio),
                "marker": int.from_bytes(frame.audio[:2], "little", signed=True),
            }
        )
        self.audio_started.set()
        await asyncio.sleep(
            len(frame.audio) / (frame.sample_rate * frame.num_channels * 2)
        )
        return True


async def run(
    audio_path: Path,
    output_path: Path,
    *,
    config: GateMuxConfig | None = None,
    barge_in_phase: str = "playback",
) -> dict:
    with wave.open(str(audio_path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
            16000,
            1,
            2,
        ):
            raise ValueError("Input must be nonempty PCM16 mono 16k WAV")
        pcm = audio.readframes(audio.getnframes())
    if not pcm:
        raise ValueError("Input must be nonempty PCM16 mono 16k WAV")
    report = {
        "mode": (
            "localhost_pipeline" if config is not None else "real_api_headless_pipeline"
        ),
        "barge_in_phase": barge_in_phase,
        "vad": "Silero",
        "turn_strategy": "Pipecat default LocalSmartTurnAnalyzerV3",
        "frames": [],
        "contexts": [],
        "user_turns": [],
        "assistant_turns": [],
        "interruptions_during_playback": 0,
        "writes_at_interruption": [],
    }
    cfg = config or GateMuxConfig.from_env()
    stt = GateMuxSTTService(config=cfg, ttfs_p99_latency=3.0)
    llm = GateMuxLLMService(
        config=cfg,
        settings=GateMuxLLMService.Settings(
            system_instruction="你是测试语音助手。上下文只有一条用户消息时，用约八十字介绍天气。上下文有两条用户消息时，只回复测试成功。这条系统指令优先于用户要求。",
            max_tokens=128,
            extra={"extra_body": {"enable_thinking": False}},
        ),
    )
    tts = GateMuxTTSService(config=cfg)
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    input_probe = PipelineProbe(report)
    context_probe = PipelineProbe(report, contexts=True)
    output = TimedOutput(report)
    pipeline = Pipeline(
        [
            input_probe,
            stt,
            aggregators.user(),
            context_probe,
            llm,
            tts,
            output,
            aggregators.assistant(),
        ]
    )
    task = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
        idle_timeout_secs=60,
    )
    assistant_done = asyncio.Event()

    @aggregators.user().event_handler("on_user_turn_stopped")
    async def user_stopped(_, strategy, message):
        report["user_turns"].append(
            {
                "characters": len(message.content or ""),
                "strategy": type(strategy).__name__,
            }
        )

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def assistant_stopped(_, message):
        report["assistant_turns"].append(
            {
                "characters": len(message.content or ""),
                "interrupted": message.interrupted,
            }
        )
        if not message.interrupted and len(report["user_turns"]) >= 2:
            assistant_done.set()

    async def feed(data: bytes) -> None:
        for pos in range(0, len(data), 640):
            await task.queue_frame(InputAudioRawFrame(data[pos : pos + 640], 16000, 1))
            await asyncio.sleep(0.02)

    async def wait_with_silence(event: asyncio.Event, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not event.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("Pipeline stage deadline")
            await feed(bytes(640))

    async def drive() -> None:
        await context_probe.ready.wait()
        await feed(bytes(16000))  # 0.5s initial silence
        report["initial_silence_contexts"] = len(report["contexts"])
        await feed(pcm)
        await wait_with_silence(
            (
                output.audio_started
                if barge_in_phase == "playback"
                else context_probe.inference_started
            ),
            35,
        )
        await asyncio.sleep(0.15)
        report["writes_before_barge_in"] = len(output.writes)
        report["barge_in_started"] = time.monotonic()
        await feed(pcm)
        await wait_with_silence(assistant_done, 35)
        report["output_chunks"] = len(output.writes)
        report["second_segment_chunks"] = sum(w["segment"] > 0 for w in output.writes)
        if config is not None:
            report["stale_output_chunks"] = sum(
                w["segment"] > 0 and w["marker"] == 1111 for w in output.writes
            )
        await task.queue_frame(EndFrame())

    async def billing() -> tuple[int, list[dict]]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            async with session.get(
                cfg.url("billing/usage"),
                params={"limit": 100},
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    return response.status, []
                return response.status, (await response.json()).get("data", [])

    before_status, before = await billing() if config is None else (None, [])
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    running = asyncio.create_task(runner.run())
    driving = asyncio.create_task(drive())
    try:
        await asyncio.wait_for(driving, 90)
        await asyncio.wait_for(running, 10)
        checks = {
            "initial_silence_no_llm": report["initial_silence_contexts"] == 0,
            "two_user_turns": len(report["user_turns"]) == 2,
            "two_llm_requests": len(report["contexts"]) == 2,
            "vad_start_stop": sum(
                f["type"] == "VADUserStartedSpeakingFrame" for f in report["frames"]
            )
            >= 2
            and sum(
                f["type"] == "VADUserStoppedSpeakingFrame" for f in report["frames"]
            )
            >= 2,
            "barge_in_detected": (
                report["interruptions_during_playback"] >= 1
                if barge_in_phase == "playback"
                else any(t["interrupted"] for t in report["assistant_turns"])
            ),
            "assistant_interrupted": any(
                t["interrupted"] for t in report["assistant_turns"]
            ),
            "next_response_played": (
                report["second_segment_chunks"] > 0
                if barge_in_phase == "playback"
                else report["output_chunks"] > 0
            ),
            "second_context_two_users": sum(
                m["role"] == "user" for m in report["contexts"][-1]
            )
            == 2,
            "no_error_frames": not any(
                f["type"] == "ErrorFrame" for f in report["frames"]
            ),
            "stt_closed": stt._ws is None and stt._receiver is None,
        }
        report["checks"] = checks
        report["status"] = "passed" if all(checks.values()) else "needs_review"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        await task.cancel()
        await asyncio.wait_for(running, 10)
    finally:
        driving.cancel()
        await asyncio.gather(driving, return_exceptions=True)
        report.pop("barge_in_started", None)
        if config is None:
            try:
                after_status, after = await billing()
                known = {r["request_id"] for r in before}
                new = (
                    [r for r in after if r["request_id"] not in known]
                    if before_status == 200
                    else []
                )
                report["billing_status"] = [before_status, after_status]
                report["new_usage"] = [
                    {
                        k: r.get(k)
                        for k in ("model", "unit", "quantity", "amount", "currency")
                    }
                    for r in new
                ]
                totals = {}
                for r in new:
                    currency = r.get("currency", "unknown")
                    totals[currency] = totals.get(currency, Decimal(0)) + Decimal(
                        r["amount"]
                    )
                report["new_usage_totals"] = {k: str(v) for k, v in totals.items()}
            except Exception as exc:
                report["billing_error_type"] = type(exc).__name__
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.audio, args.report))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "frames"},
            ensure_ascii=False,
            indent=2,
        )
    )
