# pipecat-gatemux

GateMux streaming STT, TTS, and LLM services in one Pipecat package.
The adapters use public APIs and have no dependency on the GateMux server.

[中文说明](README.zh-CN.md) · [MIT license](LICENSE)

## Install

Python 3.11+ and Pipecat >=1.12.0,<1.13 are required. The package is not yet
published on PyPI. Install from this repository:

```bash
git clone https://github.com/emmelleai/pipecat-gatemux.git
cd pipecat-gatemux
python -m pip install -e .
# For microphone/speaker examples, install system PortAudio first:
python -m pip install -e '.[local]'
```

Set `GATEMUX_API_KEY` securely in your environment. Optional
`GATEMUX_BASE_URL` defaults to `https://rest.gatemux.ai/v1`.
HTTPS is required except for localhost. Never commit credentials.

## One configuration, three services

```python
from pipecat_gatemux import (
    GateMuxConfig, GateMuxSTTService, GateMuxTTSService, GateMuxLLMService,
)

config = GateMuxConfig.from_env()
stt = GateMuxSTTService(config=config, language="zh")
llm = GateMuxLLMService(
    config=config, model="gatemux/deepseek-v4-flash-ali",
    settings=GateMuxLLMService.Settings(
        extra={"extra_body": {"enable_thinking": False}},
    ),
)
tts = GateMuxTTSService(config=config, voice="Cherry", sample_rate=24000)
```

STT defaults to `gatemux/qwen3-asr-flash` and TTS to
`gatemux/qwen3-tts-flash`. STT uses realtime transcription with 16 kHz mono
PCM16, local VAD, and manual commits. TTS streams mono PCM16 over HTTP.
LLM extends Pipecat's OpenAI Chat Completions service, including context and
tool calls. `enable_thinking` is provider-specific; confirm support when
switching models. Developer-role support defaults to false for compatibility.

## Local conversation

```bash
python examples/local_voice.py --check
python examples/local_voice.py
```

The check opens an STT session without microphone access. Conversation uses
Silero VAD, Smart Turn, and Pipecat interruption handling. Configure microphone
permissions and audio devices. Audio needs `media:create`; LLM needs
`responses:create`. Application credentials also need model permissions;
regional models may require routing consent.

Live examples call paid APIs. They read credentials from the environment and
emit redacted diagnostics. Stopping playback does not undo submitted usage.
There is no automatic audio replay or TTS retry; LLM SDK retries are disabled.
The default TTS segment limit is 600 characters.

```bash
python examples/live_smoke.py --live --output /tmp/gatemux-live
python examples/live_pipeline.py --live \
  --audio tests/audio/synthetic-zh.wav \
  --report /tmp/gatemux-pipeline-report.json
```

The headless pipeline exercises two turns and playback interruption without
audio hardware. It limits LLM calls and has timeouts. Do not disable TLS
verification; on macOS, a trusted CA file may be configured through
`SSL_CERT_FILE=/etc/ssl/cert.pem` when needed.

## Development and validation

```bash
python -m pip install -e '.[dev]' black ruff
black --check .
ruff check .
python -m pytest -q
python -m build
```

Offline tests use localhost HTTP/WebSocket servers and mock transports;
external connections are blocked. Coverage includes Silero VAD, Smart Turn,
delayed transcripts, slow first TTS audio, playback and LLM interruptions,
resampling, context preservation, and cleanup. The baseline was verified with
Python 3.12 and Pipecat 1.12.0. Opt-in live checks have validated the core
headless STT/LLM/TTS pipeline; this is not a benchmark or certification.

Echo cancellation, noisy environments, accents, long sessions, WebRTC,
telephony, and every Pipecat extension have not been comprehensively tested.
STT's initial final-transcript wait is a conservative 3 seconds, not a measured
P99; tune it for your deployment. There are no guarantees for diarization,
word timestamps, confidence scores, or hotwords. This package does not implement
OpenAI Realtime speech-to-speech. Caller-injected HTTP sessions remain owned
by the caller.

## License

Original adapter code is licensed under MIT. Dependencies retain their own
licenses. This does not change the GateMux gateway license.
