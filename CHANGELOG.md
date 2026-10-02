# Changelog

## 0.1.0 — 2026-10-02

- Unified configuration and public STT/TTS/LLM services.
- Realtime STT with local VAD commits, HTTP streaming PCM TTS, OpenAI-compatible LLM.
- Local voice example and opt-in real API checks; headless two-turn Pipeline validated.
- Local 0.5s STT pre-roll and inter-turn silence filtering; reset resampler per commit.
- Configurable conservative STT final-transcript wait and HTTP-aligned TTS audio-context timeout.
- 17 offline tests, including actual Silero/Smart Turn, delayed transcripts/audio, playback and LLM interruptions.
- Examples use PipelineWorker and WorkerRunner.
- Developer-role compatibility for Ali empty-turn recovery; safe STT handshake diagnostics and local --check / fail-fast startup.
