# pipecat-gatemux

GateMux STT、TTS、LLM 的统一 Pipecat 服务适配包。只调用公共 API，不依赖 GateMux 的 app/ee 代码；包内原创代码使用 MIT，网关保持原有许可。

## 安装

独立仓库源码安装（尚未发布 PyPI）：

```bash
git clone https://github.com/emmelleai/pipecat-gatemux.git
cd pipecat-gatemux
```

```bash
python -m pip install -e .
# 使用本机麦克风/扬声器示例：另安装系统 PortAudio，再安装 local extra
python -m pip install -e '.[local]'
```

兼容范围 Pipecat `>=1.12.0,<1.13`，Python >=3.11。测试基线 1.12.0，不以 GitHub main 代表已发布版本。升级 Pipecat 时重新验证生命周期、工具调用与音频帧。

## 一个配置，三类服务

通过环境配置 `GATEMUX_API_KEY`，不要把凭据粘贴进代码或对话。可选 `GATEMUX_BASE_URL`，默认 `https://rest.gatemux.ai/v1`（必须包含 API 前缀 `/v1`）。HTTP 仅允许 localhost。

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

- STT：`/v1/realtime/transcription`，转换为 16 kHz PCM16 单声道；中间/终稿转换成 Pipecat 识别帧。采用 **本地 VAD + 手动 commit**，在 `VADUserStoppedSpeakingFrame` 提交音频，不同时开启服务端 VAD。检测到本地 VAD 后，轮次间静音只在本地保留最多 0.5 秒前置音频，下次开始说话时恢复，避免静音收尾产生额外文本。
- TTS：`/v1/audio/speech`，显式 `stream:true`，输出 PCM16 单声道；保留不完整样本到下一块。默认 Cherry，支持注册表中实际音色，不使用 OpenAI 音色白名单。
- LLM：OpenAI Chat Completions 服务子类，复用 Pipecat 对话上下文、流式文本和工具调用；模型必须支持该协议。可通过 `settings=GateMuxLLMService.Settings(...)` 配置模型和生成参数。默认 `supports_developer_role=False`，Pipecat 将内部恢复提示转换为 user 角色且不修改原上下文；Ali 通道不接受 developer。若所选模型已确认支持，可在实例上设置此属性为 True。

语音示例对阿里百炼 DeepSeek 显式关闭思考以缩短等待；`enable_thinking` 是供应商参数（见 [百炼 DeepSeek 文档](https://www.alibabacloud.com/help/en/model-studio/deepseek-api)），切换通道时重新核实支持情况。开启思考时，小 `max_tokens` 可能全被思考消耗，返回 usage 但没有可播放文本。

完整本机对话示例：`python examples/local_voice.py`。可先加 `--check` 只检查 STT 会话就绪，不启用麦克风，也不要求 PyAudio；启动连接失败或服务错误会停止对话并打印脱敏状态。默认只保留 INFO 日志，避免打印真实对话正文。该示例调用付费接口；先配置自己的测试账户，并确认麦克风/扬声器权限。STT/TTS 需要 `media:create`；LLM 需要 `responses:create`，应用凭据还须登记模型白名单。当前语音模型为中国节点，海外账户须满足中国节点路由同意条件。

## 生命周期、费用与限制

- 不自动重放音频，不自动重试 TTS；LLM SDK 自动重试默认关闭，避免未知结果造成重复调用。STT 连接失败发错误帧，由应用决定是否重建会话。
- STT stop 尝试提交最后音频并 finish，最多等 2 秒；cancel/cleanup 尽快关连接并取消接收任务。清理可重复调用。断线后网关仍可能收尾结算。
- Pipecat 管理用户打断和输出队列，TTS 请求随生成任务取消而释放；**停止播放不代表停止计费**，网关会结算已提交文本。每个合成片段默认最多 600 字符，超限发错误帧，不自动拆成多笔请求。
- 无热词、置信度、说话人分离或词级时间戳保证；不提供 OpenAI Realtime 语音到语音协议。第一版 TTS 为 HTTP 分块，不是双向文本 WebSocket。
- STT 模型和语言在会话启动时确定；活动会话中的 settings 更新会报错误帧，切换时重新创建服务。TTS 和 LLM 的 settings 更新沿用 Pipecat 行为。
- 非 200 HTTP 响应只报告状态码；STT 只保留安全错误码，不打印响应体、音频、识别文本或 key。
- 可注入 `aiohttp.ClientSession`；调用方拥有注入的 session，插件不会关闭它。插件自建 session 在 cleanup 关闭。

## 验证

```bash
python -m pip install -e '.[dev]'
cd .
python -m pytest -q
python -m build
```

离线测试使用本机假 HTTP/WebSocket 服务与 OpenAI SDK 的 mock transport，不调用真实模型或生产服务。测试不能证明首包延迟、识别准确率、实际声音质量与生产计费；这些须另行小额联调。

2026-10-02 离线验证：Python 3.12、Pipecat 1.12.0、OpenAI SDK 3.23.0，17 个测试通过；black/ruff/bandit 通过。包含真实 Silero VAD、默认 Smart Turn、延迟 ASR 终稿、慢 TTS 首音频、两类打断、旧播放队列清理、16k/24k 两轮重采样和正常关闭。测试只连接 localhost，禁止外部服务调用。


手动联调脚本（会计费，不属于 pytest）只读取环境凭据，输出合成测试数据的摘要和本机 WAV：

```bash
python examples/live_smoke.py --live --output /tmp/gatemux-live
```

若本机 Python 缺少默认 CA，配置 `SSL_CERT_FILE` 指向受信任的 CA 文件，例如 macOS 的 `/etc/ssl/cert.pem`；不要关闭 TLS 校验。

Pipecat 社区集成说明：https://github.com/pipecat-ai/pipecat/blob/main/COMMUNITY_INTEGRATIONS.md 。适配由 GateMux 项目维护；尚未向社区提交或发布。

## 完整 Pipeline 验证

`examples/live_pipeline.py` 使用真实 Silero VAD、默认 LocalSmartTurnAnalyzerV3、上下文聚合器和 BaseOutputTransport；用 WAV 模拟麦克风，输出按播放速度消费，使用 Pipecat 实际排队/打断逻辑，不打开音频硬件。

```bash
python examples/live_pipeline.py --live \
  --audio tests/audio/synthetic-zh.wav \
  --report /tmp/gatemux-pipeline-report.json
```

该脚本会调用付费 API；每次最多转发两次 LLM 上下文、max_tokens=128，设有阶段/总体超时，无自动重试。报告仅包含帧类型、角色/字符数量、脱敏错误分类和用量，不记录对话正文或凭据。

2026-10-02 真实验证：初始静音不触发 LLM，音频内部停顿合为一轮，助手播放时用户插话，第二轮保留两条用户上下文并正常回复；10 项检查全部通过、零错误帧、STT 连接释放。插话音频开始输入到打断约 0.575 秒（包含音频自身起始静音及 VAD 门槛，不是单独网络延迟）。LLM 生成中的流取消、旧音频不泄漏通过 localhost 离线 Pipeline 验证。

STT 的 `ttfs_p99_latency=3.0` 为保守初始等待窗口，并非测得的 P99；按部署实际延迟校准，减小会提高提前切轮的风险。TTS 音频上下文默认等待 `config.timeout + 1` 秒，避免 Pipecat 默认 3 秒在首音频前误报无音频；HTTP 仍受 config.timeout 限制。示例使用 PipelineWorker/WorkerRunner。

尚未验证麦克风/扬声器、回声消除、噪声/口音、多人长会话、WebRTC/电话传输或所有 Pipecat 扩展功能。当前是核心语音 Pipeline 验证，不能视为完整功能或识别准确率认证。

