# Swift Qwen 27B IQ4_XS Serving

Windows 上基于 llama.cpp 的 Swift Qwen3.8-27B 本地推理服务。模型权重为上游发布的 IQ4_XS；默认单流 128K、Q8_0 K/V 和 MTP，每轮最多预测 2 tokens。

模型配置：

| 项目 | 名称 |
| --- | --- |
| 模型仓库 | `bartowski/ukisai_Swift-Qwen3.8-27b-GGUF` |
| 权重文件 | `ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf` |
| 客户端 Model ID | `swift-qwen3.8-27b` |

MTP 使用该模型的预测头，不是另一个需要配置名称的服务模型。

包含 OpenAI-compatible 网关、流式 reasoning/tool calls、多轮工具结果、按用户隔离的 Session、SSD KV 保存恢复、鉴权及监督进程。默认一次只运行一个模型、一个生成请求，其余请求排队。模型、密钥、聊天记录、KV 快照、运行日志及机器配置不随源码分发。

## 环境与安装

已验证的硬件布局为 RTX 5060 Laptop GPU + RX 9070，Windows x64。主模型第 0–19 层在 CUDA，后续层、输出层及 MTP 预测头在 RX 9070；不会加载另一份完整 27B 模型。设备校验会拒绝缺失、重复或身份不明确的设备。此配置不是任意显卡的通用自动调优器。

准备 Git、uv、Python 3.12、Visual Studio 18 Build Tools（C++、CMake、Ninja）、支持 SM120 的 CUDA Toolkit，以及 Vulkan SDK。参考构建使用 CUDA Toolkit 13.2；安装 GPU 驱动后设置 `VULKAN_SDK`，其 `Bin/glslc.exe` 必须存在。构建工具不在默认目录时设置 `SWIFT_VS_ROOT`。建议预留至少 40 GB 空间供模型、源码和构建，SSD KV 配额另算。

在仓库根目录的 PowerShell 执行：

```powershell
.\qwen27b\bootstrap.ps1
```

安装并行下载已固定 revision/hash 的模型及准备运行时，随后编译。模型 SHA256 校验通过才提升为正式文件。下载中断可重跑；构建修补固定 llama.cpp commit，冲突时停止而不覆盖自定义源码。Python 依赖见 `qwen27b/requirements.lock.txt`。源码补丁包含上游 MIT 代码，见 `licenses/`；模型遵循其上游许可，不由本仓库重新授权。

## 启动与连接

```powershell
.\qwen27b\start.ps1
```

默认监听 `127.0.0.1:8000`。启动返回不代表加载完成；鉴权健康检查需返回 `ready`：

```powershell
$swiftKey = (Get-Content .\qwen27b\state\api-key.txt -Raw).Trim()
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -Headers @{Authorization="Bearer $swiftKey"} -NoProxy
```

| 客户端字段 | 默认值 |
| --- | --- |
| OpenAI Base URL | `http://127.0.0.1:8000/v1` |
| Model | `swift-qwen3.8-27b` |
| API Key | 启动时随机生成在 `qwen27b/state/api-key.txt` |
| Thinking | 默认开启，`reasoning_effort=medium`；`none` 可关闭 |
| Session | 可选自定义请求头 `X-Session-ID`，同一对话保持固定 |

Session ID 是本服务扩展，不是 OpenAI 标准必填字段。每轮仍发送完整 `messages`，包含历史 assistant/tool 消息。不要把不同会话复用同一个 ID。已有前缀会匹配请求文本/模板摘要及 token 前缀；缓存命中路径不重复计算整个 KV 的 SHA256。磁盘快照校验与前缀匹配是不同步骤。

支持 `temperature`、`top_p`、`max_tokens`/`max_completion_tokens`、`stream`、`tools`、`tool_choice` 等常用字段。非零 `top_a` 不受当前后端支持，会返回 400。模型能力、采样参数和上下文会影响 MTP 接受率及速度。

停止：

```powershell
.\qwen27b\stop.ps1
```

## 局域网与安全

修改 `qwen27b/config.local.json` 的 `host` 为服务器实际局域网 IP，`allowed_networks` 为授权网段。客户端 URL 替换为该 IP。Windows 防火墙只放行该本地 IP、所需端口及远端网段；安装脚本不自动开放防火墙。内部引擎只监听回环地址 8081，并使用独立密钥；外部网关不开放原生 slot 文件管理接口，也不信任 `X-Forwarded-For`。

每位用户可签发独立 Key（仅显示一次），会话按用户隔离：

```powershell
.\qwen27b\.venv\Scripts\python.exe .\qwen27b\scripts\users.py grant client1
.\qwen27b\.venv\Scripts\python.exe .\qwen27b\scripts\users.py list
.\qwen27b\.venv\Scripts\python.exe .\qwen27b\scripts\users.py revoke client1
```

状态目录限制 NTFS ACL。授权表只存 Key 摘要。局域网 HTTP 不加密传输，非可信网络需自行配置 HTTPS。源码发布不得包含 `state/`、`reports/`、本地配置或模型文件。

## 配置和维护

部署目录为 `qwen27b`，配置环境变量使用 `LLM_SERVICE_*` 前缀。

- `config.local.json`：设备、分层、上下文、Q8 KV、MTP 参数、队列与磁盘配额。
- `models.local.json`：首次启动自动生成单模型注册配置，已有文件不覆盖。默认只有 Swift；后端故障后网关保持在线，下一次生成请求触发重载。`models.example.json` 是扩展配置参考，每个注册模型须有独立 manifest 和快照目录，网络策略一致。
- `spec_draft_n_max=2`、`spec_draft_p_min=0`：MTP 配置，修改后重启。
- `reports/runtime.log`、`gateway.log`、`supervisor-error.log`：本地故障日志，不提交。
- `state/`：密钥、用户表、KV 快照，不公开。删除前先停止服务并自行备份。
- 同时部署其他 GPU 服务时需共享同一 GPU 锁路径或先停止另一服务；不同仓库副本的锁不会自动互相感知。

性能测量和测试命令见 [PERFORMANCE.md](docs/PERFORMANCE.md)，运行时说明见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。接口测试使用模拟引擎；原生 GPU 测试需单独运行。

健康检查返回 `ready` 表示服务可用；后端故障期间返回 503。下一次生成请求会等待重载完成，加载失败时返回错误，后续请求可重试。
