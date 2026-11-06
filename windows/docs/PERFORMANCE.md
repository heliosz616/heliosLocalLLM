# 性能与复验

参考配置：Swift IQ4_XS、128K、主模型及 MTP Q8_0 K/V，MTP n_max=2、p_min=0，CUDA/Vulkan 分层 20,46，batch=8192、ubatch=512。数据来自已加载模型的真实推理，不包含模型加载时间。

| 场景 | 缓存 / 新处理 tokens | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- |
| 16K 冷前缀 | 0 / 16384 | 866.84 | 39.88 |
| 约 112K 冷前缀 | 0 / 114560 | 581.98 | 26.09 |
| 约 112K 扩展到 128K | 114044 / 16900 | 364.97 | 25.01 |

128K 行不是从零填满 128K 的吞吐。以上为固定 128 输出 tokens 的合成检索/重复输出测试；完整生成结束并检查开头口令检索。三个生成工作负载（A、B、C）（temperature=0、thinking medium，各 512 tokens）decode 分别为 32.10、39.09、38.74 tok/s，不能保证任意输入都达到此速度。

同调度 MTP n_max=1 的 KV 对照：Q8 16K/32K prefill 为 805.81/816.91，Q5_1 为 728.39/723.12 tok/s；Q5_1 节省约 1360 MiB KV 显存，但 prefill 慢约 11–13%，因此选 Q8。MTP n_max 在三个实际任务上比较 1、2、4 后选择 2。所有结果只针对该设备和任务集，不是全局最优声明。

## 本地检查

不需要 GPU 的接口/存储测试：

```powershell
.\qwen27b\.venv\Scripts\python.exe -m pytest .\qwen27b\tests -q
```

启动真实服务后，在 `qwen27b` 目录执行（会产生本地测试会话和日志）：

```powershell
.\.venv\Scripts\python.exe scripts/validate.py smoke
.\.venv\Scripts\python.exe scripts/agent_compatibility.py
.\.venv\Scripts\python.exe scripts/agent_combined_regression.py
.\.venv\Scripts\python.exe scripts/verify_security.py
.\.venv\Scripts\python.exe scripts/benchmark.py --prompt-tokens 16384 --needle --prompt-format chat
.\.venv\Scripts\python.exe scripts/benchmark.py --prompt-tokens 114560 --needle --prompt-format chat --session benchmark-long
.\.venv\Scripts\python.exe scripts/benchmark.py --prompt-tokens 130944 --needle --prompt-format chat --session benchmark-long
```

前两次长窗口请求须顺序运行，使用同一 Session；预留 128 tokens 输出。吞吐以引擎实际 `prompt_n` 计，不能用已缓存 tokens 虚增 prefill。工具多轮测试同时验证 reasoning、多个 system 消息和 tool 结果续接。SSD 恢复时间、排队时间、模型切换时间与实际 prefill 分开记录。测试报告含机器地址和生成内容，默认排除发布。
