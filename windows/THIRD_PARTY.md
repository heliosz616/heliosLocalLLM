# 第三方来源

- llama.cpp：<https://github.com/ggml-org/llama.cpp>，固定 commit 见构建脚本，MIT 许可见 `licenses/llama.cpp-MIT.txt`。运行时补丁衍生自该项目。
- 模型：<https://huggingface.co/bartowski/ukisai_Swift-Qwen3.8-27b-GGUF>，精确 revision、文件名及 SHA256 见 `qwen27b/model.source.json`。权重不随源码分发，使用者需遵守模型上游许可。
- Python 依赖：固定版本见 `qwen27b/requirements.lock.txt`，各自遵循其上游许可。

本仓库保留上游版权声明。
