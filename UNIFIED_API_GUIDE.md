# 统一 API 文本生成指南

## 概述

统一文本生成系统支持三种模式：
1. **Local (GGUF)** - 直接加载本地 GGUF 文件
2. **Ollama API** - 通过 Ollama 服务调用模型
3. **Nexa SDK API** - 通过 Nexa SDK 服务调用模型

## 节点

### 🔷 Unified Text Model Selector

统一的模型选择器，支持本地和远程模式。

**参数**：

**Mode**: `Local (GGUF)` / `Remote (API)`

**Local 模式参数**：
- `local_model`: 选择 GGUF 文件
- `n_ctx`: 上下文窗口大小（默认: 8192）
- `n_gpu_layers`: GPU 层数（-1 表示全部）

**Remote 模式参数**：
- `base_url`: API 服务地址
- `api_type`: `Ollama` / `Nexa SDK` / `OpenAI Compatible`
- `remote_model`: 模型名称（留空自动获取第一个）
- `refresh_models`: 刷新模型列表

**通用参数**：
- `system_prompt`: 系统提示词（可选）

**输出**：
- `model_config`: 模型配置（传给 Text Generation 节点）

### 🔷 Unified Text Generation

统一的文本生成节点，自动适配本地/远程模式。

**参数**：
- `model_config`: 从 Model Selector
- `max_tokens`: 最大生成 token 数（推荐: 256）
- `temperature`: 温度参数（0.0-2.0）
- `top_p`: Top-p 采样（0.0-1.0）
- `top_k`: Top-k 采样（0-100）
- `repetition_penalty`: 重复惩罚（1.0-2.0）
- `enable_thinking`: 启用思考模式
- `prompt`: 输入提示词

**输出**：
- `context`: 生成的文本
- `thinking`: 思考过程（如果启用）

## 使用场景

### 场景 1: 本地 GGUF 文件

**适用于**：
- 已有 GGUF 文件
- 不需要额外服务
- 快速测试

**配置**：
```
[Unified Text Model Selector]
├─ mode: Local (GGUF)
├─ local_model: Huihui-Qwen3-4B-Instruct-2507-abliterated.Q8_0.gguf
├─ n_ctx: 8192
└─ n_gpu_layers: -1
    ↓
[Unified Text Generation]
└─ prompt: "Hello"
```

### 场景 2: Ollama API

**适用于**：
- 使用 Ollama 生态
- 需要 Ollama 的模型管理
- 多个应用共享模型

**准备工作**：

1. 安装 Ollama：
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. 创建 Modelfile：
```bash
cat > Modelfile << 'EOF'
FROM /workspace/ComfyUI/models/LLM/GGUF/your-model.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
EOF
```

3. 创建模型：
```bash
ollama create my-model -f Modelfile
```

4. 启动服务：
```bash
ollama serve
# 或使用自定义端口
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

**配置**：
```
[Unified Text Model Selector]
├─ mode: Remote (API)
├─ base_url: http://127.0.0.1:11434
├─ api_type: Ollama
└─ remote_model: my-model:latest
    ↓
[Unified Text Generation]
└─ prompt: "Hello"
```

### 场景 3: Nexa SDK API

**适用于**：
- 使用 Nexa SDK 生态
- 需要 Nexa 的模型管理
- 标准化的模型格式

**准备工作**：

1. 安装 Nexa SDK：
```bash
pip install nexaai
```

2. 下载模型：
```bash
nexa pull mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF:Q8_0 --model-type llm
```

3. 启动服务：
```bash
nexa serve
```

**配置**：
```
[Unified Text Model Selector]
├─ mode: Remote (API)
├─ base_url: http://127.0.0.1:11434
├─ api_type: Nexa SDK
└─ remote_model: (自动获取)
    ↓
[Unified Text Generation]
└─ prompt: "Hello"
```

## API 兼容性

### OpenAI 兼容格式

Ollama 和 Nexa SDK 都使用 OpenAI 兼容的 API 格式：

**端点**：
- 模型列表：`GET /v1/models`
- 聊天补全：`POST /v1/chat/completions`

**请求格式**：
```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**响应格式**：
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hi there!"
      }
    }
  ]
}
```

### 参数差异

| 参数 | Ollama | Nexa SDK / OpenAI |
|------|--------|-------------------|
| 重复惩罚 | `repeat_penalty` | `repetition_penalty` |

统一 API 引擎会自动转换这些参数。

## 对比

| 特性 | Local GGUF | Ollama | Nexa SDK |
|------|-----------|--------|----------|
| **服务依赖** | 无 | Ollama 服务 | Nexa 服务 |
| **模型管理** | 手动 | `ollama` CLI | `nexa` CLI |
| **模型格式** | 任意 GGUF | Modelfile | author/model:quant |
| **启动速度** | 快 | 中 | 中 |
| **适用场景** | 本地文件 | Ollama 生态 | Nexa 生态 |

## 故障排除

### 问题 1: 服务不可用

**症状**：`⚠️ service is not available`

**解决**：
```bash
# 检查服务
curl http://127.0.0.1:11434/v1/models

# Ollama
ollama serve

# Nexa SDK
nexa serve
```

### 问题 2: 模型列表为空

**症状**：`⚠️ No models found`

**Ollama 解决**：
```bash
# 创建模型
ollama create model-name -f Modelfile

# 查看模型
ollama list
```

**Nexa SDK 解决**：
```bash
# 下载模型
nexa pull model-name --model-type llm

# 查看模型
nexa list
```

### 问题 3: 端口冲突

**症状**：`bind: address already in use`

**解决**：
```bash
# 使用不同端口
OLLAMA_HOST=127.0.0.1:11435 ollama serve

# 在节点中使用对应端口
base_url: http://127.0.0.1:11435
```

## 测试

运行测试脚本：

```bash
# API 对比测试
python3 tests/test_ollama_nexa_comparison.py

# 节点集成测试
python3 tests/test_unified_text_node.py
```

## 最佳实践

1. **开发阶段**：使用 Local GGUF 模式，快速迭代
2. **生产环境**：使用 Ollama/Nexa SDK，统一管理
3. **多用户**：使用 API 模式，共享模型资源
4. **离线使用**：使用 Local GGUF 模式

## 示例工作流

### 简单对话

```
[Unified Text Model Selector]
├─ mode: Remote (API)
├─ api_type: Ollama
└─ system_prompt: "You are a helpful assistant."
    ↓
[Unified Text Generation]
├─ prompt: "Hello, how are you?"
└─ max_tokens: 100
    ↓
Output: "I'm doing well, thank you! How can I help you today?"
```

### 思考模式

```
[Unified Text Model Selector]
├─ mode: Local (GGUF)
└─ local_model: Qwen3-Thinking-model.gguf
    ↓
[Unified Text Generation]
├─ prompt: "Solve: 25 * 37 = ?"
├─ enable_thinking: True
└─ max_tokens: 256
    ↓
Outputs:
├─ context: "The answer is 925"
└─ thinking: "<think>Let me calculate... 25 * 37 = 25 * 30 + 25 * 7...</think>"
```

## 技术细节

### UnifiedAPIEngine

核心 API 引擎，位于 `core/inference/unified_api_engine.py`。

**特性**：
- 自动检测 API 类型
- 参数自动转换
- 错误处理和重试
- 详细的调试日志

**使用**：
```python
from core.inference.unified_api_engine import get_unified_api_engine

engine = get_unified_api_engine("http://127.0.0.1:11434", "ollama")
response = engine.chat_completion(
    model="model-name",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=512
)
```

## 相关文档

- [README.md](README.md) - 主要文档
- [NEXA_SDK_GUIDE.md](NEXA_SDK_GUIDE.md) - Nexa SDK 详细指南
- [tests/](tests/) - 测试脚本

---

**版本**: 2.3  
**更新日期**: 2025-10-29
