# ComfyUI-GGUF-FX

Complete GGUF model support for ComfyUI with three inference modes and auto-download capabilities.

## 🌟 Features

### Three Inference Modes

1. **GGUF Mode** - Local quantized models using llama-cpp-python
2. **Transformers Mode** - Full HuggingFace models
3. **Nexa SDK Mode** - Remote/local GGUF via Nexa SDK service (NEW)

### Nexa SDK Integration (NEW)

- ✅ **Auto-download models** from HuggingFace
- ✅ **Preset model list** with popular models
- ✅ **Dual mode support**: Remote service + Local GGUF files
- ✅ **Configurable API endpoints**
- ✅ **ComfyUI /models/LLM integration**
- ✅ **Thinking mode support** (DeepSeek-R1, Qwen3-Thinking)
- ✅ **Conversation history management**
- ✅ **Multi-endpoint support**

### Other Features

- 🖼️ **Multi-image analysis** (up to 6 images)
- 🎯 **System prompt presets**
- 📝 **Unified output naming**: all text outputs as `context`
- 🔄 **Automatic model caching**
- ⚡ **Device optimization** (CUDA, MPS, CPU)

## 📦 Installation

### 1. Install ComfyUI Custom Node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/weekii/ComfyUI-GGUF-FX.git
cd ComfyUI-GGUF-FX
pip install -r requirements.txt
```

### 2. Install Nexa SDK (for Nexa mode)

```bash
pip install nexaai
```

### 3. Start Nexa SDK Service

```bash
# Start the service
nexa server

# Or specify a custom port
nexa server --port 11434
```

The service will be available at `http://127.0.0.1:11434`

## 🚀 Quick Start

### Using Nexa SDK Mode (Recommended)

#### 1. Basic Text Generation with Auto-Download

```
[Nexa Model Selector]
├─ base_url: http://127.0.0.1:11434
├─ models_dir: /workspace/ComfyUI/models/LLM
├─ model_source: Remote (Nexa Service)
└─ system_prompt: "You are a helpful assistant."
    ↓
[Nexa Text Generation]
├─ preset_model: DavidAU/Qwen3-8B-64k-Josiefied-Uncensored-HORROR-Max-GGUF:Q6_K
├─ auto_download: True  ✅ Will download if not exists
├─ prompt: "Hello, how are you?"
└─ enable_thinking: False
    ↓
Output: context, thinking, raw_response
```

#### 2. Using Custom Model (HuggingFace URL)

```
[Nexa Text Generation]
├─ preset_model: Custom (输入自定义模型)
├─ custom_model: https://huggingface.co/mradermacher/Qwen3-4B-Thinking-2507-Uncensored-Fixed-GGUF/resolve/main/Qwen3-4B-Thinking-2507-Uncensored-Fixed.Q8_0.gguf
├─ auto_download: True
└─ prompt: "Explain quantum computing"
```

The node will:
1. Parse the HuggingFace URL
2. Convert to model ID: `mradermacher/Qwen3-4B-Thinking-2507-Uncensored-Fixed-GGUF:Q8_0`
3. Download the model if not exists
4. Load and run inference

#### 3. Using Local GGUF Files

```bash
# Copy your GGUF file to ComfyUI models directory
cp my-model.gguf /workspace/ComfyUI/models/LLM/
```

```
[Nexa Model Selector]
├─ model_source: Local (GGUF File)
    ↓
[Nexa Text Generation]
├─ preset_model: Custom (输入自定义模型)
├─ custom_model: my-model.gguf
└─ auto_download: False
```

### Using GGUF Mode (Local)

```
[Text Model Loader]
├─ model: Select from dropdown
└─ device: cuda/cpu/mps
    ↓
[Text Generation Node]
├─ prompt: "Your prompt here"
├─ max_tokens: 512
└─ temperature: 0.7
    ↓
Output: context, thinking
```

### Using Transformers Mode

```
[Transformers Vision Model Loader]
├─ model: Qwen/Qwen2-VL-7B-Instruct
└─ device: cuda
    ↓
[Transformers Vision Generation]
├─ image: Connect image
└─ prompt: "Describe this image"
    ↓
Output: context
```

## 📋 Available Nodes

### Nexa SDK Nodes

#### 🔷 Nexa Model Selector
Configure Nexa SDK service and model source.

**Parameters:**
- `base_url`: Nexa SDK service URL (default: `http://127.0.0.1:11434`)
- `models_dir`: Local models directory (default: `/workspace/ComfyUI/models/LLM`)
- `model_source`: Remote (Nexa Service) or Local (GGUF File)
- `refresh_models`: Refresh model list
- `system_prompt`: System prompt (optional)

**Outputs:**
- `model_config`: Configuration for Text Generation node
- `available_models`: List of available models

#### 🔷 Nexa Text Generation
Generate text using Nexa SDK with auto-download support.

**Parameters:**
- `model_config`: From Model Selector
- `preset_model`: Select from preset models or use custom
- `custom_model`: Custom model ID, HuggingFace URL, or local filename
- `auto_download`: Auto-download model if not exists (default: True)
- `prompt`: Input prompt
- `max_tokens`: Maximum tokens to generate (1-8192)
- `temperature`: Temperature (0.0-2.0)
- `top_p`: Top-p sampling (0.0-1.0)
- `top_k`: Top-k sampling (0-100)
- `repetition_penalty`: Repetition penalty (1.0-2.0)
- `enable_thinking`: Enable thinking mode
- `conversation_history`: JSON format conversation history (optional)

**Outputs:**
- `context`: Generated text (final answer)
- `thinking`: Thinking process (if enabled)
- `raw_response`: Raw API response

**Preset Models:**
1. `DavidAU/Qwen3-8B-64k-Josiefied-Uncensored-HORROR-Max-GGUF:Q6_K`
2. `prithivMLmods/Qwen3-4B-2507-abliterated-GGUF:Q8_0`
3. `mradermacher/Qwen3-4B-Thinking-2507-Uncensored-Fixed-GGUF:Q8_0`
4. `mradermacher/Qwen3-Short-Story-Instruct-Uncensored-262K-ctx-4B-GGUF:Q8_0`
5. `Triangle104/Josiefied-Qwen3-4B-abliterated-v2-Q8_0-GGUF`

**Supported Input Formats:**
- Model ID: `user/repo:quantization`
- HuggingFace URL: `https://huggingface.co/user/repo/blob/main/file.gguf`
- Local file: `model.gguf`

#### 🔷 Nexa Service Status
Check Nexa SDK service status and list models.

**Parameters:**
- `base_url`: Service URL
- `models_dir`: Local models directory
- `refresh`: Refresh model list

**Outputs:**
- `status`: Service status summary
- `remote_models`: Remote models list
- `local_models`: Local models list

### GGUF Nodes

- **Text Model Loader** - Load GGUF text models
- **Text Generation Node** - Generate text with GGUF models
- **Vision Model Loader (GGUF)** - Load GGUF vision models
- **Vision Description Node** - Generate image descriptions

### Transformers Nodes

- **Transformers Vision Model Loader** - Load HuggingFace vision models
- **Transformers Vision Generation** - Generate with Transformers models
- **Multi-Image Analysis** - Analyze multiple images (up to 6)

### Utility Nodes

- **System Prompt Config** - Configure system prompts with presets

## 🎯 Use Cases

### 1. Auto-Download and Chat

```
[Nexa Model Selector]
└─ system_prompt: "You are a creative writer."
    ↓
[Nexa Text Generation]
├─ preset_model: mradermacher/Qwen3-Short-Story-Instruct-Uncensored-262K-ctx-4B-GGUF:Q8_0
├─ auto_download: True  ✅ Downloads automatically
├─ prompt: "Write a short story about AI"
└─ max_tokens: 1024
```

### 2. Thinking Mode (DeepSeek-R1 Style)

```
[Nexa Text Generation]
├─ preset_model: mradermacher/Qwen3-4B-Thinking-2507-Uncensored-Fixed-GGUF:Q8_0
├─ enable_thinking: True  ✅ Extracts thinking process
└─ prompt: "Solve: What is 25 * 37?"
    ↓
Output:
├─ context: "The answer is 925"
└─ thinking: "<think>Let me calculate... 25 * 37 = 25 * 30 + 25 * 7...</think>"
```

### 3. Conversation with History

```python
# Build conversation history
history = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "What are its main features?"}
]

# In ComfyUI node
[Nexa Text Generation]
├─ conversation_history: json.dumps(history)
└─ prompt: "Give me an example"
```

### 4. Multi-Image Analysis

```
[Multi-Image Analysis]
├─ image_1: Connect image
├─ image_2: Connect image
├─ image_3: Connect image
└─ prompt: "Compare these images"
    ↓
Output: Detailed comparison
```

### 5. Custom Endpoint

```
[Nexa Model Selector]
├─ base_url: http://192.168.1.100:8080  # Custom server
├─ models_dir: /custom/path/to/models
└─ model_source: Remote (Nexa Service)
```

## 🔧 Configuration

### Model Directory

Default: `/workspace/ComfyUI/models/LLM`

All GGUF models (Nexa SDK, GGUF mode) are stored in this directory.

### API Endpoint

Default: `http://127.0.0.1:11434`

Configurable in each Nexa node. Supports:
- Local: `http://127.0.0.1:11434`
- Custom port: `http://localhost:8080`
- Remote: `http://192.168.1.100:11434`

### Auto-Download

When enabled, the system will:
1. Check if model exists in Nexa service
2. If not, download from HuggingFace using `nexa pull`
3. Store in `/models/LLM` directory
4. Load and run inference

**Requirements:**
- Nexa SDK installed: `pip install nexaai`
- Internet connection for first download
- Sufficient disk space

## 💭 Thinking Mode

Supports automatic extraction of thinking process from models like DeepSeek-R1 and Qwen3-Thinking.

**Supported Tags:**
- `<think>...</think>` (DeepSeek-R1)
- `<thinking>...</thinking>`
- `[THINKING]...[/THINKING]`

**Usage:**
```
[Nexa Text Generation]
├─ enable_thinking: True
└─ prompt: "Explain your reasoning"
    ↓
Outputs:
├─ context: Final answer (thinking tags removed)
└─ thinking: Extracted thinking process
```

**Disable Thinking:**
Set `enable_thinking: False` or add `no_think` to system prompt.

## 📊 Model Comparison

| Mode | Pros | Cons | Use Case |
|------|------|------|----------|
| **Nexa SDK** | Auto-download, Remote service, Easy switching | Requires service running | Quick testing, Shared models |
| **GGUF** | Fast, Low memory, Offline | Manual download | Production, Offline use |
| **Transformers** | Full precision, Latest models | High memory, Slow | Research, Best quality |

## 🐛 Troubleshooting

### Nexa Service Not Available

**Problem:** `❌ Nexa SDK service is not available`

**Solution:**
1. Check if service is running: `curl http://127.0.0.1:11434/v1/models`
2. Start service: `nexa server`
3. Check firewall settings
4. Verify correct URL in node

### Model Download Failed

**Problem:** `❌ Download failed`

**Solution:**
1. Check internet connection
2. Verify HuggingFace is accessible
3. Check disk space
4. Install Nexa SDK: `pip install nexaai`
5. Try manual download: `nexa pull user/repo:quantization`

### Local Model Not Found

**Problem:** `❌ Local model not found`

**Solution:**
1. Check file exists: `ls /workspace/ComfyUI/models/LLM/`
2. Verify filename ends with `.gguf`
3. Check file permissions
4. Use absolute path if needed

### Out of Memory

**Problem:** CUDA out of memory

**Solution:**
1. Use smaller quantization (Q4_0 instead of Q8_0)
2. Reduce `max_tokens`
3. Use CPU mode
4. Close other applications

### Thinking Mode Not Working

**Problem:** Thinking output is empty

**Solution:**
1. Enable `enable_thinking: True`
2. Use a thinking-capable model (e.g., Qwen3-Thinking)
3. Check model output contains thinking tags
4. Remove `no_think` from system prompt

## 📁 Directory Structure

```
ComfyUI-GGUF-FX/
├── __init__.py                     # Node registration
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── config/
│   ├── node_definitions.py        # Node parameter definitions
│   └── paths.py                    # Path configuration
├── core/
│   ├── inference_engine.py        # GGUF inference engine
│   ├── model_loader.py            # Model loader
│   └── inference/
│       ├── nexa_engine.py         # Nexa SDK engine (with auto-download)
│       └── transformers_engine.py # Transformers engine
├── nodes/
│   ├── text_node.py               # GGUF text nodes
│   ├── vision_node.py             # GGUF vision nodes
│   ├── nexa_text_node.py          # Nexa SDK nodes (with presets)
│   ├── vision_node_transformers.py # Transformers vision nodes
│   ├── multi_image_node.py        # Multi-image analysis
│   └── system_prompt_node.py      # System prompt config
└── utils/
    ├── downloader.py              # Model downloader
    ├── device_optimizer.py        # Device optimization
    └── system_prompts.py          # System prompt presets
```

## 🔄 Updates

### v2.1 (Latest)
- ✅ **Auto-download support** for Nexa SDK models
- ✅ **Preset model list** with 5 popular models
- ✅ **HuggingFace URL parsing** - paste URLs directly
- ✅ **Custom model input** - support model ID, URL, or filename
- ✅ **Auto-download toggle** - enable/disable per request

### v2.0
- ✅ Nexa SDK integration
- ✅ ComfyUI /models/LLM directory integration
- ✅ Configurable API endpoints
- ✅ Unified output naming (context)
- ✅ Dual mode support (Remote + Local)

### v1.0
- ✅ GGUF mode
- ✅ Transformers mode
- ✅ Multi-image analysis
- ✅ System prompt presets

## 📝 Requirements

```txt
llama-cpp-python>=0.2.0
transformers>=4.30.0
torch>=2.0.0
Pillow>=9.0.0
requests>=2.25.0
nexaai  # For Nexa SDK mode
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- **GitHub**: https://github.com/weekii/ComfyUI-GGUF-FX
- **Nexa SDK**: https://github.com/NexaAI/nexa-sdk
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

## 👤 Author

**weekii** <weekii2024@gmail.com>

## 🙏 Acknowledgments

- ComfyUI team for the amazing framework
- Nexa AI for the SDK
- HuggingFace for model hosting
- llama.cpp team for GGUF support

---

**Status**: ✅ Production Ready  
**Version**: 2.1  
**Last Updated**: 2025-10-29
