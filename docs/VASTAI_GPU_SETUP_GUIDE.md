# Vast.ai GPU 主机 llama-cpp-python CUDA 安装指南

## 快速开始

### 一键安装

```bash
# 下载脚本
wget https://raw.githubusercontent.com/weekii/ComfyUI-GGUF-FX/main/setup_llama_cpp_cuda.sh

# 或者如果你已经有脚本
chmod +x setup_llama_cpp_cuda.sh

# 运行安装
./setup_llama_cpp_cuda.sh
```

## 脚本功能

### 自动检测

- ✅ **Python 版本** - 自动检测 Python 3.x
- ✅ **虚拟环境** - 自动检测并激活 `/venv/main` 或当前环境
- ✅ **CUDA 版本** - 从 nvidia-smi 读取
- ✅ **GPU 型号** - 自动识别并设置计算能力
  - H100: 9.0
  - A100: 8.0
  - RTX 4090/4080: 8.9
  - RTX 3090/3080: 8.6
- ✅ **CUDA 路径** - 自动查找 `/usr/local/cuda*`

### 自动安装

1. 卸载现有 llama-cpp-python
2. 从源码编译 CUDA 版本
3. 针对 GPU 优化编译参数
4. 修复 numpy 版本冲突
5. 验证 CUDA 支持
6. 创建测试脚本

### 安全特性

- 检查现有安装，避免重复
- 遇到错误自动退出
- 保存安装日志
- 彩色输出，易于查看

## 支持的 GPU

### 已测试

- ✅ NVIDIA H100 (计算能力 9.0)
- ✅ NVIDIA A100 (计算能力 8.0)
- ✅ RTX 4090 (计算能力 8.9)
- ✅ RTX 3090 (计算能力 8.6)

### 理论支持

所有支持 CUDA 的 NVIDIA GPU（计算能力 >= 6.0）

## 使用场景

### 场景 1: 新的 Vast.ai 实例

```bash
# 1. SSH 登录到 Vast.ai 实例
ssh -p PORT root@IP

# 2. 下载脚本
wget https://raw.githubusercontent.com/weekii/ComfyUI-GGUF-FX/main/setup_llama_cpp_cuda.sh

# 3. 运行
chmod +x setup_llama_cpp_cuda.sh
./setup_llama_cpp_cuda.sh

# 4. 完成！
```

### 场景 2: 已有 ComfyUI 环境

```bash
# 1. 激活虚拟环境（如果有）
source /venv/main/bin/activate

# 2. 运行脚本
./setup_llama_cpp_cuda.sh

# 3. 重启 ComfyUI
cd /workspace/ComfyUI
python3 main.py
```

### 场景 3: 更换 GPU 型号

```bash
# 如果更换了 GPU（如从 A100 换到 H100）
# 重新运行脚本即可，会自动检测新 GPU
./setup_llama_cpp_cuda.sh
```

## 验证安装

### 方法 1: 使用测试脚本

```bash
python3 /tmp/test_llama_cpp_cuda.py
```

**预期输出**:
```
============================================================
llama-cpp-python CUDA 测试
============================================================

✅ llama-cpp-python 已安装

库路径: /venv/main/lib/python3.12/site-packages/llama_cpp/lib

✅ CUDA 库:
   - libggml-cuda.so (130.8 MB)

✅ PyTorch CUDA: True
   CUDA 版本: 12.6
   GPU: NVIDIA H100 80GB HBM3

============================================================
测试完成！
============================================================
```

### 方法 2: 在 ComfyUI 中测试

1. **加载模型**
   ```
   [Text Model Loader]
   ├─ model: 选择 GGUF 模型
   ├─ device: GPU  ✅
   └─ n_ctx: 8192
   ```

2. **监控 GPU**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **生成文本**
   ```
   [Text Generation]
   ├─ prompt: "Write a story"
   └─ max_tokens: 512
   ```

**预期结果**:
- GPU 内存使用增加
- GPU 利用率 30-90%
- 生成速度: 100-300+ tokens/s

## 故障排除

### 问题 1: 编译失败

**错误**: `CUDA not found`

**解决方案**:
```bash
# 手动设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新运行脚本
./setup_llama_cpp_cuda.sh
```

### 问题 2: 未检测到 GPU

**错误**: `未检测到 NVIDIA GPU`

**解决方案**:
```bash
# 检查驱动
nvidia-smi

# 如果失败，重启实例或联系 Vast.ai 支持
```

### 问题 3: 虚拟环境问题

**错误**: `未检测到虚拟环境`

**解决方案**:
```bash
# 手动激活虚拟环境
source /venv/main/bin/activate

# 或创建新的虚拟环境
python3 -m venv /venv/main
source /venv/main/bin/activate

# 重新运行脚本
./setup_llama_cpp_cuda.sh
```

### 问题 4: GPU 未使用

**症状**: ComfyUI 日志显示 "💻 Using CPU only"

**解决方案**:
1. 确认 `device` 设置为 `GPU`
2. 重启 ComfyUI
3. 检查日志是否有错误

### 问题 5: 依赖冲突

**警告**: `numpy version incompatible`

**解决方案**:
```bash
# 脚本会自动修复，如果仍有问题：
pip install 'numpy<2.0,>=1.20' --force-reinstall
```

## 性能基准

### H100 GPU

| 模型大小 | CPU | H100 GPU | 加速比 |
|---------|-----|----------|--------|
| 4B Q4_K_M | 5-8 tokens/s | 150-250 tokens/s | 30-40x |
| 7B Q4_K_M | 3-5 tokens/s | 100-180 tokens/s | 30-50x |
| 13B Q4_K_M | 2-3 tokens/s | 60-120 tokens/s | 30-60x |

### A100 GPU

| 模型大小 | CPU | A100 GPU | 加速比 |
|---------|-----|----------|--------|
| 4B Q4_K_M | 5-8 tokens/s | 120-200 tokens/s | 25-35x |
| 7B Q4_K_M | 3-5 tokens/s | 80-150 tokens/s | 25-40x |
| 13B Q4_K_M | 2-3 tokens/s | 50-100 tokens/s | 25-50x |

### RTX 4090

| 模型大小 | CPU | RTX 4090 | 加速比 |
|---------|-----|----------|--------|
| 4B Q4_K_M | 5-8 tokens/s | 100-180 tokens/s | 20-30x |
| 7B Q4_K_M | 3-5 tokens/s | 70-130 tokens/s | 20-35x |
| 13B Q4_K_M | 2-3 tokens/s | 40-80 tokens/s | 20-40x |

## 文件说明

### 脚本生成的文件

- `/tmp/llama_cpp_install.log` - 安装日志
- `/tmp/test_llama_cpp_cuda.py` - 测试脚本

### 脚本位置

推荐保存在：
- `/workspace/setup_llama_cpp_cuda.sh` - 工作目录
- 或 GitHub 仓库中便于下载

## 高级用法

### 自定义计算能力

如果脚本未正确识别你的 GPU：

```bash
# 编辑脚本，找到 COMPUTE_CAP 设置
# 或者手动设置环境变量
export COMPUTE_CAP="80"  # 你的 GPU 计算能力
./setup_llama_cpp_cuda.sh
```

### 自定义 CUDA 路径

```bash
export CUDA_HOME=/path/to/cuda
./setup_llama_cpp_cuda.sh
```

### 静默安装

```bash
# 跳过确认提示
yes | ./setup_llama_cpp_cuda.sh
```

## 与 ComfyUI-GGUF-FX 集成

此脚本专为 ComfyUI-GGUF-FX 设计，安装后：

1. **Text Model Loader** 节点支持 GPU 加速
2. **Vision Description** 节点支持 GPU 加速
3. 所有 GGUF 模型推理都会使用 GPU

## 更新和维护

### 更新 llama-cpp-python

```bash
# 重新运行脚本即可
./setup_llama_cpp_cuda.sh
```

### 检查更新

```bash
# 检查最新版本
pip search llama-cpp-python

# 或访问
# https://github.com/abetlen/llama-cpp-python
```

## 支持

### 问题反馈

- GitHub Issues: https://github.com/weekii/ComfyUI-GGUF-FX/issues
- 脚本问题请附上 `/tmp/llama_cpp_install.log`

### 相关资源

- llama-cpp-python 官方文档: https://llama-cpp-python.readthedocs.io/
- llama.cpp 项目: https://github.com/ggerganov/llama.cpp
- ComfyUI-GGUF-FX: https://github.com/weekii/ComfyUI-GGUF-FX

## 许可证

此脚本遵循 MIT 许可证，可自由使用和修改。

---

**版本**: 1.0  
**日期**: 2025-10-29  
**作者**: ComfyUI-GGUF-FX  
**测试环境**: Vast.ai H100, A100, RTX 4090
