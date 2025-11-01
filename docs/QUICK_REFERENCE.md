# 快速参考 - llama-cpp-python CUDA 安装

## 🚀 一键安装

```bash
wget https://raw.githubusercontent.com/walke2019/ComfyUI-GGUF-VLM/main/setup_llama_cpp_cuda.sh
chmod +x setup_llama_cpp_cuda.sh
./setup_llama_cpp_cuda.sh
```

## 📋 脚本功能

✅ 自动检测 Python、CUDA、GPU  
✅ 自动识别 GPU 型号和计算能力  
✅ 从源码编译 CUDA 版本  
✅ 修复依赖冲突  
✅ 验证安装  
✅ 创建测试脚本  

## 🎯 支持的 GPU

| GPU | 计算能力 | 性能提升 |
|-----|---------|---------|
| H100 | 9.0 | 30-60x |
| A100 | 8.0 | 25-50x |
| RTX 4090 | 8.9 | 20-40x |
| RTX 3090 | 8.6 | 20-35x |

## ✅ 验证安装

```bash
python3 /tmp/test_llama_cpp_cuda.py
```

## 🔧 在 ComfyUI 中使用

```
[Text Model Loader]
├─ device: GPU  ✅ 选择 GPU
└─ n_ctx: 8192
```

## 📊 监控 GPU

```bash
watch -n 1 nvidia-smi
```

## 📁 文件位置

- **安装脚本**: `/workspace/setup_llama_cpp_cuda.sh`
- **测试脚本**: `/tmp/test_llama_cpp_cuda.py`
- **安装日志**: `/tmp/llama_cpp_install.log`
- **完整文档**: `/workspace/VASTAI_GPU_SETUP_GUIDE.md`

## 🐛 常见问题

### GPU 未使用？
1. 确认 `device: GPU`
2. 重启 ComfyUI
3. 检查日志

### 编译失败？
```bash
export CUDA_HOME=/usr/local/cuda
./setup_llama_cpp_cuda.sh
```

### 依赖冲突？
```bash
pip install 'numpy<2.0,>=1.20' --force-reinstall
```

## 📞 支持

- GitHub: https://github.com/walke2019/ComfyUI-GGUF-VLM
- 文档: https://llama-cpp-python.readthedocs.io/

---

**快速、简单、自动化！** 🎉
