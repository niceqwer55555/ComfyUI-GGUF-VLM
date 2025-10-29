#!/bin/bash

################################################################################
# llama-cpp-python CUDA 安装脚本
# 适用于 Vast.ai 和其他 GPU 主机
# 支持自动检测 CUDA 版本和 GPU 型号
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

################################################################################
# 1. 检测环境
################################################################################

print_header "检测系统环境"

# 检测 Python 版本
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python 版本: $PYTHON_VERSION"
else
    print_error "Python3 未安装"
    exit 1
fi

# 检测虚拟环境
if [ -n "$VIRTUAL_ENV" ]; then
    print_success "虚拟环境: $VIRTUAL_ENV"
    PYTHON_CMD="python3"
    PIP_CMD="pip"
elif [ -d "/venv/main" ]; then
    print_info "检测到 /venv/main，将激活虚拟环境"
    source /venv/main/bin/activate
    PYTHON_CMD="python3"
    PIP_CMD="pip"
else
    print_warning "未检测到虚拟环境，使用系统 Python"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# 检测 CUDA
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA 驱动已安装"
    
    # 获取 CUDA 版本
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_info "CUDA 版本: $CUDA_VERSION"
    
    # 获取 GPU 型号
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_info "GPU: $GPU_NAME"
    
    # 检测计算能力
    if [[ "$GPU_NAME" == *"H100"* ]]; then
        COMPUTE_CAP="90"
        print_info "检测到 H100，计算能力: 9.0"
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        COMPUTE_CAP="80"
        print_info "检测到 A100，计算能力: 8.0"
    elif [[ "$GPU_NAME" == *"RTX 4090"* ]] || [[ "$GPU_NAME" == *"RTX 4080"* ]]; then
        COMPUTE_CAP="89"
        print_info "检测到 RTX 40 系列，计算能力: 8.9"
    elif [[ "$GPU_NAME" == *"RTX 3090"* ]] || [[ "$GPU_NAME" == *"RTX 3080"* ]]; then
        COMPUTE_CAP="86"
        print_info "检测到 RTX 30 系列，计算能力: 8.6"
    else
        COMPUTE_CAP="80"
        print_warning "未识别的 GPU，使用默认计算能力: 8.0"
    fi
else
    print_error "未检测到 NVIDIA GPU"
    exit 1
fi

# 检测 CUDA 路径
if [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
    print_success "CUDA 路径: $CUDA_HOME"
elif [ -d "/usr/local/cuda-12.6" ]; then
    CUDA_HOME="/usr/local/cuda-12.6"
    print_success "CUDA 路径: $CUDA_HOME"
elif [ -d "/usr/local/cuda-12" ]; then
    CUDA_HOME="/usr/local/cuda-12"
    print_success "CUDA 路径: $CUDA_HOME"
else
    print_error "未找到 CUDA 安装路径"
    exit 1
fi

################################################################################
# 2. 检查现有安装
################################################################################

print_header "检查现有 llama-cpp-python 安装"

if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
    CURRENT_VERSION=$($PYTHON_CMD -c "import llama_cpp; print(llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else '0.3.x')" 2>/dev/null || echo "未知")
    print_info "当前版本: $CURRENT_VERSION"
    
    # 检查是否有 CUDA 支持
    HAS_CUDA=$($PYTHON_CMD << 'EOF'
import llama_cpp
import os
lib_path = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib')
if os.path.exists(lib_path):
    for f in os.listdir(lib_path):
        if 'cuda' in f.lower() and f.endswith('.so'):
            print("yes")
            exit(0)
print("no")
EOF
)
    
    if [ "$HAS_CUDA" = "yes" ]; then
        print_success "已有 CUDA 支持"
        read -p "是否重新安装? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "保持现有安装"
            exit 0
        fi
    else
        print_warning "当前版本无 CUDA 支持，需要重新安装"
    fi
else
    print_info "llama-cpp-python 未安装"
fi

################################################################################
# 3. 卸载现有版本
################################################################################

print_header "卸载现有版本"

$PIP_CMD uninstall -y llama-cpp-python 2>/dev/null || true
print_success "卸载完成"

################################################################################
# 4. 安装 CUDA 版本
################################################################################

print_header "安装 llama-cpp-python (CUDA 支持)"

# 设置环境变量
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

print_info "编译参数:"
print_info "  CUDA_HOME: $CUDA_HOME"
print_info "  计算能力: $COMPUTE_CAP"

# 从源码编译安装
print_info "开始编译（这可能需要 5-10 分钟）..."

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$COMPUTE_CAP" \
$PIP_CMD install llama-cpp-python --no-cache-dir --verbose 2>&1 | tee /tmp/llama_cpp_install.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "安装成功"
else
    print_error "安装失败，查看日志: /tmp/llama_cpp_install.log"
    exit 1
fi

################################################################################
# 5. 验证安装
################################################################################

print_header "验证 CUDA 支持"

VERIFICATION=$($PYTHON_CMD << 'EOF'
import llama_cpp
import os

lib_path = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib')
cuda_found = False
cuda_size = 0

if os.path.exists(lib_path):
    for f in os.listdir(lib_path):
        if 'cuda' in f.lower() and f.endswith('.so'):
            cuda_found = True
            cuda_size = os.path.getsize(os.path.join(lib_path, f)) / (1024*1024)
            print(f"CUDA库: {f}")
            print(f"大小: {cuda_size:.1f} MB")

if cuda_found:
    print("状态: 成功")
else:
    print("状态: 失败")
EOF
)

echo "$VERIFICATION"

if echo "$VERIFICATION" | grep -q "状态: 成功"; then
    print_success "CUDA 支持已启用"
else
    print_error "CUDA 支持未启用"
    exit 1
fi

################################################################################
# 6. 修复 numpy 版本
################################################################################

print_header "修复依赖版本"

print_info "安装兼容的 numpy 版本..."
$PIP_CMD install 'numpy<2.0,>=1.20' --force-reinstall -q

print_success "依赖修复完成"

################################################################################
# 7. 创建测试脚本
################################################################################

print_header "创建测试脚本"

cat > /tmp/test_llama_cpp_cuda.py << 'EOF'
#!/usr/bin/env python3
"""
测试 llama-cpp-python CUDA 支持
"""

import sys
import os

print("=" * 60)
print("llama-cpp-python CUDA 测试")
print("=" * 60)

# 1. 导入检查
try:
    from llama_cpp import Llama
    print("\n✅ llama-cpp-python 已安装")
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    sys.exit(1)

# 2. 检查 CUDA 库
import llama_cpp
lib_path = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib')

print(f"\n库路径: {lib_path}")

if os.path.exists(lib_path):
    cuda_libs = [f for f in os.listdir(lib_path) if 'cuda' in f.lower() and f.endswith('.so')]
    
    if cuda_libs:
        print("\n✅ CUDA 库:")
        for lib in cuda_libs:
            size = os.path.getsize(os.path.join(lib_path, lib)) / (1024*1024)
            print(f"   - {lib} ({size:.1f} MB)")
    else:
        print("\n❌ 未找到 CUDA 库")
        sys.exit(1)
else:
    print("\n❌ lib 目录不存在")
    sys.exit(1)

# 3. 检查 PyTorch CUDA（可选）
try:
    import torch
    print(f"\n✅ PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("\n⚠️  PyTorch 未安装（可选）")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n下一步:")
print("1. 在 ComfyUI 中选择 device: GPU")
print("2. 运行 nvidia-smi 监控 GPU 使用")
print("3. 享受 GPU 加速！")
print("=" * 60)
EOF

chmod +x /tmp/test_llama_cpp_cuda.py

print_success "测试脚本已创建: /tmp/test_llama_cpp_cuda.py"

################################################################################
# 8. 运行测试
################################################################################

print_header "运行测试"

$PYTHON_CMD /tmp/test_llama_cpp_cuda.py

################################################################################
# 9. 完成
################################################################################

print_header "安装完成"

print_success "llama-cpp-python CUDA 支持已安装"
print_info "环境信息:"
print_info "  Python: $PYTHON_VERSION"
print_info "  CUDA: $CUDA_VERSION"
print_info "  GPU: $GPU_NAME"
print_info "  计算能力: $COMPUTE_CAP"

echo ""
print_info "使用方法:"
echo "  1. 在 ComfyUI Text Model Loader 中选择 device: GPU"
echo "  2. 运行 'watch -n 1 nvidia-smi' 监控 GPU"
echo "  3. 预期性能提升: 20-60倍"

echo ""
print_info "测试脚本: /tmp/test_llama_cpp_cuda.py"
print_info "安装日志: /tmp/llama_cpp_install.log"

echo ""
print_success "🎉 所有设置完成！"
