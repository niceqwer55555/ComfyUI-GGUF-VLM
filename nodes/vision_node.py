"""
Vision Language Node - 视觉语言模型节点（修复版）
"""

import os, platform, gc
import sys
import uuid
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import folder_paths
from comfy.comfy_types import IO

# 添加父目录到路径
module_path = Path(__file__).parent.parent
if str(module_path) not in sys.path:
    sys.path.insert(0, str(module_path))

try:
    from core.model_loader import ModelLoader
    from core.inference_engine import InferenceEngine
    from core.cache_manager import CacheManager
    from utils.registry import RegistryManager
    from utils.downloader import FileDownloader
    from models.vision_models import VisionModelConfig, VisionModelPresets
    from utils.device_optimizer import DeviceOptimizer
    from utils.mmproj_validator import MMProjValidator
    from utils.memory_manager import memory_manager  # 新增内存管理器
except ImportError as e:
    print(f"[ComfyUI-GGUF-VLM] Import error in vision_node: {e}")
    # 尝试相对导入
    from ..core.model_loader import ModelLoader
    from ..core.inference_engine import InferenceEngine
    from ..core.cache_manager import CacheManager
    from ..utils.registry import RegistryManager
    from ..utils.downloader import FileDownloader
    from ..models.vision_models import VisionModelConfig, VisionModelPresets
    from ..utils.device_optimizer import DeviceOptimizer
    from ..utils.mmproj_finder import MMProjFinder
    from ..utils.memory_manager import memory_manager  # 新增内存管理器

class VisionModelLoader:
    """视觉语言模型加载器节点"""

    # 全局实例
    _model_loader = None
    _cache_manager = None
    _registry = None
    _device_optimizer = None
    _loaded_configs = {}

    @classmethod
    def _get_instances(cls):
        """获取全局实例"""
        if cls._model_loader is None:
            cls._model_loader = ModelLoader()
        if cls._cache_manager is None:
            cls._cache_manager = CacheManager()
        if cls._registry is None:
            cls._registry = RegistryManager()
        if cls._device_optimizer is None:
            cls._device_optimizer = DeviceOptimizer()
        return cls._model_loader, cls._cache_manager, cls._registry, cls._device_optimizer

    @classmethod
    def INPUT_TYPES(cls):
        loader, cache, registry, optimizer = cls._get_instances()

        # 获取本地模型
        all_local_models = loader.list_models()

        # 过滤本地模型：只显示视觉语言类型的模型
        local_models = []
        for model_file in all_local_models:
            model_info = registry.find_model_by_filename(model_file)
            if model_info is None or model_info.get('business_type') in ['image_analysis', 'video_analysis']:
                local_models.append(model_file)

        # 获取不同类型的可下载模型
        image_models = registry.get_downloadable_models(business_type='image_analysis', model_loader=loader)
        video_models = registry.get_downloadable_models(business_type='video_analysis', model_loader=loader)

        # 添加类型标签
        categorized_models = []

        if image_models:
            categorized_models.append("--- 🖼️ 图像分析模型 ---")
            categorized_models.extend([name for name, _ in image_models])

        if video_models:
            categorized_models.append("--- 🎥 视频分析模型 ---")
            categorized_models.extend([name for name, _ in video_models])

        if local_models:
            categorized_models.append("--- 💾 本地模型 ---")
            categorized_models.extend(local_models)

        if not categorized_models:
            categorized_models = ["No models found"]

        return {
            "required": {
                "model": (categorized_models, {
                    "default": categorized_models[0] if categorized_models else "No models found",
                    "tooltip": "选择视觉语言模型（按类型分组）"
                }),
                "n_ctx": ("INT", {
                    "default": 8192,
                    "min": 512,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "上下文窗口大小"
                }),
                "device": (["Auto", "GPU", "CPU"], {
                    "default": "Auto",
                    "tooltip": "运行设备 (Auto=自动检测, GPU=全部GPU, CPU=仅CPU)"
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "GPU层数量（-1=全部加载，0=仅CPU，正数=指定层数）"
                }),
            },
            "optional": {
                "mmproj_file": ("STRING", {
                    "default": "",
                    "tooltip": "手动指定 mmproj 文件（可选）"
                }),
                "auto_cleanup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "切换模型时自动清理旧模型显存"
                }),
                "aggressive_cleanup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用激进的显存清理策略（推荐开启）"
                }),
            }
        }

    RETURN_TYPES = ("VISION_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "🤖 GGUF-VLM/🖼️ Vision Models"

    def load_model(self, model, n_ctx=8192, device="Auto", n_gpu_layers=-1, 
                  mmproj_file="", auto_cleanup=True, aggressive_cleanup=True):
        """加载视觉语言模型"""
        loader, cache, registry, optimizer = self._get_instances()

        # 检查CUDA支持状态
        cuda_info = memory_manager.check_cuda_support()
        print(f"\n=== CUDA支持状态 ===")
        print(f"PyTorch CUDA: {cuda_info['pytorch_cuda']}")
        print(f"GPU数量: {cuda_info['gpu_count']}")
        if cuda_info['pytorch_cuda']:
            for i, name in enumerate(cuda_info['gpu_names']):
                print(f"GPU {i}: {name}")
        print(f"llama-cpp-python CUDA: {cuda_info['llama_cuda']}")
        print(f"llama-cpp-python版本: {cuda_info['llama_version']}")

        # 检查 llama-cpp-python 安装状态
        llama_status = optimizer.check_llama_cpp_installation()
        if not llama_status['installed']:
            error_msg = "❌ llama-cpp-python not installed.\n"
            error_msg += "Install with:\n"
            error_msg += "  pip install llama-cpp-python\n"
            error_msg += "\nFor CUDA support, use:\n"
            error_msg += "  pip install llama-cpp-python --force-reinstall --index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121"
            raise RuntimeError(error_msg)

        if llama_status['issues']:
            for issue in llama_status['issues']:
                print(f"⚠️  {issue}")

        # 显示设备信息
        device_summary = optimizer.get_device_summary()
        print(f"\n{device_summary}\n")

        # 处理n_gpu_layers参数 - 根据CUDA支持调整
        if not cuda_info['llama_cuda']:
            print("⚠️  llama-cpp-python没有CUDA支持，强制使用CPU模式")
            n_gpu_layers = 0
            device = "CPU"
        elif n_gpu_layers == -1:
            if device == "Auto":
                optimized_params = optimizer.get_optimized_params(model_size_gb=7.0)
                n_gpu_layers = optimized_params['n_gpu_layers']
                n_batch = optimized_params.get('n_batch', 512)
                print(f"🎯 Auto-optimized: {optimized_params['device_info']}")
                print(f"   GPU layers: {n_gpu_layers}")
                print(f"   Batch size: {n_batch}")
            elif device == "GPU":
                n_gpu_layers = -1
                n_batch = 512
                print(f"🎮 Using GPU (all layers)")
            else:
                n_gpu_layers = 0
                n_batch = 128
                print(f"💻 Using CPU only")
        else:
            n_batch = 512 if n_gpu_layers > 0 else 128
            print(f"⚙️ Manual GPU layers: {n_gpu_layers}")
            print(f"   Batch size: {n_batch}")

        # 检查是否是分组标题
        if model.startswith("---"):
            raise ValueError("请选择一个具体的模型，而不是分组标题")

        print(f"📦 加载模型: {model}")

        # 监控内存使用
        memory_manager.monitor_memory_usage("before model loading")

        # 自动清理旧模型（如果开启）
        if auto_cleanup and VisionLanguageNode._inference_engine is not None:
            print("🧹 Auto-cleanup: cleaning up previous models...")
            VisionLanguageNode.cleanup_all_models(aggressive=aggressive_cleanup)

        # 检查是否需要下载
        if model.startswith("✗"):
            print(f"📥 Model needs to be downloaded: {model}")
            download_info = registry.get_model_download_info(model)

            if download_info:
                downloader = FileDownloader()
                model_dir = loader.model_dirs[0]

                # 下载模型文件
                downloaded_path = downloader.download_from_huggingface(
                    repo_id=download_info['repo'],
                    filename=download_info['filename'],
                    dest_dir=model_dir
                )

                if not downloaded_path:
                    raise RuntimeError(f"Failed to download model: {model}")

                # 下载 mmproj 文件
                if download_info.get('mmproj'):
                    mmproj_repo = download_info.get('mmproj_repo', download_info['repo'])
                    mmproj_downloaded = downloader.download_from_huggingface(
                        repo_id=mmproj_repo,
                        filename=download_info['mmproj'],
                        dest_dir=model_dir
                    )
                    if mmproj_downloaded:
                        mmproj_file = download_info['mmproj']
                        print(f"✅ Downloaded mmproj from {mmproj_repo}")

                model = download_info['filename']
                cache.clear("new model downloaded")
            else:
                raise ValueError(f"Cannot find download info for: {model}")
        elif model.startswith("✓"):
            import re
            model = re.sub(r'^✓\s*', '', model)

        # 查找模型路径
        model_path = loader.find_model(model)
        if not model_path:
            raise FileNotFoundError(f"Model not found: {model}")

        # 查找 mmproj 文件
        mmproj_path = None
        if mmproj_file:
            mmproj_path = loader.find_mmproj(model, mmproj_file)
            if not mmproj_path:
                raise FileNotFoundError(f"mmproj file not found: {mmproj_file}")
        else:
            print(f"🔍 Auto-searching for mmproj file...")
            mmproj_path = loader.find_mmproj(model)

            if not mmproj_path:
                mmproj_name = registry.smart_match_mmproj(model)
                if mmproj_name:
                    mmproj_path = loader.find_mmproj(model, mmproj_name)

                if not mmproj_path:
                    print(f"⚠️  mmproj not found locally, attempting auto-download...")
                    model_info = registry.find_model_by_filename(model)
                    if model_info and model_info.get('mmproj'):
                        downloader = FileDownloader()
                        model_dir = os.path.dirname(model_path)
                        mmproj_path = downloader.download_from_huggingface(
                            repo_id=model_info['repo'],
                            filename=model_info['mmproj'],
                            dest_dir=model_dir
                        )

        if not mmproj_path:
            from ..utils.mmproj_finder import MMProjFinder
            from ..utils.mmproj_validator import MMProjValidator

            finder = MMProjFinder([os.path.dirname(model_path)])
            validator = MMProjValidator()

            suggestions = validator.suggest_mmproj_for_model(model)
            available = finder.list_all_mmproj_files(os.path.dirname(model_path))

            error_msg = f"❌ Could not find mmproj file for {model}.\n\n"
            error_msg += f"💡 Recommended mmproj filename:\n"
            error_msg += f"   {suggestions['primary']}\n\n"

            if available:
                error_msg += f"📁 Available mmproj files in model directory:\n"
                for mmproj_path_item in available:
                    mmproj_name = os.path.basename(mmproj_path_item)
                    compat = validator.check_compatibility(model, mmproj_name)

                    if compat['confidence'] == 'high':
                        error_msg += f"   ✅ {mmproj_name} (推荐使用)\n"
                    elif compat['confidence'] == 'medium':
                        error_msg += f"   ⚠️  {mmproj_name} (可能兼容)\n"
                    else:
                        error_msg += f"   ❌ {mmproj_name} (可能不兼容)\n"

                error_msg += "\n"

            error_msg += "解决方案:\n"
            error_msg += "1. 下载与模型匹配的 mmproj 文件\n"
            error_msg += "2. 如果有推荐的文件，重命名为推荐的文件名\n"
            error_msg += "3. 在节点中手动指定 mmproj_file 参数\n"

            raise FileNotFoundError(error_msg)

        # 应用预设配置
        preset = VisionModelPresets.get_preset(model)
        if preset:
            print(f"📋 Applying preset for {model}")
            if n_ctx == 8192:
                n_ctx = preset.get('n_ctx', n_ctx)

        # 创建配置
        config = VisionModelConfig(
            model_name=model,
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            auto_cleanup=auto_cleanup
        )

        # 验证配置
        validation = config.validate()
        if not validation['valid']:
            raise ValueError(f"Invalid config: {validation['errors']}")

        # 记录已加载的配置
        self._loaded_configs[model_path] = config.to_dict()

        print(f"✅ Vision model loaded: {model}")
        print(f"📁 Using mmproj: {os.path.basename(mmproj_path)}")
        print(f"⚙️ GPU layers: {n_gpu_layers}, Auto-cleanup: {auto_cleanup}")

        # 监控内存使用
        memory_manager.monitor_memory_usage("after model loading")

        return (config.to_dict(),)

    @classmethod
    def cleanup_loaded_configs(cls):
        """清理已加载的配置缓存"""
        cls._loaded_configs.clear()
        print(f"🧹 Cleared all loaded model configs")

class VisionLanguageNode:
    """视觉语言生成节点（增强显存管理版）"""

    # 全局推理引擎
    _inference_engine = None

    @classmethod
    def _get_engine(cls):
        """获取推理引擎"""
        if cls._inference_engine is None:
            cls._inference_engine = InferenceEngine()
        return cls._inference_engine

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VISION_MODEL", {
                    "tooltip": "视觉语言模型配置"
                }),
                "prompt": (IO.STRING, {
                    "default": "Describe this image in detail.",
                    "multiline": False,
                    "tooltip": "用户提示词"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "最大生成 token 数"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "温度参数"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p 采样"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-k 采样"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "随机种子"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "输入图像（与视频二选一）"
                }),
                "video": ("IMAGE", {
                    "tooltip": "输入视频帧序列（与图像二选一）"
                }),
                "system_prompt": (IO.STRING, {
                    "default": "You are a helpful assistant that describes images and videos accurately and in detail.",
                    "multiline": True,
                    "tooltip": "系统提示词（可自定义模型行为）"
                }),
                "cleanup_after_inference": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "推理完成后自动清理模型显存（推荐开启）"
                }),
                "aggressive_cleanup": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "使用激进的显存清理策略（推荐开启）"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("context",)
    FUNCTION = "describe_image"
    CATEGORY = "🤖 GGUF-VLM/🖼️ Vision Models"
    OUTPUT_NODE = True

    def describe_image(self, model, prompt, max_tokens=512,
                      temperature=0.7, top_p=0.9, top_k=40, seed=0,
                      image=None, video=None, system_prompt=None, 
                      cleanup_after_inference=True, aggressive_cleanup=True):
        """生成图像/视频描述"""
        llm = None
        model_path = model.get('model_path')
        
        # 监控内存使用
        memory_info = memory_manager.monitor_memory_usage("start of inference")
        
        # 检测运行模式
        if not memory_info["cuda_available"]:
            print("⚠️  WARNING: Running in CPU mode - performance will be slower")
            print("💡 Tip: Install CUDA-enabled llama-cpp-python for GPU acceleration")
        
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            # 验证输入
            if image is None and video is None:
                raise ValueError("必须提供 image 或 video 输入之一")
            if image is not None and video is not None:
                raise ValueError("不能同时提供 image 和 video 输入，请只选择一个")

            engine = self._get_engine()

            # 确定输入类型
            is_video = video is not None
            input_data = video if is_video else image

            print(f"📊 输入类型: {'视频' if is_video else '图像'}")
            if is_video:
                print(f"🎬 视频帧数: {input_data.shape[0]}")

            # 获取 auto_cleanup 设置
            auto_cleanup = model.get('auto_cleanup', True)

            # 自动清理旧模型
            if auto_cleanup and engine.is_model_loaded(model_path):
                print(f"🧹 Auto-cleanup: unloading previous model")
                self.cleanup_model(model_path, aggressive=aggressive_cleanup)

            # 加载模型（如果未加载）
            if not engine.is_model_loaded(model_path):
                print(f"🔄 Loading vision model into memory...")
                print(f"📁 Model: {os.path.basename(model_path)}")
                print(f"📁 mmproj: {os.path.basename(model['mmproj_path'])}")
                
                # 显示运行模式
                cuda_info = memory_manager.check_cuda_support()
                if cuda_info['llama_cuda'] and model.get('n_gpu_layers', -1) > 0:
                    print(f"🎮 Running on GPU with {model.get('n_gpu_layers', -1)} layers")
                else:
                    print(f"💻 Running on CPU")

                chat_handler = Qwen25VLChatHandler(clip_model_path=model['mmproj_path'])
                llm = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=model.get('n_ctx', 8192),
                    n_gpu_layers=model.get('n_gpu_layers', -1),
                    verbose=model.get('verbose', False),
                    seed=seed
                )

                engine.loaded_models[model_path] = llm
                print(f"✅ Vision model loaded successfully")
                
                # 监控内存使用
                memory_manager.monitor_memory_usage("after model loading")
            else:
                llm = engine.loaded_models[model_path]

            # 处理图像或视频帧
            if is_video:
                image_paths = self._save_video_frames(input_data, seed)
            else:
                image_paths = [self._save_temp_image(input_data, seed)]

            # 构建消息内容
            content = []

            # 添加图像/视频帧
            for img_path in image_paths:
                if not img_path or not os.path.exists(img_path):
                    raise FileNotFoundError(f"无效的图像路径：{img_path}")

                if platform.system() == "Windows":
                    abs_path = os.path.abspath(img_path)
                    img_url = f"file:///{abs_path.replace(os.sep, '/')}"
                else:
                    abs_path = os.path.abspath(img_path)
                    img_url = f"file://{abs_path}"

                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
            
            # 添加用户提示词
            content.append({
                "type": "text",
                "text": prompt
            })

            # 构建消息列表
            messages = []

            # 添加系统提示词（如果提供）
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                print(f"📋 系统提示词: {system_prompt[:50]}...")

            messages.append({"role": "user", "content": content})

            print(f"🤖 Generating {'video' if is_video else 'image'} description...")
            print(f"📝 用户提示词: {prompt[:50]}...")

            # 生成描述
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=False
            )

            output_text = response["choices"][0]["message"]["content"]
            output_text = output_text.strip()

            # 清理临时文件
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                except:
                    pass

            print(f"✅ Generated description ({len(output_text)} chars)")

            # 推理后自动清理模型
            if cleanup_after_inference:
                self.cleanup_model(model_path, aggressive=aggressive_cleanup)
                print(f"🧹 Model cleaned up after inference")

            return (output_text,)

        except ImportError as e:
            error_msg = "❌ llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            print(error_msg)
            return (error_msg,)

        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}"
            print(f"❌ Detailed error:\n{traceback.format_exc()}")

            # 异常时也清理模型
            if cleanup_after_inference and model_path and engine.is_model_loaded(model_path):
                self.cleanup_model(model_path, aggressive=aggressive_cleanup)

            return (error_msg,)

    def _save_temp_image(self, image, seed):
        """保存图像到临时文件"""
        unique_id = uuid.uuid4().hex
        image_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换 tensor 到 PIL Image
        img_array = image.cpu().numpy()
        if img_array.ndim == 4:
            img_array = img_array[0]
        img_array = np.clip(255.0 * img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(str(image_path))

        return str(image_path.resolve())

    def _save_video_frames(self, video, seed, max_frames=8):
        """保存视频帧到临时文件"""
        unique_id = uuid.uuid4().hex
        temp_dir = Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        video_array = video.cpu().numpy()
        num_frames = video_array.shape[0]

        print(f"🎬 处理视频: {num_frames} 帧")

        if num_frames > max_frames:
            indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
            video_array = video_array[indices]
            print(f"📊 采样到 {max_frames} 帧")

        image_paths = []
        for i, frame in enumerate(video_array):
            img_array = np.clip(255.0 * frame, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            frame_path = temp_dir / f"frame_{i:04d}.png"
            img.save(str(frame_path))
            image_paths.append(str(frame_path.resolve()))

        print(f"✅ 保存了 {len(image_paths)} 个视频帧")
        return image_paths

    @classmethod
    def cleanup_model(cls, model_path, aggressive=True):
        """清理指定模型的显存占用"""
        if cls._inference_engine is None:
            return

        if model_path in cls._inference_engine.loaded_models:
            print(f"🧹 Cleaning up model: {os.path.basename(model_path)}")

            # 获取模型实例
            llm = cls._inference_engine.loaded_models.pop(model_path)
            
            # 使用内存管理器强制清理
            memory_manager.force_llama_cleanup(llm)

            # 激进的显存清理
            if aggressive:
                memory_manager.aggressive_memory_cleanup()
            else:
                memory_manager.aggressive_memory_cleanup(max_retries=1)
        else:
            print(f"ℹ️ Model not found in loaded models: {os.path.basename(model_path)}")

    @classmethod
    def cleanup_all_models(cls, aggressive=True):
        """清理所有已加载模型的显存占用"""
        if cls._inference_engine is None:
            return

        if cls._inference_engine.loaded_models:
            print(f"🧹 Cleaning up all loaded vision models ({len(cls._inference_engine.loaded_models)} models)")

            # 删除所有模型引用
            for model_path in list(cls._inference_engine.loaded_models.keys()):
                llm = cls._inference_engine.loaded_models.pop(model_path)
                memory_manager.force_llama_cleanup(llm)

            # 清理配置缓存
            VisionModelLoader.cleanup_loaded_configs()

            # 显存清理
            if aggressive:
                memory_manager.aggressive_memory_cleanup()
            else:
                memory_manager.aggressive_memory_cleanup(max_retries=1)
            
            print(f"✅ All vision models cleaned up")
        else:
            print(f"ℹ️ No loaded vision models to clean up")

    # 新增：手动触发显存清理
    @classmethod
    def manual_memory_cleanup(cls):
        """手动触发显存清理"""
        print("🔄 Manual memory cleanup triggered")
        cls.cleanup_all_models(aggressive=True)

    @classmethod
    def __del__(cls):
        try:
            cls.cleanup_all_models(aggressive=True)
        except:
            pass

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VisionModelLoader": VisionModelLoader,
    "VisionLanguageNode": VisionLanguageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionModelLoader": "🖼️ Vision Model Loader (GGUF)",
    "VisionLanguageNode": "🖼️ Image Analysis (GGUF)",
}
