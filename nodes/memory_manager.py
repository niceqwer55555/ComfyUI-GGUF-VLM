"""
Memory Manager - 显存管理工具（增强版）
"""

import gc
import torch
import psutil
import os
import subprocess
from typing import Dict, Any

class MemoryManager:
    """显存管理器"""
    
    def __init__(self):
        self.initial_memory = self.get_memory_info()
    
    def get_memory_info(self):
        """获取完整的显存和内存信息"""
        memory_info = {}
        
        # GPU显存信息
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_allocated": torch.cuda.memory_allocated(),
                "gpu_reserved": torch.cuda.memory_reserved(),
                "gpu_total": torch.cuda.get_device_properties(0).total_memory,
                "cuda_available": True
            })
        else:
            memory_info.update({
                "gpu_allocated": 0,
                "gpu_reserved": 0,
                "gpu_total": 0,
                "cuda_available": False
            })
        
        # 系统内存信息
        system_memory = psutil.virtual_memory()
        memory_info.update({
            "system_total": system_memory.total,
            "system_available": system_memory.available,
            "system_used": system_memory.used,
            "system_percent": system_memory.percent
        })
        
        # 进程内存信息
        process = psutil.Process()
        process_memory = process.memory_info()
        memory_info.update({
            "process_rss": process_memory.rss,  # 常驻内存
            "process_vms": process_memory.vms   # 虚拟内存
        })
        
        return memory_info
    
    def monitor_memory_usage(self, stage_name):
        """监控内存使用情况"""
        memory_info = self.get_memory_info()
        
        print(f"📊 Memory usage at {stage_name}:")
        
        if memory_info["cuda_available"]:
            print(f"   GPU - Allocated: {memory_info['gpu_allocated']/1024**3:.2f}GB, "
                  f"Reserved: {memory_info['gpu_reserved']/1024**3:.2f}GB")
        else:
            print(f"   ⚠️  CUDA not available - running on CPU")
            
        print(f"   System - Used: {memory_info['system_used']/1024**3:.1f}GB "
              f"({memory_info['system_percent']:.1f}%)")
        print(f"   Process - RSS: {memory_info['process_rss']/1024**3:.2f}GB, "
              f"VMS: {memory_info['process_vms']/1024**3:.2f}GB")
        
        return memory_info
    
    def aggressive_memory_cleanup(self, max_retries=3):
        """激进的显存清理"""
        print("🧹 Starting aggressive memory cleanup...")
        
        # 记录清理前状态
        before_memory = self.get_memory_info()
        
        for attempt in range(max_retries):
            print(f"  Attempt {attempt + 1}/{max_retries}")
            
            # 强制Python垃圾回收
            collected = gc.collect()
            print(f"    GC collected {collected} objects")
            
            if torch.cuda.is_available():
                # 清理PyTorch缓存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 清理CUDA IPC缓存
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                
                # 重置内存统计
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
        
        # 记录清理后状态
        after_memory = self.get_memory_info()
        
        # 计算释放的内存
        freed_memory = {
            "gpu_allocated": before_memory["gpu_allocated"] - after_memory["gpu_allocated"],
            "gpu_reserved": before_memory["gpu_reserved"] - after_memory["gpu_reserved"],
            "process_rss": before_memory["process_rss"] - after_memory["process_rss"]
        }
        
        print(f"✅ Memory cleanup completed:")
        
        if after_memory["cuda_available"]:
            print(f"   GPU - Allocated: {after_memory['gpu_allocated']/1024**3:.2f}GB "
                  f"(Freed: {freed_memory['gpu_allocated']/1024**3:.2f}GB)")
            print(f"   GPU - Reserved: {after_memory['gpu_reserved']/1024**3:.2f}GB "
                  f"(Freed: {freed_memory['gpu_reserved']/1024**3:.2f}GB)")
        else:
            print(f"   ⚠️  Running on CPU - no GPU memory to free")
            
        print(f"   System - Used: {after_memory['system_used']/1024**3:.1f}GB "
              f"({after_memory['system_percent']:.1f}%)")
        print(f"   Process - RSS: {after_memory['process_rss']/1024**3:.2f}GB "
              f"(Freed: {freed_memory['process_rss']/1024**3:.2f}GB)")
        
        return freed_memory
    
    def force_llama_cleanup(self, llama_instance):
        """强制清理Llama实例"""
        if llama_instance is None:
            return
        
        try:
            print("    Force cleaning Llama instance...")
            
            # 尝试各种清理方法
            cleanup_methods = ['close', '__del__', 'free', 'cleanup']
            
            for method_name in cleanup_methods:
                if hasattr(llama_instance, method_name):
                    try:
                        method = getattr(llama_instance, method_name)
                        method()
                        print(f"      ✅ Called {method_name}()")
                    except Exception as e:
                        print(f"      ⚠️  {method_name}() failed: {e}")
            
            # 强制删除引用
            del llama_instance
            
            # 立即垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"    ⚠️  Llama cleanup failed: {e}")
    
    def check_cuda_support(self):
        """检查CUDA支持状态"""
        cuda_info = {
            "pytorch_cuda": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": []
        }
        
        if cuda_info["pytorch_cuda"]:
            for i in range(cuda_info["gpu_count"]):
                cuda_info["gpu_names"].append(torch.cuda.get_device_name(i))
        
        # 检查llama-cpp-python CUDA支持
        try:
            import llama_cpp
            cuda_info["llama_cuda"] = hasattr(llama_cpp, 'LLAMA_CUBLAS')
            cuda_info["llama_version"] = getattr(llama_cpp, '__version__', 'unknown')
        except ImportError:
            cuda_info["llama_cuda"] = False
            cuda_info["llama_version"] = "not installed"
        
        return cuda_info

# 全局内存管理器实例
memory_manager = MemoryManager()