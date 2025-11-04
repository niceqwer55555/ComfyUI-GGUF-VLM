"""
ComfyUI-GGUF-VLM API Routes
提供后端 API 端点，用于前端 JavaScript 调用
"""

import json
from aiohttp import web
from server import PromptServer
from .core.inference.nexa_engine import get_nexa_engine


# 注册 API 路由
@PromptServer.instance.routes.get("/gguf-vlm/refresh-models")
async def refresh_models(request):
    """
    刷新远程 API 模型列表
    
    Query Parameters:
        base_url: API 服务地址
        api_type: API 类型 (ollama, nexa, openai)
    
    Returns:
        JSON: {"success": bool, "models": list, "error": str}
    """
    try:
        # 获取参数
        base_url = request.query.get('base_url', 'http://127.0.0.1:11434')
        api_type = request.query.get('api_type', 'ollama').lower()
        
        # 创建引擎并获取模型
        engine = get_nexa_engine(base_url)
        
        # 检查服务是否可用
        if not engine.is_service_available():
            return web.json_response({
                "success": False,
                "models": [],
                "error": f"Service not available at {base_url}"
            })
        
        # 获取模型列表
        models = engine.get_available_models(force_refresh=True)
        
        if not models:
            return web.json_response({
                "success": False,
                "models": [],
                "error": "No models found"
            })
        
        return web.json_response({
            "success": True,
            "models": models,
            "error": None
        })
        
    except Exception as e:
        return web.json_response({
            "success": False,
            "models": [],
            "error": str(e)
        })
