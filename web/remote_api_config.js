/**
 * ComfyUI-GGUF-VLM Remote API Config 前端扩展
 * 支持动态刷新模型列表
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 注册节点扩展
app.registerExtension({
    name: "ComfyUI.GGUF-VLM.RemoteAPIConfig",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 只处理 RemoteAPIConfig 节点
        if (nodeData.name === "RemoteAPIConfig") {
            
            // 添加刷新按钮到节点
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 找到 model widget（空元组会自动创建 combo widget）
                const modelWidget = this.widgets.find(w => w.name === "model");
                const modelWidgetIndex = this.widgets.findIndex(w => w.name === "model");
                
                // 在 model 后面插入刷新按钮
                const refreshButton = this.addWidget(
                    "button",
                    "🔄 Refresh Models",
                    null,
                    () => {
                        this.refreshModels();
                    }
                );
                
                // 将刷新按钮移到 model 后面
                if (modelWidgetIndex !== -1 && this.widgets.length > 1) {
                    const button = this.widgets.pop();
                    this.widgets.splice(modelWidgetIndex + 1, 0, button);
                }
                
                // 刷新模型列表的方法
                this.refreshModels = async function() {
                    try {
                        // 获取当前的 base_url 和 api_type
                        const baseUrlWidget = this.widgets.find(w => w.name === "base_url");
                        const apiTypeWidget = this.widgets.find(w => w.name === "api_type");
                        const modelWidget = this.widgets.find(w => w.name === "model");
                        
                        if (!baseUrlWidget || !apiTypeWidget || !modelWidget) {
                            console.error("❌ Cannot find required widgets");
                            return;
                        }
                        
                        const baseUrl = baseUrlWidget.value.replace(/\/$/, ''); // 移除末尾斜杠
                        const apiType = apiTypeWidget.value;
                        
                        // 通过 ComfyUI 后端 API 获取模型列表
                        // 这样可以避免浏览器直接访问服务器的 127.0.0.1
                        const apiEndpoint = `/gguf-vlm/refresh-models?base_url=${encodeURIComponent(baseUrl)}&api_type=${encodeURIComponent(apiType)}`;
                        
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), 10000);
                        
                        const response = await fetch(apiEndpoint, {
                            method: 'GET',
                            signal: controller.signal
                        });
                        
                        clearTimeout(timeoutId);
                        
                        if (response.ok) {
                            const data = await response.json();
                            
                            if (data.success && data.models && data.models.length > 0) {
                                // 保存当前选择的模型
                                const currentModel = modelWidget.value;
                                
                                // 更新模型下拉列表
                                modelWidget.options.values = data.models;
                                
                                // 如果之前选择的模型仍然存在,保持选择;否则选择第一个
                                if (data.models.includes(currentModel)) {
                                    modelWidget.value = currentModel;
                                } else {
                                    modelWidget.value = data.models[0];
                                }
                                
                                // 触发节点更新
                                this.setDirtyCanvas(true, true);
                            } else {
                                const errorMsg = data.error || "No models found";
                                modelWidget.options.values = [`⚠️ ${errorMsg}`];
                                modelWidget.value = `⚠️ ${errorMsg}`;
                                this.setDirtyCanvas(true, true);
                            }
                        } else {
                            modelWidget.options.values = [`❌ API Error ${response.status}`];
                            modelWidget.value = `❌ API Error ${response.status}`;
                            this.setDirtyCanvas(true, true);
                        }
                        
                    } catch (error) {
                        const modelWidget = this.widgets.find(w => w.name === "model");
                        if (modelWidget) {
                            if (error.name === 'AbortError') {
                                modelWidget.options.values = ["❌ Request timeout"];
                                modelWidget.value = "❌ Request timeout";
                            } else {
                                modelWidget.options.values = ["❌ Request failed"];
                                modelWidget.value = "❌ Request failed";
                            }
                            this.setDirtyCanvas(true, true);
                        }
                    }
                };
                
                return result;
            };
        }
    }
});
