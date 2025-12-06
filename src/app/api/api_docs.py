"""
API 文档配置

使用 Flasgger 为 Flask API 生成 Swagger 文档。
"""

from flasgger import Swagger


def init_swagger(app):
    """
    初始化 Swagger 文档
    
    Args:
        app: Flask 应用实例
    """
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/docs"  # Swagger UI 访问路径
    }
    
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "漫画翻译器 API",
            "description": "AI驱动的漫画图像翻译系统 RESTful API 文档",
            "version": "1.0.0",
            "contact": {
                "name": "Saber-Translator",
                "url": "https://github.com/yourusername/Saber-Translator"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "host": "127.0.0.1:5000",  # 修改为你的主机
        "basePath": "/",
        "schemes": [
            "http"
        ],
        "tags": [
            {
                "name": "翻译",
                "description": "图像翻译相关接口"
            },
            {
                "name": "会话",
                "description": "会话管理接口"
            },
            {
                "name": "系统",
                "description": "系统信息和插件管理"
            },
            {
                "name": "配置",
                "description": "配置管理接口"
            }
        ]
    }
    
    Swagger(app, config=swagger_config, template=swagger_template)
    
    return app


# Swagger 文档示例 - 用于在各个API端点中使用

TRANSLATE_IMAGE_SPEC = {
    "tags": ["翻译"],
    "summary": "翻译漫画图像",
    "description": "上传漫画图像，执行气泡检测、OCR识别、文本翻译和渲染",
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Base64编码的图像数据",
                        "example": "data:image/png;base64,iVBORw0KG..."
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言",
                        "default": "zh",
                        "enum": ["zh", "en", "ja"]
                    },
                    "source_language": {
                        "type": "string",
                        "description": "源语言",
                        "default": "japan"
                    },
                    "model_provider": {
                        "type": "string",
                        "description": "翻译模型提供商",
                        "default": "siliconflow",
                        "enum": ["siliconflow", "deepseek", "ollama", "sakura", "baidu_translate"]
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API密钥（部分服务需要）"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "模型名称"
                    },
                    "fontSize": {
                        "type": "integer",
                        "description": "字体大小",
                        "default": 25
                    },
                    "fontFamily": {
                        "type": "string",
                        "description": "字体文件名"
                    },
                    "use_inpainting": {
                        "type": "boolean",
                        "description": "是否使用智能修复",
                        "default": False
                    },
                    "use_lama": {
                        "type": "boolean",
                        "description": "是否使用LAMA修复",
                        "default": False
                    }
                },
                "required": ["image"]
            }
        }
    ],
    "responses": {
        "200": {
            "description": "翻译成功",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "example": True
                    },
                    "translated_image": {
                        "type": "string",
                        "description": "Base64编码的翻译后图像"
                    },
                    "original_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "原始文本列表"
                    },
                    "translated_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "翻译后文本列表"
                    },
                    "processing_time": {
                        "type": "number",
                        "description": "处理耗时（秒）"
                    }
                }
            }
        },
        "400": {
            "description": "请求参数错误"
        },
        "500": {
            "description": "服务器内部错误"
        }
    }
}

SESSION_LIST_SPEC = {
    "tags": ["会话"],
    "summary": "列出所有会话",
    "description": "获取所有已保存的会话列表",
    "responses": {
        "200": {
            "description": "成功",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "sessions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "created_time": {"type": "string"},
                                "modified_time": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
}

PLUGIN_LIST_SPEC = {
    "tags": ["系统"],
    "summary": "列出所有插件",
    "description": "获取所有已安装的插件及其状态",
    "responses": {
        "200": {
            "description": "成功",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "plugins": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "version": {"type": "string"},
                                "enabled": {"type": "boolean"},
                                "description": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
}
