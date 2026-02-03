"""
Manga Insight 配置 API

处理配置的获取和保存。
"""

import logging
import json
import os
import asyncio
from flask import request, jsonify

from . import manga_insight_bp
from src.core.manga_insight.config_utils import (
    load_insight_config,
    save_insight_config,
    update_provider_config,
    set_current_provider
)
from src.core.manga_insight.config_models import (
    MangaInsightConfig, 
    ARCHITECTURE_PRESETS,
    DEFAULT_BATCH_ANALYSIS_PROMPT,
    DEFAULT_SEGMENT_SUMMARY_PROMPT,
    DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT,
    DEFAULT_QA_SYSTEM_PROMPT,
)

logger = logging.getLogger("MangaInsight.API.Config")


@manga_insight_bp.route('/config/architecture-presets', methods=['GET'])
def get_architecture_presets():
    """获取可用的分析架构预设"""
    try:
        return jsonify({
            "success": True,
            "presets": ARCHITECTURE_PRESETS
        })
    except Exception as e:
        logger.error(f"获取架构预设失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config', methods=['GET'])
def get_config():
    """获取 Manga Insight 配置"""
    try:
        config = load_insight_config()
        return jsonify({
            "success": True,
            "config": config.to_dict()
        })
    except Exception as e:
        logger.error(f"获取配置失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config', methods=['POST'])
def save_config():
    """保存 Manga Insight 配置"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "无效的配置数据"
            }), 400
        
        # 从字典创建配置对象
        config = MangaInsightConfig.from_dict(data)
        
        # 保存配置
        success = save_insight_config(config)
        
        return jsonify({
            "success": success,
            "message": "配置已保存" if success else "保存失败"
        })
    except Exception as e:
        logger.error(f"保存配置失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config/<config_type>/provider/<provider>', methods=['POST'])
def save_provider_config(config_type: str, provider: str):
    """
    保存指定服务商的配置
    
    Args:
        config_type: 配置类型 (vlm, embedding, reranker)
        provider: 服务商名称
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                "success": False,
                "error": "无效的配置数据"
            }), 400
        
        success = update_provider_config(config_type, provider, data)
        
        return jsonify({
            "success": success,
            "message": f"{provider} 配置已保存" if success else "保存失败"
        })
    except Exception as e:
        logger.error(f"保存服务商配置失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config/<config_type>/current', methods=['POST'])
def set_current(config_type: str):
    """
    设置当前选择的服务商
    
    Args:
        config_type: 配置类型
    """
    try:
        data = request.json
        provider = data.get("provider")
        
        if not provider:
            return jsonify({
                "success": False,
                "error": "未指定服务商"
            }), 400
        
        success = set_current_provider(config_type, provider)
        
        return jsonify({
            "success": success,
            "message": f"已切换到 {provider}" if success else "切换失败"
        })
    except Exception as e:
        logger.error(f"设置当前服务商失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config/test/vlm', methods=['POST'])
def test_vlm_connection():
    """测试 VLM 连接"""
    async def _test():
        from src.core.manga_insight.vlm_client import VLMClient
        from src.core.manga_insight.config_models import VLMConfig
        
        data = request.json or {}
        config = VLMConfig.from_dict(data) if data else load_insight_config().vlm
        
        client = VLMClient(config)
        success = await client.test_connection()
        await client.close()
        return success
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(_test())
        loop.close()
        
        return jsonify({
            "success": success,
            "message": "连接成功" if success else "连接失败"
        })
    except Exception as e:
        logger.error(f"测试 VLM 连接失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config/test/embedding', methods=['POST'])
def test_embedding_connection():
    """测试 Embedding 连接"""
    async def _test():
        from src.core.manga_insight.embedding_client import EmbeddingClient
        from src.core.manga_insight.config_models import EmbeddingConfig
        
        data = request.json or {}
        config = EmbeddingConfig.from_dict(data) if data else load_insight_config().embedding
        
        client = EmbeddingClient(config)
        success = await client.test_connection()
        await client.close()
        return success
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(_test())
        loop.close()
        
        return jsonify({
            "success": success,
            "message": "连接成功" if success else "连接失败"
        })
    except Exception as e:
        logger.error(f"测试 Embedding 连接失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config/test/llm', methods=['POST'])
def test_llm_connection():
    """测试 LLM（对话模型）连接"""
    async def _test():
        from src.core.manga_insight.embedding_client import ChatClient
        from src.core.manga_insight.config_models import ChatLLMConfig
        
        data = request.json or {}
        config = ChatLLMConfig.from_dict(data) if data else load_insight_config().chat_llm
        
        client = ChatClient(config)
        success = await client.test_connection()
        await client.close()
        return success
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(_test())
        loop.close()
        
        return jsonify({
            "success": success,
            "message": "连接成功" if success else "连接失败"
        })
    except Exception as e:
        logger.error(f"测试 LLM 连接失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/config/test/reranker', methods=['POST'])
def test_reranker_connection():
    """测试 Reranker 连接"""
    async def _test():
        from src.core.manga_insight.reranker_client import RerankerClient
        from src.core.manga_insight.config_models import RerankerConfig
        
        data = request.json or {}
        config = RerankerConfig.from_dict(data) if data else load_insight_config().reranker
        
        client = RerankerClient(config)
        success = await client.test_connection()
        await client.close()
        return success
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(_test())
        loop.close()
        
        return jsonify({
            "success": success,
            "message": "连接成功" if success else "连接失败"
        })
    except Exception as e:
        logger.error(f"测试 Reranker 连接失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==================== 提示词库 API ====================

def get_prompts_library_path():
    """获取提示词库文件路径"""
    from src.shared.path_helpers import resource_path
    config_dir = resource_path("config")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "manga_insight_prompts_library.json")


@manga_insight_bp.route('/prompts/library', methods=['GET'])
def get_prompts_library():
    """获取提示词库"""
    try:
        library_path = get_prompts_library_path()
        
        if os.path.exists(library_path):
            with open(library_path, 'r', encoding='utf-8') as f:
                library = json.load(f)
        else:
            library = []
        
        return jsonify({
            "success": True,
            "library": library
        })
    except Exception as e:
        logger.error(f"获取提示词库失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "library": []
        }), 500


@manga_insight_bp.route('/prompts/library', methods=['POST'])
def add_prompt_to_library():
    """添加提示词到库"""
    try:
        new_prompt = request.json
        
        # 验证数据
        if not new_prompt or not new_prompt.get('id') or not new_prompt.get('name'):
            return jsonify({
                "success": False,
                "error": "提示词数据无效"
            }), 400
        
        library_path = get_prompts_library_path()
        
        # 加载现有库
        if os.path.exists(library_path):
            with open(library_path, 'r', encoding='utf-8') as f:
                library = json.load(f)
        else:
            library = []
        
        # 添加新提示词
        library.append(new_prompt)
        
        # 保存
        with open(library_path, 'w', encoding='utf-8') as f:
            json.dump(library, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": True,
            "message": "提示词已添加到库"
        })
    except Exception as e:
        logger.error(f"添加提示词到库失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/prompts/library/<prompt_id>', methods=['DELETE'])
def delete_prompt_from_library(prompt_id):
    """从库中删除提示词"""
    try:
        library_path = get_prompts_library_path()
        
        if not os.path.exists(library_path):
            return jsonify({
                "success": False,
                "error": "提示词库不存在"
            }), 404
        
        with open(library_path, 'r', encoding='utf-8') as f:
            library = json.load(f)
        
        # 过滤掉要删除的
        library = [p for p in library if p.get('id') != prompt_id]
        
        with open(library_path, 'w', encoding='utf-8') as f:
            json.dump(library, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": True,
            "message": "提示词已删除"
        })
    except Exception as e:
        logger.error(f"删除提示词失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/prompts/library/import', methods=['POST'])
def import_prompts_library():
    """导入提示词库"""
    try:
        data = request.json
        library = data.get('library', [])
        library_path = get_prompts_library_path()
        
        with open(library_path, 'w', encoding='utf-8') as f:
            json.dump(library, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "success": True,
            "message": f"已导入 {len(library)} 个提示词"
        })
    except Exception as e:
        logger.error(f"导入提示词库失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@manga_insight_bp.route('/prompts/defaults', methods=['GET'])
def get_default_prompts():
    """获取默认提示词"""
    try:
        return jsonify({
            "success": True,
            "prompts": {
                "batch_analysis": DEFAULT_BATCH_ANALYSIS_PROMPT,
                "segment_summary": DEFAULT_SEGMENT_SUMMARY_PROMPT,
                "chapter_summary": DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT,
                "qa_response": DEFAULT_QA_SYSTEM_PROMPT,
            }
        })
    except Exception as e:
        logger.error(f"获取默认提示词失败: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
