"""
翻译API模块

使用统一的 BubbleState 进行气泡状态管理。
所有翻译、渲染相关的 API 都通过 BubbleState 进行数据交换。
"""

from flask import Blueprint, request, jsonify
import base64
import io
import traceback
import logging
from PIL import Image, ImageDraw, ImageFont

# 导入核心处理函数
from src.core.processing import process_image_translation
from src.core.rendering import (
    re_render_text_in_bubbles, 
    render_single_bubble,
    render_bubbles_unified,
    re_render_with_states,
    render_single_bubble_unified
)
from src.core.translation import translate_single_text
from src.interfaces.lama_interface import LAMA_AVAILABLE

# 导入统一状态模型
from src.core.config_models import (
    TranslationRequest,
    BubbleState,
    bubble_states_to_api_response,
    bubble_states_from_api_request
)

# 导入共享模块
from src.shared import constants
from src.shared.path_helpers import get_font_path
from ..config_api import save_model_info_api

# 获取logger
logger = logging.getLogger("TranslateAPI")

# 创建翻译API蓝图
translate_bp = Blueprint('translate_api', __name__, url_prefix='/api')

# 导入所有路由
from . import routes

__all__ = ['translate_bp']
