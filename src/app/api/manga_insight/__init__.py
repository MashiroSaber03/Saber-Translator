"""
Manga Insight API 模块

提供漫画分析相关的 REST API。
"""

from flask import Blueprint

# 创建蓝图
manga_insight_bp = Blueprint('manga_insight', __name__, url_prefix='/api/manga-insight')

# 导入路由
from . import config_routes
from . import analysis_routes
from . import chat_routes
from . import reanalyze_routes
from . import data_routes

# 续写功能路由（已拆分为子模块）
from . import continuation  # 导入 continuation 子模块，自动注册所有子路由
