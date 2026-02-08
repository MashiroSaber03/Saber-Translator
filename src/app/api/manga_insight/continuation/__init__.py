"""
Manga Insight 续写功能 API 子模块

拆分 continuation_routes.py 为多个功能模块，提高可维护性。

模块结构:
- file_routes.py: 通用文件服务
- character_routes.py: 角色和形态管理
- story_routes.py: 剧情生成（脚本、页面详情）
- image_routes.py: 图片生成和参考图管理
- export_routes.py: 导出功能（图片、PDF）
"""

from flask import Blueprint

# 导入主蓝图
from .. import manga_insight_bp

# 导入所有子模块路由（注册到主蓝图）
from . import file_routes
from . import character_routes
from . import story_routes
from . import image_routes
from . import export_routes

__all__ = ['manga_insight_bp']
