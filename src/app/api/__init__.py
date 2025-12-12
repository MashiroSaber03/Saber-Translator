"""
API 模块初始化文件
"""

# 导入所有API蓝图
from .config_api import config_bp
from .session_api import session_bp
from .bookshelf_api import bookshelf_bp  # ✨ 书架 API

# 使用新的模块化蓝图
from .system import system_bp          # ✨ 模块化 system API
from .translation import translate_bp  # ✨ 模块化 translation API

# 导入漫画分析蓝图
from .manga_insight import manga_insight_bp  # ✨ 漫画智能分析 API

# 这个列表将在应用初始化时被导入和注册
all_blueprints = [translate_bp, config_bp, system_bp, session_bp, bookshelf_bp, manga_insight_bp]