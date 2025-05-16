"""
API 模块初始化文件
"""

# 导入所有API蓝图
from .translate_api import translate_bp
from .config_api import config_bp
from .system_api import system_bp
from .session_api import session_bp

# 这个列表将在应用初始化时被导入和注册
all_blueprints = [translate_bp, config_bp, system_bp, session_bp]