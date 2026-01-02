"""
应用程序包初始化，包含蓝图定义和注册
"""

import logging
from flask import Blueprint

logger = logging.getLogger("AppBlueprints")

# 创建主蓝图实例
main_bp = Blueprint(
    'main',
    __name__
)

# 导入路由定义
from . import routes

# 从 API 模块导入所有蓝图
from .api import all_blueprints

# 定义一个函数用于注册蓝图到Flask应用
def register_blueprints(app):
    # 注册主蓝图
    app.register_blueprint(main_bp)
    
    # 注册所有API蓝图
    for bp in all_blueprints:
        app.register_blueprint(bp)
    
    logger.debug(f"已注册 {len(all_blueprints)} 个 API 蓝图")
    return app