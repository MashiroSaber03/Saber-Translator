"""
应用程序包初始化，包含蓝图定义和注册
"""

from flask import Blueprint

# 创建主蓝图实例
main_bp = Blueprint(
    'main',
    __name__,
    template_folder='../../templates',  
    static_folder='../../static'  
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
        print(f"Registered API blueprint: {bp.name} with url_prefix={bp.url_prefix}")
    
    return app