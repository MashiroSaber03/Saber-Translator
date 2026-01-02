"""
包含所有 Flask 路由定义的模块
用于处理 Web 界面路由和基本页面渲染

Vue SPA 前端模式 - 所有页面路由由 Vue Router 处理
"""

from flask import send_from_directory
# 导入路径辅助函数
from src.shared.path_helpers import resource_path
# 导入蓝图实例
from . import main_bp


def serve_vue_app():
    """
    服务 Vue SPA 应用
    返回 Vue 构建的 index.html，由 Vue Router 处理前端路由
    """
    vue_dist_path = resource_path('src/app/static/vue')
    return send_from_directory(vue_dist_path, 'index.html')


@main_bp.route('/')
def index():
    """首页 - Vue SPA 入口"""
    return serve_vue_app()


@main_bp.route('/reader')
def reader():
    """阅读页面"""
    return serve_vue_app()


@main_bp.route('/translate')
def translate():
    """翻译页面"""
    return serve_vue_app()


@main_bp.route('/insight')
def manga_insight():
    """漫画分析页面"""
    return serve_vue_app()


@main_bp.route('/pic/<path:filename>')
def serve_pic(filename):
    """服务 pic 目录下的图片资源"""
    pic_dir = resource_path('pic')
    return send_from_directory(pic_dir, filename)


# Vue SPA 静态资源路由（base='/'）
@main_bp.route('/js/<path:filename>')
def serve_vue_js(filename):
    """服务 Vue 构建的 JS 文件"""
    vue_dist_path = resource_path('src/app/static/vue/js')
    return send_from_directory(vue_dist_path, filename)


@main_bp.route('/assets/<path:filename>')
def serve_vue_assets(filename):
    """服务 Vue 构建的资源文件（CSS、图片等）"""
    vue_dist_path = resource_path('src/app/static/vue/assets')
    return send_from_directory(vue_dist_path, filename)


# Vue SPA 通配路由 - 处理所有未匹配的前端路由
@main_bp.route('/<path:path>')
def catch_all(path):
    """
    通配路由 - 用于 Vue SPA 的客户端路由支持
    
    所有未匹配的路由都返回 Vue 的 index.html
    由 Vue Router 在客户端处理路由
    
    注意：API 路由（/api/*）不会被此路由捕获，因为它们有更高的优先级
    """
    # 排除 API 路由和静态资源路由
    if path.startswith('api/') or path.startswith('static/'):
        from flask import abort
        abort(404)
    
    # 返回 Vue 应用
    return serve_vue_app()