"""
包含所有 Flask 路由定义的模块
用于处理 Web 界面路由和基本页面渲染

支持两种前端模式：
1. 传统模式（Jinja2 模板）- 默认
2. Vue SPA 模式 - 通过环境变量 USE_VUE_FRONTEND=true 启用
"""

from flask import render_template, send_from_directory, request
import os
# 导入配置加载函数和常量 (用于加载提示词列表)
from src.shared.config_loader import load_json_config
from src.shared import constants
# 导入路径辅助函数
from src.shared.path_helpers import resource_path
# 导入蓝图实例
from . import main_bp

# 检查是否启用 Vue 前端模式
# 默认启用 Vue 前端，可通过环境变量 USE_VUE_FRONTEND=false 禁用（切换回原版）
USE_VUE_FRONTEND = os.environ.get('USE_VUE_FRONTEND', 'true').lower() == 'true'

# 辅助函数
def load_prompts():
    PROMPTS_FILE = constants.PROMPTS_FILE
    # 默认提示词可以考虑移到 constants.py
    
    # 使用新的加载函数，并提供详细的默认结构
    default_data = {"default_prompt": constants.DEFAULT_PROMPT, "saved_prompts": []}
    prompt_data = load_json_config(PROMPTS_FILE, default_value=default_data)
    # 确保返回的数据结构完整，即使文件为空或部分损坏
    if not isinstance(prompt_data, dict):
        return default_data
    if 'default_prompt' not in prompt_data:
        prompt_data['default_prompt'] = constants.DEFAULT_PROMPT
    if 'saved_prompts' not in prompt_data or not isinstance(prompt_data['saved_prompts'], list):
        prompt_data['saved_prompts'] = []
    return prompt_data

def get_default_prompt_content():
    prompts = load_prompts()
    return prompts.get('default_prompt', constants.DEFAULT_PROMPT)

def load_textbox_prompts():
    TEXTBOX_PROMPTS_FILE = constants.TEXTBOX_PROMPTS_FILE
    
    # 使用新的加载函数，并提供详细的默认结构
    default_data = {"default_prompt": constants.DEFAULT_TEXTBOX_PROMPT, "saved_prompts": []}
    prompt_data = load_json_config(TEXTBOX_PROMPTS_FILE, default_value=default_data)
    # 确保返回的数据结构完整
    if not isinstance(prompt_data, dict):
        return default_data
    if 'default_prompt' not in prompt_data:
        prompt_data['default_prompt'] = constants.DEFAULT_TEXTBOX_PROMPT
    if 'saved_prompts' not in prompt_data or not isinstance(prompt_data['saved_prompts'], list):
        prompt_data['saved_prompts'] = []
    return prompt_data

def get_default_textbox_prompt_content():
    prompts = load_textbox_prompts()
    return prompts.get('default_prompt', constants.DEFAULT_TEXTBOX_PROMPT)

# 路由处理函数

def serve_vue_app():
    """
    服务 Vue SPA 应用
    返回 Vue 构建的 index.html，由 Vue Router 处理前端路由
    """
    vue_dist_path = resource_path('src/app/static/vue')
    return send_from_directory(vue_dist_path, 'index.html')


@main_bp.route('/')
def bookshelf():
    """书架首页 - 显示所有书籍"""
    if USE_VUE_FRONTEND:
        return serve_vue_app()
    return render_template('bookshelf.html')


@main_bp.route('/reader')
def reader():
    """阅读页面 - 竖向流式阅读漫画"""
    if USE_VUE_FRONTEND:
        return serve_vue_app()
    return render_template('reader.html')


@main_bp.route('/translate')
def translate():
    """翻译页面 - 支持书籍/章节参数"""
    if USE_VUE_FRONTEND:
        return serve_vue_app()
    prompts = load_prompts()
    prompt_names = [prompt['name'] for prompt in prompts['saved_prompts']]
    default_prompt_content = get_default_prompt_content()
    textbox_prompts = load_textbox_prompts()
    textbox_prompt_names = [prompt['name'] for prompt in textbox_prompts['saved_prompts']]
    default_textbox_prompt_content = get_default_textbox_prompt_content()
    # 高质量翻译和AI视觉OCR的默认提示词
    default_hq_prompt = constants.DEFAULT_HQ_TRANSLATE_PROMPT
    default_ai_vision_ocr_prompt = constants.DEFAULT_AI_VISION_OCR_PROMPT
    return render_template('index.html', prompt_names=prompt_names, default_prompt_content=default_prompt_content, 
                           textbox_prompt_names=textbox_prompt_names, default_textbox_prompt_content=default_textbox_prompt_content,
                           default_hq_prompt=default_hq_prompt, default_ai_vision_ocr_prompt=default_ai_vision_ocr_prompt)


@main_bp.route('/insight')
def manga_insight():
    """漫画分析页面 - 智能分析漫画内容"""
    if USE_VUE_FRONTEND:
        return serve_vue_app()
    return render_template('manga_insight.html')


@main_bp.route('/pic/<path:filename>')
def serve_pic(filename):
    """服务 pic 目录下的图片资源"""
    pic_dir = resource_path('pic')
    return send_from_directory(pic_dir, filename)


# Vue SPA 静态资源路由
@main_bp.route('/static/vue/<path:filename>')
def serve_vue_static(filename):
    """
    服务 Vue 构建的静态资源
    包括 JS、CSS、图片等资源文件
    """
    vue_dist_path = resource_path('src/app/static/vue')
    return send_from_directory(vue_dist_path, filename)


# Vue SPA 通配路由 - 处理所有未匹配的前端路由
# 注意：此路由优先级较低，只有在其他路由都不匹配时才会触发
@main_bp.route('/<path:path>')
def catch_all(path):
    """
    通配路由 - 用于 Vue SPA 的客户端路由支持
    
    当启用 Vue 前端时，所有未匹配的路由都返回 Vue 的 index.html
    由 Vue Router 在客户端处理路由
    
    注意：API 路由（/api/*）不会被此路由捕获，因为它们有更高的优先级
    """
    # 排除 API 路由和静态资源路由
    if path.startswith('api/') or path.startswith('static/'):
        # 返回 404，让 Flask 处理
        from flask import abort
        abort(404)
    
    # 如果启用了 Vue 前端，返回 Vue 应用
    if USE_VUE_FRONTEND:
        return serve_vue_app()
    
    # 否则返回 404
    from flask import abort
    abort(404)