"""
包含所有 Flask 路由定义的模块
用于处理 Web 界面路由和基本页面渲染
"""

from flask import render_template, send_from_directory
import os
# 导入配置加载函数和常量 (用于加载提示词列表)
from src.shared.config_loader import load_json_config
from src.shared import constants
# 导入路径辅助函数
from src.shared.path_helpers import resource_path
# 导入蓝图实例
from . import main_bp

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

@main_bp.route('/')
def index():
    prompts = load_prompts()
    prompt_names = [prompt['name'] for prompt in prompts['saved_prompts']]
    default_prompt_content = get_default_prompt_content()
    textbox_prompts = load_textbox_prompts()
    textbox_prompt_names = [prompt['name'] for prompt in textbox_prompts['saved_prompts']]
    default_textbox_prompt_content = get_default_textbox_prompt_content()
    return render_template('index.html', prompt_names=prompt_names, default_prompt_content=default_prompt_content, 
                           textbox_prompt_names=textbox_prompt_names, default_textbox_prompt_content=default_textbox_prompt_content)

@main_bp.route('/test_lama_page')
def test_lama_page():
    """显示LAMA测试页面"""
    return render_template('test_lama.html')

@main_bp.route('/pic/<path:filename>')
def serve_pic(filename):
    pic_dir = resource_path('pic')  # 获取pic目录的绝对路径
    print(f"Serving file from {pic_dir}: {filename}")
    return send_from_directory(pic_dir, filename)