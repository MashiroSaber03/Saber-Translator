"""
包含与配置相关的API端点
"""

from flask import Blueprint, request, jsonify # 已有
# 导入配置加载/保存函数和常量
from src.shared.config_loader import load_json_config, save_json_config
from src.shared import constants
import logging # 需要 logging

# 获取 logger
logger = logging.getLogger("ConfigAPI")

# 定义蓝图实例 (已在步骤 2 定义)
config_bp = Blueprint('config_api', __name__, url_prefix='/api')

# --- 需要加载/保存配置的辅助函数 ---
# (这些函数原本在 app.py，现在移到这里)

def load_model_info():
    return load_json_config(constants.MODEL_HISTORY_FILE, default_value={})

def load_prompts():
    default_data = {"default_prompt": constants.DEFAULT_PROMPT, "saved_prompts": []}
    prompt_data = load_json_config(constants.PROMPTS_FILE, default_value=default_data)
    if not isinstance(prompt_data, dict): return default_data
    if 'default_prompt' not in prompt_data: prompt_data['default_prompt'] = constants.DEFAULT_PROMPT
    if 'saved_prompts' not in prompt_data or not isinstance(prompt_data['saved_prompts'], list): prompt_data['saved_prompts'] = []
    return prompt_data

def save_prompts(prompt_data):
    success = save_json_config(constants.PROMPTS_FILE, prompt_data)
    if not success: logger.warning(f"保存提示词信息失败: {constants.PROMPTS_FILE}")

def load_textbox_prompts():
    default_data = {"default_prompt": constants.DEFAULT_TEXTBOX_PROMPT, "saved_prompts": []}
    prompt_data = load_json_config(constants.TEXTBOX_PROMPTS_FILE, default_value=default_data)
    if not isinstance(prompt_data, dict): return default_data
    if 'default_prompt' not in prompt_data: prompt_data['default_prompt'] = constants.DEFAULT_TEXTBOX_PROMPT
    if 'saved_prompts' not in prompt_data or not isinstance(prompt_data['saved_prompts'], list): prompt_data['saved_prompts'] = []
    return prompt_data

def save_textbox_prompts(prompt_data):
    success = save_json_config(constants.TEXTBOX_PROMPTS_FILE, prompt_data)
    if not success: logger.warning(f"保存文本框提示词信息失败: {constants.TEXTBOX_PROMPTS_FILE}")
# ------------------------------------

@config_bp.route('/get_used_models', methods=['GET'])
def get_used_models():
    model_provider = request.args.get('model_provider')
    if not model_provider:
        return jsonify({'error': '缺少 model_provider 参数'}), 400

    model_info = load_model_info()
    used_models = model_info.get(model_provider, [])
    return jsonify({'models': used_models})

@config_bp.route('/get_model_info', methods=['GET'])
def get_model_info():
    model_info = load_model_info()
    return jsonify(model_info)

@config_bp.route('/save_model_info', methods=['POST'])
def save_model_info_api():
    data = request.get_json()
    if not data or 'modelProvider' not in data or 'modelName' not in data:
        return jsonify({'error': '缺少模型供应商或模型名称'}), 400

    model_provider = data['modelProvider']
    model_name = data['modelName']
    
    # 保存模型信息
    model_info = load_model_info()
    if model_provider not in model_info:
        model_info[model_provider] = []

    if model_name and model_name not in model_info[model_provider]:
        model_info[model_provider].insert(0, model_name)
        model_info[model_provider] = model_info[model_provider][:constants.MAX_MODEL_HISTORY]

    success = save_json_config(constants.MODEL_HISTORY_FILE, model_info)
    if not success:
        logger.warning(f"保存模型历史信息失败: {constants.MODEL_HISTORY_FILE}")
    
    return jsonify({'message': '模型信息保存成功'})

@config_bp.route('/get_prompts', methods=['GET'])
def get_prompts():
    prompts = load_prompts()
    prompt_names = [prompt['name'] for prompt in prompts['saved_prompts']]
    default_prompt_content = prompts.get('default_prompt', constants.DEFAULT_PROMPT)
    return jsonify({'prompt_names': prompt_names, 'default_prompt_content': default_prompt_content})

@config_bp.route('/save_prompt', methods=['POST'])
def save_prompt():
    data = request.get_json()
    if not data or 'prompt_name' not in data or 'prompt_content' not in data:
        return jsonify({'error': '缺少提示词名称或内容'}), 400

    prompt_name = data['prompt_name']
    prompt_content = data['prompt_content']

    prompts = load_prompts()
    existing_prompt_index = next((index for (index, d) in enumerate(prompts['saved_prompts']) if d["name"] == prompt_name), None)
    if existing_prompt_index is not None:
        prompts['saved_prompts'][existing_prompt_index]['content'] = prompt_content
    else:
        prompts['saved_prompts'].append({'name': prompt_name, 'content': prompt_content})

    save_prompts(prompts)
    return jsonify({'message': '提示词保存成功'})

@config_bp.route('/get_prompt_content', methods=['GET'])
def get_prompt_content():
    prompt_name = request.args.get('prompt_name')
    if not prompt_name:
        return jsonify({'error': '缺少提示词名称'}), 400

    prompts = load_prompts()
    if prompt_name == constants.DEFAULT_PROMPT_NAME:
        prompt_content = prompts.get('default_prompt', constants.DEFAULT_PROMPT)
    else:
        saved_prompt = next((prompt for prompt in prompts['saved_prompts'] if prompt['name'] == prompt_name), None)
        prompt_content = saved_prompt['content'] if saved_prompt else None

    if prompt_content:
        return jsonify({'prompt_content': prompt_content})
    else:
        return jsonify({'error': '提示词未找到'}), 404

@config_bp.route('/reset_prompt_to_default', methods=['POST'])
def reset_prompt_to_default():
    prompts = load_prompts()
    prompts['default_prompt'] = constants.DEFAULT_PROMPT
    save_prompts(prompts)
    return jsonify({'message': '提示词已重置为默认'})

@config_bp.route('/delete_prompt', methods=['POST'])
def delete_prompt():
    data = request.get_json()
    if not data or 'prompt_name' not in data:
        return jsonify({'error': '缺少提示词名称'}), 400

    prompt_name = data['prompt_name']
    prompts = load_prompts()
    prompts['saved_prompts'] = [prompt for prompt in prompts['saved_prompts'] if prompt['name'] != prompt_name]
    save_prompts(prompts)
    return jsonify({'message': '提示词删除成功'})

@config_bp.route('/get_textbox_prompts', methods=['GET'])
def get_textbox_prompts():
    prompts = load_textbox_prompts()
    prompt_names = [prompt['name'] for prompt in prompts['saved_prompts']]
    default_prompt_content = prompts.get('default_prompt', constants.DEFAULT_TEXTBOX_PROMPT)
    return jsonify({'prompt_names': prompt_names, 'default_prompt_content': default_prompt_content})

@config_bp.route('/save_textbox_prompt', methods=['POST'])
def save_textbox_prompt():
    data = request.get_json()
    if not data or 'prompt_name' not in data or 'prompt_content' not in data:
        return jsonify({'error': '缺少提示词名称或内容'}), 400

    prompt_name = data['prompt_name']
    prompt_content = data['prompt_content']

    prompts = load_textbox_prompts()
    existing_prompt_index = next((index for (index, d) in enumerate(prompts['saved_prompts']) if d["name"] == prompt_name), None)
    if existing_prompt_index is not None:
        prompts['saved_prompts'][existing_prompt_index]['content'] = prompt_content
    else:
        prompts['saved_prompts'].append({'name': prompt_name, 'content': prompt_content})

    save_textbox_prompts(prompts)
    return jsonify({'message': '文本框提示词保存成功'})

@config_bp.route('/get_textbox_prompt_content', methods=['GET'])
def get_textbox_prompt_content():
    prompt_name = request.args.get('prompt_name')
    if not prompt_name:
        return jsonify({'error': '缺少提示词名称'}), 400

    prompts = load_textbox_prompts()
    if prompt_name == constants.DEFAULT_PROMPT_NAME:
        prompt_content = prompts.get('default_prompt', constants.DEFAULT_TEXTBOX_PROMPT)
    else:
        saved_prompt = next((prompt for prompt in prompts['saved_prompts'] if prompt['name'] == prompt_name), None)
        prompt_content = saved_prompt['content'] if saved_prompt else None

    if prompt_content:
        return jsonify({'prompt_content': prompt_content})
    else:
        return jsonify({'error': '文本框提示词未找到'}), 404

@config_bp.route('/reset_textbox_prompt_to_default', methods=['POST'])
def reset_textbox_prompt_to_default():
    prompts = load_textbox_prompts()
    prompts['default_prompt'] = constants.DEFAULT_TEXTBOX_PROMPT
    save_textbox_prompts(prompts)
    return jsonify({'message': '文本框提示词已重置为默认'})

@config_bp.route('/delete_textbox_prompt', methods=['POST'])
def delete_textbox_prompt():
    data = request.get_json()
    if not data or 'prompt_name' not in data:
        return jsonify({'error': '缺少提示词名称'}), 400

    prompt_name = data['prompt_name']
    prompts = load_textbox_prompts()
    prompts['saved_prompts'] = [prompt for prompt in prompts['saved_prompts'] if prompt['name'] != prompt_name]
    save_textbox_prompts(prompts)
    return jsonify({'message': '文本框提示词删除成功'})