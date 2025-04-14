import json
import yaml
import os
import logging

# 获取当前脚本所在的目录，然后向上两级找到项目根目录
# 这假设 config_loader.py 在 src/shared/ 下
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_DIR = os.path.join(project_root, 'config')

logger = logging.getLogger("ConfigLoader")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_config_path(filename):
    """获取配置文件的绝对路径"""
    return os.path.join(CONFIG_DIR, filename)

def load_json_config(filename, default_value={}):
    """
    加载指定名称的 JSON 配置文件。

    Args:
        filename (str): 配置文件名 (例如 'prompts.json').
        default_value (any): 如果文件不存在或解析失败时返回的默认值。

    Returns:
        dict or list or any: 解析后的配置数据或默认值。
    """
    config_path = get_config_path(filename)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"成功加载 JSON 配置文件: {config_path}")
                return data
        else:
            logger.warning(f"JSON 配置文件未找到: {config_path}，返回默认值。")
            # 如果文件不存在，可以考虑创建一个空的默认文件
            # save_json_config(filename, default_value)
            return default_value
    except json.JSONDecodeError as e:
        logger.error(f"解析 JSON 配置文件失败: {config_path} - {e}", exc_info=True)
        return default_value
    except Exception as e:
        logger.error(f"加载 JSON 配置文件时发生未知错误: {config_path} - {e}", exc_info=True)
        return default_value

def save_json_config(filename, data):
    """
    保存数据到指定的 JSON 配置文件。

    Args:
        filename (str): 配置文件名 (例如 'prompts.json').
        data (dict or list): 要保存的数据。

    Returns:
        bool: 保存是否成功。
    """
    config_path = get_config_path(filename)
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"成功保存 JSON 配置文件: {config_path}")
        return True
    except Exception as e:
        logger.error(f"保存 JSON 配置文件失败: {config_path} - {e}", exc_info=True)
        return False

def load_yaml_config(filename, default_value={}):
    """
    加载指定名称的 YAML 配置文件。

    Args:
        filename (str): 配置文件名 (例如 'settings.yaml').
        default_value (any): 如果文件不存在或解析失败时返回的默认值。

    Returns:
        dict or any: 解析后的配置数据或默认值。
    """
    config_path = get_config_path(filename)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                # 使用 SafeLoader 防止执行任意代码
                data = yaml.safe_load(f)
                logger.info(f"成功加载 YAML 配置文件: {config_path}")
                # 如果加载结果为 None（空文件），返回默认值
                return data if data is not None else default_value
        else:
            logger.warning(f"YAML 配置文件未找到: {config_path}，返回默认值。")
            # 如果文件不存在，可以考虑创建一个空的默认文件
            # save_yaml_config(filename, default_value)
            return default_value
    except yaml.YAMLError as e:
        logger.error(f"解析 YAML 配置文件失败: {config_path} - {e}", exc_info=True)
        return default_value
    except Exception as e:
        logger.error(f"加载 YAML 配置文件时发生未知错误: {config_path} - {e}", exc_info=True)
        return default_value

def save_yaml_config(filename, data):
    """
    保存数据到指定的 YAML 配置文件。

    Args:
        filename (str): 配置文件名 (例如 'settings.yaml').
        data (dict): 要保存的数据。

    Returns:
        bool: 保存是否成功。
    """
    config_path = get_config_path(filename)
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"成功保存 YAML 配置文件: {config_path}")
        return True
    except Exception as e:
        logger.error(f"保存 YAML 配置文件失败: {config_path} - {e}", exc_info=True)
        return False

# --- 测试代码 ---
if __name__ == '__main__':
    print(f"项目根目录: {project_root}")
    print(f"配置目录: {CONFIG_DIR}")

    # 测试加载 JSON
    print("\n--- 测试加载 JSON ---")
    prompts_data = load_json_config('prompts.json', {"default_prompt": "默认", "saved_prompts": []})
    print("加载 prompts.json:", prompts_data)

    history_data = load_json_config('model_history.json')
    print("加载 model_history.json:", history_data)

    # 测试加载不存在的 JSON
    non_exist_json = load_json_config('non_existent.json', {'error': True})
    print("加载 non_existent.json:", non_exist_json)

    # 测试加载 YAML
    print("\n--- 测试加载 YAML ---")
    settings_data = load_yaml_config('settings.yaml', {'default_setting': 'value'})
    print("加载 settings.yaml:", settings_data)

    # 测试加载不存在的 YAML
    non_exist_yaml = load_yaml_config('non_existent.yaml', {'error': True})
    print("加载 non_existent.yaml:", non_exist_yaml)

    # 测试保存 JSON (可选，会覆盖文件)
    # print("\n--- 测试保存 JSON ---")
    # test_save_data = {"test": "data", "value": 123}
    # save_json_config('test_save.json', test_save_data)
    # loaded_test_save = load_json_config('test_save.json')
    # print("加载 test_save.json:", loaded_test_save)

    # 测试保存 YAML (可选，会覆盖文件)
    # print("\n--- 测试保存 YAML ---")
    # test_save_yaml = {"yaml_test": "数据", "number": 456}
    # save_yaml_config('test_save.yaml', test_save_yaml)
    # loaded_test_yaml = load_yaml_config('test_save.yaml')
    # print("加载 test_save.yaml:", loaded_test_yaml)