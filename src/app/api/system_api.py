"""
包含与系统功能相关的API端点
"""

from flask import Blueprint, request, jsonify # 导入 Blueprint, request, jsonify
# 导入系统相关模块 (os, shutil, requests, pdf processor 等)
import os
import shutil
import requests
import PyPDF2 # 导入 PyPDF2
import base64 # 需要 base64
import io # 需要 io
from PIL import Image, ImageDraw, ImageFont # 需要 Image, ImageDraw 和 ImageFont
import traceback # 需要traceback
import time # 需要time
import logging # 需要logging
import threading # 需要threading

from src.core.pdf_processor import extract_images_from_pdf # 导入 PDF 处理函数
from src.shared.path_helpers import get_debug_dir, resource_path # 需要调试目录函数和路径助手
from src.interfaces.lama_interface import clean_image_with_lama, LAMA_AVAILABLE # 导入LAMA接口
<<<<<<< HEAD
from src.interfaces.baidu_ocr_interface import test_baidu_ocr_connection # 导入百度OCR接口测试方法
from src.interfaces.vision_interface import test_ai_vision_ocr # 导入AI视觉OCR测试函数
from src.interfaces.baidu_translate_interface import baidu_translate # 导入百度翻译接口
from src.plugins.manager import get_plugin_manager # 需要插件管理器
from src.plugins.base import PluginBase # 需要基类来检查类型
from src.shared.image_helpers import base64_to_image # 需要 image_helpers
from src.core.detection import get_bubble_coordinates # 需要 detection
from src.shared import constants # 导入常量
=======
from src.plugins.manager import get_plugin_manager # 需要插件管理器
from src.plugins.base import PluginBase # 需要基类来检查类型
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
# ... 其他需要的导入 ...

# 获取 logger
logger = logging.getLogger("SystemAPI")

system_bp = Blueprint('system_api', __name__, url_prefix='/api')

# --- API 路由函数将在此处定义 (后续步骤迁移) ---
@system_bp.route('/upload_pdf', methods=['POST'])
def upload_pdf_api():
    if 'pdfFile' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    pdf_file = request.files['pdfFile']
    if pdf_file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if pdf_file:
        try:
            # 使用从core模块导入的函数处理PDF
            pdf_bytes = pdf_file.read()
            pdf_stream = io.BytesIO(pdf_bytes)
            
            images = extract_images_from_pdf(pdf_stream)
            
            image_data_list = []
            for i, image in enumerate(images):
                try:
                    buffered = io.BytesIO()
                    # 确保图像是PNG格式
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_data_list.append(img_str)
                except Exception as save_e:
                    print(f"保存图片失败，跳过图片{i+1}: {save_e}")

            return jsonify({'images': image_data_list}), 200
        except Exception as e:
            print(f"处理 PDF 文件时出错: {e}")
            return jsonify({'error': f"处理 PDF 文件时出错: {e}"}), 500

    return jsonify({'error': '上传失败'}), 500

@system_bp.route('/clean_debug_files', methods=['POST'])
def clean_debug_files():
    """清理调试目录中的文件"""
    try:
        debug_dir = get_debug_dir()
        
        # 检查debug目录是否存在
        if os.path.exists(debug_dir):
            # 记录清理前的状态
            files_count = 0
            total_size = 0
            for root, dirs, files in os.walk(debug_dir):
                files_count += len(files)
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
            
            # 以MB为单位的总大小
            total_size_mb = total_size / (1024 * 1024)
            
            # 删除debug目录中的所有内容
            for item in os.listdir(debug_dir):
                item_path = os.path.join(debug_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            # 保留目录结构但清空内容
            os.makedirs(os.path.join(debug_dir, "bubbles"), exist_ok=True)
            
            return jsonify({
                'success': True, 
                'message': f'已清理 {files_count} 个调试文件，释放了 {total_size_mb:.2f}MB 空间'
            })
        else:
            return jsonify({'success': True, 'message': '没有找到需要清理的调试文件'})
    except Exception as e:
        print(f"清理调试文件失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@system_bp.route('/test_ollama_connection', methods=['GET'])
def test_ollama_connection():
    """测试Ollama连接状态的端点"""
    try:
        # 使用requests库，而不是ollama库，确保在打包环境中兼容
        import requests
        import json
        
        # 先检查Ollama服务是否可用
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'message': f'Ollama服务响应异常，状态码: {response.status_code}'
                }), 500
                
            version_info = response.json()
            version = version_info.get('version', '未知')
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'无法连接到Ollama服务，请确认服务是否启动: {str(e)}'
            }), 500
        
        # 获取已安装的模型列表
        try:
            models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if models_response.status_code != 200:
                return jsonify({
                    'success': False,
                    'message': f'获取模型列表失败，状态码: {models_response.status_code}'
                }), 500
                
            models_data = models_response.json()
            models = models_data.get('models', [])
            model_names = [m.get('name', '') for m in models]
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'获取模型列表时出错: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'version': version,
            'models': model_names
        })
        
    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        return jsonify({
            'success': False,
            'message': f'测试Ollama连接时发生错误: {str(e)}',
            'error_details': error_message
        }), 500

@system_bp.route('/test_lama_repair', methods=['GET'])
def test_lama_repair():
    """测试LAMA修复功能的端点"""
    try:
        # 测试图片路径，使用debug目录中已有的图像
        from flask import current_app
        # 使用get_debug_dir获取调试目录
        debug_dir = get_debug_dir()
        test_img_path = os.path.join(debug_dir, "result_image.png")
        if not os.path.exists(test_img_path):
            return jsonify({
                'error': '测试图像不存在',
                'path': test_img_path
            })
        
        # 记录测试开始
        logger.info("开始LAMA修复功能测试")
        
        # 加载测试图像
        image = Image.open(test_img_path).convert("RGB")
        
        # 创建一个简单的掩码
        mask = Image.new("RGB", image.size, color=(0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        # 在图像中央绘制一个白色矩形作为掩码
        width, height = image.size
        rect_width, rect_height = width // 3, height // 3
        left = (width - rect_width) // 2
        top = (height - rect_height) // 2
        draw.rectangle(
            [(left, top), (left + rect_width, top + rect_height)],
            fill=(255, 255, 255)
        )
        
        # 保存掩码供检查
        mask_path = os.path.join(debug_dir, "test_mask.png")
        mask.save(mask_path)
        logger.info(f"保存掩码图像：{mask_path}")
        
        # 确认LAMA可用
        if not LAMA_AVAILABLE:
            return jsonify({
                'error': 'LAMA功能不可用',
                'LAMA_AVAILABLE': LAMA_AVAILABLE
            })
        
        # 使用LAMA执行修复
        logger.info("开始使用LAMA进行修复")
        try:
            repaired_image = clean_image_with_lama(image, mask)
            
            # 保存修复后的图像
            result_path = os.path.join(debug_dir, "test_lama_web_result.png")
            repaired_image.save(result_path)
            logger.info(f"成功保存修复结果：{result_path}")
            
            # 转换图像为base64
            buffered = io.BytesIO()
            repaired_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'message': 'LAMA修复成功',
                'result_image': img_str
            })
        except Exception as e:
            logger.error(f"LAMA修复失败：{e}")
            return jsonify({
                'error': f'LAMA修复失败：{str(e)}'
            })
    except Exception as e:
        logger.error(f"测试端点出错：{e}")
        return jsonify({
            'error': f'测试端点出错：{str(e)}'
        })

@system_bp.route('/test_params', methods=['POST'])
def test_params():
    """测试参数解析功能的端点"""
    try:
        data = request.get_json()
        logger.info(f"收到测试参数：{data}")
        
        # 提取修复方法参数
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)
        
        logger.info(f"修复方法参数：use_inpainting={use_inpainting}, use_lama={use_lama}")
        
        return jsonify({
            'success': True,
            'received_params': data,
            'parsed_params': {
                'use_inpainting': use_inpainting,
                'use_lama': use_lama
            }
        })
    except Exception as e:
        logger.error(f"测试参数端点出错：{e}")
        return jsonify({
            'error': f'测试参数端点出错：{str(e)}'
        })

# 全局变量，用于存储Sakura服务状态和模型列表
SAKURA_STATUS = {
    'available': False,
    'models': [
        "sakura-7b-qwen2.5-v1.0",
        "sakura-14b-qwen2.5-v1.0",
        "sakura-32b-qwen2beta-v0.9"
    ],
    'last_check_time': 0
}

@system_bp.route('/test_sakura_connection', methods=['GET'])
def test_sakura_connection():
    """测试Sakura服务连接状态的端点"""
    try:
        global SAKURA_STATUS
        import requests
        import json
        import time
        
        # 检查是否需要强制刷新模型列表
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        current_time = time.time()
        
        # 如果上次检查时间在30秒内且不是强制刷新，则使用缓存的结果
        if not force_refresh and current_time - SAKURA_STATUS['last_check_time'] < 30 and SAKURA_STATUS['available']:
            print(f"使用缓存的Sakura模型列表: {len(SAKURA_STATUS['models'])}个模型")
            return jsonify({
                'success': True,
                'models': SAKURA_STATUS['models'],
                'cached': True
            })
        
        # 增加重试次数和超时时间
        max_retries = 3
        timeout_seconds = 10
        
        for retry in range(max_retries):
            try:
                # Sakura使用OpenAI兼容的API，尝试获取models列表
                print(f"尝试连接Sakura服务 ({retry+1}/{max_retries})...")
                response = requests.get("http://localhost:8080/v1/models", timeout=timeout_seconds)
                
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get('data', [])
                    model_names = [m.get('id', '') for m in models]
                    
                    # 如果没有获取到模型列表，则使用默认的模型列表
                    if not model_names:
                        model_names = SAKURA_STATUS['models']
                    
                    # 更新全局状态
                    SAKURA_STATUS['available'] = True
                    SAKURA_STATUS['models'] = model_names
                    SAKURA_STATUS['last_check_time'] = current_time
                    
                    print(f"成功连接到Sakura服务，获取到 {len(model_names)} 个模型")
                    return jsonify({
                        'success': True,
                        'models': model_names,
                        'cached': False
                    })
                else:
                    # 如果不是最后一次重试，等待后继续
                    if retry < max_retries - 1:
                        print(f"Sakura服务响应异常，状态码: {response.status_code}，将在2秒后重试")
                        time.sleep(2)
                    else:
                        # 更新状态为不可用
                        SAKURA_STATUS['available'] = False
                        SAKURA_STATUS['last_check_time'] = current_time
                        return jsonify({
                            'success': False,
                            'message': f'Sakura服务响应异常，状态码: {response.status_code}'
                        }), 500
                    
            except Exception as e:
                # 记录错误但继续重试
                print(f"连接Sakura尝试 {retry+1}/{max_retries} 失败: {e}")
                if retry < max_retries - 1:
                    print("将在2秒后重试...")
                    time.sleep(2)
        
        # 所有重试都失败，更新状态为不可用
        SAKURA_STATUS['available'] = False
        SAKURA_STATUS['last_check_time'] = current_time
        return jsonify({
            'success': False,
            'message': f'无法连接到Sakura服务，请确认服务是否启动'
        }), 500
            
    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        return jsonify({
            'success': False,
            'message': f'测试Sakura连接时发生错误: {str(e)}',
            'error_details': error_message
        }), 500

# 创建Sakura服务可用性检查函数的包装版本
def check_services_availability():
    """后台定期检查Sakura服务可用性"""
    global SAKURA_STATUS
    import requests
    import time
    import logging
    
    logger = logging.getLogger("sakura_service_checker")
    logger.setLevel(logging.INFO)
    
    # 创建一个处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("启动Sakura服务监控线程")
    
    while True:
        try:
            response = requests.get("http://localhost:8080/v1/models", timeout=5)
            if response.status_code == 200:
                # 成功获取模型列表
                models_data = response.json()
                models = models_data.get('data', [])
                model_names = [m.get('id', '') for m in models]
                
                # 如果获取到模型列表为空，使用默认列表
                if not model_names:
                    model_names = [
                        "sakura-7b-qwen2.5-v1.0",
                        "sakura-14b-qwen2.5-v1.0",
                        "sakura-32b-qwen2beta-v0.9"
                    ]
                
                # 更新全局状态
                was_available = SAKURA_STATUS['available']
                SAKURA_STATUS['available'] = True
                SAKURA_STATUS['models'] = model_names
                SAKURA_STATUS['last_check_time'] = time.time()
                
                if not was_available:
                    logger.info(f"Sakura服务已连接，可用模型: {', '.join(model_names)}")
            else:
                if SAKURA_STATUS['available']:
                    logger.warning(f"Sakura服务响应异常，状态码: {response.status_code}")
                    SAKURA_STATUS['available'] = False
        except Exception as e:
            if SAKURA_STATUS['available']:
                logger.warning(f"Sakura服务连接中断: {e}")
                SAKURA_STATUS['available'] = False
        
        # 每30秒检查一次
        time.sleep(30)

# 在蓝图导入时启动服务监控线程（仅在app.py作为主程序时）
def start_service_monitor():
    """启动定期检查服务可用性的后台线程"""
    service_check_thread = threading.Thread(target=check_services_availability, daemon=True)
    service_check_thread.start()
    logger.info("服务监控线程已启动")
    return service_check_thread

# --- 插件管理API ---
@system_bp.route('/plugins', methods=['GET'])
def get_plugins_list():
    """获取所有已加载插件的列表及其状态"""
    try:
        plugin_mgr = get_plugin_manager()
        plugins_data = []
        for plugin_instance in plugin_mgr.get_all_plugins():
            meta = plugin_instance.get_metadata()
            meta['enabled'] = plugin_instance.is_enabled() # 添加当前启用状态
            plugins_data.append(meta)
        return jsonify({'success': True, 'plugins': plugins_data})
    except Exception as e:
        logger.error(f"获取插件列表失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '无法获取插件列表'}), 500

@system_bp.route('/plugins/<plugin_name>/enable', methods=['POST'])
def enable_plugin_api(plugin_name):
    """启用指定的插件"""
    try:
        plugin_mgr = get_plugin_manager()
        plugin = plugin_mgr.get_plugin(plugin_name)
        if plugin:
            plugin.enable()
            logger.info(f"插件 '{plugin_name}' 已通过 API 启用。")
            # TODO: 可能需要考虑持久化插件状态 (例如写入配置文件)
            return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 已启用。"})
        else:
            logger.warning(f"尝试启用不存在的插件: {plugin_name}")
            return jsonify({'success': False, 'error': '插件未找到'}), 404
    except Exception as e:
        logger.error(f"启用插件 '{plugin_name}' 失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '启用插件时出错'}), 500

@system_bp.route('/plugins/<plugin_name>/disable', methods=['POST'])
def disable_plugin_api(plugin_name):
    """禁用指定的插件"""
    try:
        plugin_mgr = get_plugin_manager()
        plugin = plugin_mgr.get_plugin(plugin_name)
        if plugin:
            plugin.disable()
            logger.info(f"插件 '{plugin_name}' 已通过 API 禁用。")
            # TODO: 可能需要考虑持久化插件状态
            return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 已禁用。"})
        else:
            logger.warning(f"尝试禁用不存在的插件: {plugin_name}")
            return jsonify({'success': False, 'error': '插件未找到'}), 404
    except Exception as e:
        logger.error(f"禁用插件 '{plugin_name}' 失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '禁用插件时出错'}), 500

@system_bp.route('/plugins/<plugin_name>', methods=['DELETE'])
def delete_plugin_api(plugin_name):
    """删除指定的插件（物理删除文件夹）"""
    logger.warning(f"收到删除插件 '{plugin_name}' 的请求。")
    try:
        plugin_mgr = get_plugin_manager()
        plugin = plugin_mgr.get_plugin(plugin_name)

        if not plugin:
            logger.warning(f"尝试删除不存在的插件: {plugin_name}")
            return jsonify({'success': False, 'error': '插件未找到'}), 404

        # 确定插件目录路径
        plugin_path = None
        for p_dir in plugin_mgr.plugin_dirs:
            potential_path = os.path.join(p_dir, plugin_name)
            if os.path.isdir(potential_path):
                plugin_path = potential_path
                break

        if plugin_path and os.path.exists(plugin_path):
            logger.info(f"准备删除插件目录: {plugin_path}")
            try:
                shutil.rmtree(plugin_path)
                logger.info(f"插件目录 '{plugin_path}' 已成功删除。")
                # 从管理器中移除插件实例和元数据
                if plugin_name in plugin_mgr.plugins:
                    del plugin_mgr.plugins[plugin_name]
                if plugin_name in plugin_mgr.plugin_metadata:
                    del plugin_mgr.plugin_metadata[plugin_name]
                # 清理钩子
                plugin_mgr.unregister_plugin_hooks(plugin_name)

                # TODO: 可能需要更新持久化的插件状态

                return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 已删除。建议重启应用以完全移除。"})
            except OSError as e:
                logger.error(f"删除插件目录 '{plugin_path}' 失败: {e}", exc_info=True)
                return jsonify({'success': False, 'error': f'删除插件文件失败: {e.strerror}'}), 500
        else:
            logger.error(f"找不到插件 '{plugin_name}' 的目录进行删除。")
            return jsonify({'success': False, 'error': '找不到插件目录'}), 404

    except Exception as e:
        logger.error(f"删除插件 '{plugin_name}' 时发生未知错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '删除插件时出错'}), 500

@system_bp.route('/plugins/<plugin_name>/config_schema', methods=['GET'])
def get_plugin_config_schema(plugin_name):
    """获取指定插件的配置规范"""
    plugin_mgr = get_plugin_manager()
    plugin = plugin_mgr.get_plugin(plugin_name)
    if plugin and isinstance(plugin, PluginBase):
        schema = plugin.get_config_spec()
        return jsonify({'success': True, 'schema': schema or []}) # 返回空列表如果没有配置
    else:
        return jsonify({'success': False, 'error': '插件未找到或无效'}), 404

@system_bp.route('/plugins/<plugin_name>/config', methods=['GET'])
def get_plugin_config(plugin_name):
    """获取指定插件的当前配置值"""
    plugin_mgr = get_plugin_manager()
    plugin = plugin_mgr.get_plugin(plugin_name)
    if plugin and isinstance(plugin, PluginBase):
        # 直接返回插件实例中加载的配置
        return jsonify({'success': True, 'config': plugin.config})
    else:
        # 或者从文件加载（如果实例中没有）
        # config_data = plugin_mgr._load_plugin_config_file(plugin_name)
        # return jsonify({'success': True, 'config': config_data})
        return jsonify({'success': False, 'error': '插件未找到或无效'}), 404

@system_bp.route('/plugins/<plugin_name>/config', methods=['POST'])
def save_plugin_config_api(plugin_name):
    """保存指定插件的配置值"""
    data = request.get_json()
    if data is None:
        return jsonify({'success': False, 'error': '请求体必须是 JSON'}), 400

    plugin_mgr = get_plugin_manager()
    if plugin_mgr.save_plugin_config(plugin_name, data):
        return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 的配置已保存。"})
    else:
        # 区分是插件不存在还是保存失败
        if not plugin_mgr.get_plugin(plugin_name):
             return jsonify({'success': False, 'error': '插件未找到'}), 404
        else:
             return jsonify({'success': False, 'error': '保存插件配置失败'}), 500

<<<<<<< HEAD
@system_bp.route('/detect_boxes', methods=['POST'])
def detect_boxes_api():
    """接收图片数据，仅返回检测到的气泡坐标"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': '缺少图像数据 (image)'}), 400

    image_data = data.get('image')
    # 可选：允许前端传递置信度阈值
    conf_threshold = float(data.get('conf_threshold', 0.6)) # 使用默认值 0.6

    try:
        # 将 Base64 转换为 PIL Image
        img_pil = base64_to_image(image_data)

        # 调用核心检测函数
        coords = get_bubble_coordinates(img_pil, conf_threshold=conf_threshold)

        return jsonify({'success': True, 'bubble_coords': coords})
    except Exception as e:
        logger.error(f"仅检测坐标时出错: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'检测坐标失败: {str(e)}'}), 500

# --- 新增：插件默认状态 API ---

@system_bp.route('/plugins/default_states', methods=['GET'])
def get_plugin_default_states():
    """获取所有插件的默认启用/禁用状态。"""
    try:
        plugin_mgr = get_plugin_manager()
        # 直接返回管理器中加载的状态字典
        return jsonify({'success': True, 'states': plugin_mgr.plugin_default_states})
    except Exception as e:
        logger.error(f"获取插件默认状态失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '无法获取插件默认状态'}), 500

@system_bp.route('/plugins/<plugin_name>/set_default_state', methods=['POST'])
def set_plugin_default_state_api(plugin_name):
    """设置指定插件的默认启用/禁用状态。"""
    data = request.get_json()
    if data is None or 'enabled' not in data or not isinstance(data['enabled'], bool):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 且包含布尔型的 "enabled" 字段'}), 400

    enabled = data['enabled']
    logger.info(f"API 请求：设置插件 '{plugin_name}' 默认状态为 {enabled}")

    try:
        plugin_mgr = get_plugin_manager()
        success = plugin_mgr.set_plugin_default_state(plugin_name, enabled)
        if success:
            return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 的默认状态已更新。"})
        else:
            # 检查插件是否存在
            if plugin_mgr.get_plugin(plugin_name):
                return jsonify({'success': False, 'error': '保存插件默认状态失败'}), 500
            else:
                return jsonify({'success': False, 'error': '插件未找到'}), 404
    except Exception as e:
        logger.error(f"设置插件 '{plugin_name}' 默认状态时出错: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '设置插件默认状态时发生内部错误'}), 500

# --- 插件管理 API 结束 ---

# 当该模块被导入时自动调用（但仅在应用作为主程序运行时）
if __name__ != '__main__':
    # 我们不在蓝图导入时就启动服务，而是让应用决定何时启动
    print("系统API蓝图已加载，可通过 start_service_monitor() 启动服务监控")

@system_bp.route('/test_baidu_ocr_connection', methods=['POST'])
def test_baidu_ocr_connection_api():
    """测试百度OCR连接状态的端点"""
    try:
        # 从请求中获取API Key和Secret Key
        data = request.json
        api_key = data.get('api_key')
        secret_key = data.get('secret_key')
        
        if not api_key or not secret_key:
            return jsonify({
                'success': False,
                'message': '请提供API Key和Secret Key'
            }), 400
            
        # 调用测试函数
        result = test_baidu_ocr_connection(api_key, secret_key)
        
        return jsonify({
            'success': result.get('success', False),
            'message': result.get('message', '未知结果')
        })
        
    except Exception as e:
        logger.error(f"测试百度OCR连接时发生错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'测试百度OCR连接时发生错误: {str(e)}'
        }), 500

@system_bp.route('/test_ai_vision_ocr', methods=['POST'])
def test_ai_vision_ocr_api():
    """测试AI视觉OCR连接状态的端点"""
    try:
        # 从请求中获取参数
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据为空'
            }), 400
            
        provider = data.get('provider')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        prompt = data.get('prompt')
        
        # 检查必要参数
        if not all([provider, api_key, model_name]):
            missing = []
            if not provider: missing.append('provider')
            if not api_key: missing.append('api_key')
            if not model_name: missing.append('model_name')
            
            return jsonify({
                'success': False,
                'message': f'缺少必要参数: {", ".join(missing)}'
            }), 400
        
        # 检查provider是否受支持
        if provider not in constants.SUPPORTED_AI_VISION_PROVIDERS:
            return jsonify({
                'success': False,
                'message': f'不支持的AI视觉服务商: {provider}，'
                           f'支持的服务商: {", ".join(constants.SUPPORTED_AI_VISION_PROVIDERS.keys())}'
            }), 400
            
        # 获取测试图片路径 - 使用debug目录中的一个示例图片
        debug_dir = get_debug_dir()
        # 尝试找到一个可用的测试图片
        test_img_path = None
        possible_imgs = ['result_image.png', 'test_lama_web_result.png']
        
        for img_name in possible_imgs:
            path = os.path.join(debug_dir, img_name)
            if os.path.exists(path):
                test_img_path = path
                break
        
        # 如果没有现成的测试图片，创建一个简单的测试图像
        if not test_img_path:
            logger.info("未找到现有测试图片，创建简单测试图像")
            test_img_path = os.path.join(debug_dir, "ai_vision_test.png")
            # 创建一个包含文字的测试图像
            test_img = Image.new('RGB', (300, 100), color=(255, 255, 255))
            draw = ImageDraw.Draw(test_img)
            font = ImageFont.truetype(resource_path(constants.DEFAULT_FONT_RELATIVE_PATH), 20)
            draw.text((10, 40), "AI视觉OCR测试文本", fill=(0, 0, 0), font=font)
            test_img.save(test_img_path)
        
        logger.info(f"使用测试图片: {test_img_path}")
        
        # 调用测试函数
        success, result = test_ai_vision_ocr(
            test_img_path, 
            provider, 
            api_key, 
            model_name,
            prompt
        )
        
        return jsonify({
            'success': success,
            'message': result,
            'test_image_path': test_img_path
        })
        
    except Exception as e:
        logger.error(f"测试AI视觉OCR连接时发生错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'测试AI视觉OCR连接时发生错误: {str(e)}'
        }), 500

@system_bp.route('/test_baidu_translate_connection', methods=['POST'])
def test_baidu_translate_connection():
    """测试百度翻译API连接状态的端点"""
    try:
        # 从请求中获取API Key和Secret Key
        data = request.json
        app_id = data.get('app_id')
        app_key = data.get('app_key')
        
        if not app_id or not app_key:
            return jsonify({
                'success': False,
                'message': '请提供App ID和App Key'
            }), 400
            
        # 设置百度翻译接口的认证信息
        baidu_translate.set_credentials(app_id, app_key)
        
        # 调用测试连接方法
        success, message = baidu_translate.test_connection()
        
        return jsonify({
            'success': success,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"测试百度翻译API连接时发生错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'测试百度翻译API连接时发生错误: {str(e)}'
        }), 500

@system_bp.route('/test_youdao_translate', methods=['POST'])
def test_youdao_translate():
    """测试有道翻译连接"""
    data = request.get_json()
    app_key = data.get('appKey')
    app_secret = data.get('appSecret')
    
    if not app_key or not app_secret:
        return jsonify({
            'success': False,
            'message': '请提供有效的AppKey和AppSecret'
        })
    
    try:
        from src.interfaces.youdao_translate_interface import YoudaoTranslateInterface
        
        # 创建接口实例
        translator = YoudaoTranslateInterface(app_key, app_secret)
        
        # 尝试翻译一个简单的测试文本
        test_text = "Hello, this is a test."
        result = translator.translate(test_text, from_lang="en", to_lang="zh-CHS")
        
        if result and result != test_text:
            return jsonify({
                'success': True,
                'message': f'连接成功！测试翻译结果: {result}'
            })
        else:
            return jsonify({
                'success': False,
                'message': '连接失败：未获得预期的翻译结果'
            })
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            'success': False,
            'message': f'连接失败：{error_msg}'
        })
=======
# 当该模块被导入时自动调用（但仅在应用作为主程序运行时）
if __name__ != '__main__':
    # 我们不在蓝图导入时就启动服务，而是让应用决定何时启动
    print("系统API蓝图已加载，可通过 start_service_monitor() 启动服务监控")
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
