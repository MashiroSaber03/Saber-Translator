import os
from flask import Flask, redirect, request
import webbrowser
import threading
import secrets
from flask_cors import CORS
from src.shared.path_helpers import resource_path
from src.shared import constants
import logging
import logging.config
import platform
import sys
import colorama
from datetime import datetime
from src.plugins.manager import get_plugin_manager
from src.interfaces.yolo_interface import load_yolo_model
import mimetypes

# 显式地为 .js 文件添加正确的 MIME 类型
# Flask/Werkzeug 在服务静态文件时通常会参考这个
mimetypes.add_type('text/javascript', '.js')

colorama.init()

# 配置日志
def setup_logging():
    """配置统一的日志系统"""
    # 创建日志目录
    log_dir = os.path.join(basedir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名，包含日期
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'comic_translator_{today}.log')
    
    # 彩色日志格式
    class ColoredFormatter(logging.Formatter):
        """自定义的彩色日志格式器"""
        COLORS = {
            'DEBUG': colorama.Fore.CYAN,
            'INFO': colorama.Fore.GREEN,
            'WARNING': colorama.Fore.YELLOW,
            'ERROR': colorama.Fore.RED,
            'CRITICAL': colorama.Fore.RED + colorama.Style.BRIGHT,
        }
        
        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{colorama.Style.RESET_ALL}"
                if not record.name.startswith('werkzeug'):  # 不对werkzeug的消息着色
                    record.msg = f"{self.COLORS[levelname]}{record.msg}{colorama.Style.RESET_ALL}"
            return super().format(record)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(message)s'
            },
            'colored': {
                '()': ColoredFormatter,
                'format': '%(asctime)s [%(levelname)s] %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'colored',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': log_file,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'werkzeug': {
                'handlers': ['file'],  # 工作日志只记录到文件
                'level': 'WARNING',  # 只显示警告及以上级别的werkzeug日志
                'propagate': False
            },
            'manga_ocr': {
                'handlers': ['file'],  # MangaOCR日志只记录到文件
                'level': 'INFO',
                'propagate': False
            },
            'PaddleOCR': {
                'handlers': ['console', 'file'],  # PaddleOCR日志记录到控制台和文件
                'level': 'INFO',  
                'propagate': False
            },
            'CoreTranslation': {
                'handlers': ['console', 'file'],  # 翻译模块日志同时输出到控制台和文件
                'level': 'INFO',
                'propagate': False
            },
            'urllib3': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            },
            'PIL': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # 创建应用日志记录器
    logger = logging.getLogger('comic_translator')
    
    # 输出佛祖保佑，永无BUG的ASCII艺术
    buddha_art = r"""
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  佛祖保佑       永无BUG
        """
    
    print(f"{colorama.Fore.GREEN}{buddha_art}{colorama.Style.RESET_ALL}")
    
    # 添加应用启动分隔线
    logger.info("="*30)
    logger.info("漫画翻译器启动")
    logger.info("="*30)
    
    # 记录系统信息
    logger.info(f"系统信息: {platform.platform()}")
    logger.info(f"Python版本: {sys.version.split()[0]}")
    
    return logger

# 确定应用根目录 (app.py 所在的目录，即项目根目录)
basedir = os.path.abspath(os.path.dirname(__file__))

# 创建日志记录器
logger = setup_logging()
logger.info("已手动添加 '.js' 的 MIME 类型为 'text/javascript'")

# 准备应用程序
app = Flask(__name__,
           # 相对于 app.py (项目根目录) 的路径
           static_folder=os.path.join('src', 'app', 'static'),
           template_folder=os.path.join('src', 'app', 'templates'),
           static_url_path='') # 保持 static_url_path 为空，以便 URL 保持 /style.css 等形式
CORS(app)

# --- 预加载YOLOv5模型 ---
try:
    logger.info("预加载YOLOv5气泡检测模型...")
    yolo_model = load_yolo_model()
    if yolo_model is not None:
        logger.info("YOLOv5模型预加载成功")
    else:
        logger.warning("YOLOv5模型预加载失败，将在首次使用时再次尝试加载")
except Exception as e:
    logger.error(f"预加载YOLOv5模型时发生错误: {e}", exc_info=True)
# -----------------------

# --- 初始化插件管理器 ---
try:
    plugin_manager = get_plugin_manager(app=app) # 传入 app 实例
    logger.info("插件管理器初始化完成。")
except Exception as e:
    logger.error(f"初始化插件管理器失败: {e}", exc_info=True)
    # 根据需要决定是否要在此处退出应用
    # raise e
# -----------------------

# --- 导入并注册蓝图 ---
try:
    # 通过src/app/__init__.py中的register_blueprints函数注册所有蓝图
    from src.app import register_blueprints
    register_blueprints(app)
    logger.info("蓝图注册成功")
except ImportError as e:
    logger.error(f"导入或注册蓝图失败 - {e}")
    # 在开发阶段，这通常意味着文件或变量名错误，或者循环导入
    # 在打包后，可能意味着 spec 文件没有包含这些模块
    raise e
# -----------------

# 设置Flask应用的密钥，用于session加密
app.secret_key = secrets.token_hex(16)

# 添加临时文件夹配置
TEMP_FOLDER_PATH = resource_path(constants.TEMP_FOLDER_NAME)
app.config['TEMP_FOLDER'] = TEMP_FOLDER_PATH
os.makedirs(TEMP_FOLDER_PATH, exist_ok=True)

# 设置上传文件夹路径
UPLOAD_FOLDER_PATH = resource_path(constants.UPLOAD_FOLDER_NAME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH
os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

# ---------- 添加重定向路由以保持向后兼容性 ----------

# 添加配置API的重定向路由
@app.route('/get_prompts', methods=['GET'])
def get_prompts_redirect():
    """重定向到配置API蓝图中的获取提示词端点"""
    return redirect("/api/get_prompts", code=307)

@app.route('/get_textbox_prompts', methods=['GET'])
def get_textbox_prompts_redirect():
    """重定向到配置API蓝图中的获取文本框提示词端点"""
    return redirect("/api/get_textbox_prompts", code=307)

@app.route('/get_used_models', methods=['GET'])
def get_used_models_redirect():
    """重定向到配置API蓝图中的获取使用过的模型端点"""
    query_string = request.query_string.decode()
    if query_string:
        return redirect(f"/api/get_used_models?{query_string}", code=307)
    else:
        return redirect("/api/get_used_models", code=307)

@app.route('/get_model_info', methods=['GET'])
def get_model_info_redirect():
    """重定向到配置API蓝图中的获取模型信息端点"""
    return redirect("/api/get_model_info", code=307)

@app.route('/save_model_info', methods=['POST'])
def save_model_info_redirect():
    """重定向到配置API蓝图中的保存模型信息端点"""
    return redirect("/api/save_model_info", code=307)

@app.route('/get_prompt_content', methods=['GET'])
def get_prompt_content_redirect():
    """重定向到配置API蓝图中的获取提示词内容端点"""
    query_string = request.query_string.decode()
    if query_string:
        return redirect(f"/api/get_prompt_content?{query_string}", code=307)
    else:
        return redirect("/api/get_prompt_content", code=307)

@app.route('/save_prompt', methods=['POST'])
def save_prompt_redirect():
    """重定向到配置API蓝图中的保存提示词端点"""
    return redirect("/api/save_prompt", code=307)

@app.route('/reset_prompt_to_default', methods=['POST'])
def reset_prompt_to_default_redirect():
    """重定向到配置API蓝图中的重置提示词端点"""
    return redirect("/api/reset_prompt_to_default", code=307)

@app.route('/delete_prompt', methods=['POST'])
def delete_prompt_redirect():
    """重定向到配置API蓝图中的删除提示词端点"""
    return redirect("/api/delete_prompt", code=307)

@app.route('/get_textbox_prompt_content', methods=['GET'])
def get_textbox_prompt_content_redirect():
    """重定向到配置API蓝图中的获取文本框提示词内容端点"""
    query_string = request.query_string.decode()
    if query_string:
        return redirect(f"/api/get_textbox_prompt_content?{query_string}", code=307)
    else:
        return redirect("/api/get_textbox_prompt_content", code=307)

@app.route('/save_textbox_prompt', methods=['POST'])
def save_textbox_prompt_redirect():
    """重定向到配置API蓝图中的保存文本框提示词端点"""
    return redirect("/api/save_textbox_prompt", code=307)

@app.route('/reset_textbox_prompt_to_default', methods=['POST'])
def reset_textbox_prompt_to_default_redirect():
    """重定向到配置API蓝图中的重置文本框提示词端点"""
    return redirect("/api/reset_textbox_prompt_to_default", code=307)

@app.route('/delete_textbox_prompt', methods=['POST'])
def delete_textbox_prompt_redirect():
    """重定向到配置API蓝图中的删除文本框提示词端点"""
    return redirect("/api/delete_textbox_prompt", code=307)

# 添加翻译API的重定向路由
@app.route('/translate_image', methods=['POST'])
def translate_image_redirect():
    """重定向到翻译API蓝图中的翻译图片端点"""
    return redirect("/api/translate_image", code=307)

@app.route('/re_render_image', methods=['POST'])
def re_render_image_redirect():
    """重定向到翻译API蓝图中的重新渲染图片端点"""
    return redirect("/api/re_render_image", code=307)

@app.route('/re_render_single_bubble', methods=['POST'])
def re_render_single_bubble_redirect():
    """重定向到翻译API蓝图中的重新渲染单个气泡端点"""
    return redirect("/api/re_render_single_bubble", code=307)

@app.route('/apply_settings_to_all_images', methods=['POST'])
def apply_settings_to_all_images_redirect():
    """重定向到翻译API蓝图中的应用设置到所有图片端点"""
    return redirect("/api/apply_settings_to_all_images", code=307)

@app.route('/translate_single_text', methods=['POST'])
def translate_single_text_redirect():
    """重定向到翻译API蓝图中的翻译单个文本端点"""
    return redirect("/api/translate_single_text", code=307)

# 添加系统API的重定向路由
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf_redirect():
    """重定向到系统API蓝图中的上传PDF端点"""
    return redirect("/api/upload_pdf", code=307)

@app.route('/clean_debug_files', methods=['POST'])
def clean_debug_files_redirect():
    """重定向到系统API蓝图中的清理调试文件端点"""
    return redirect("/api/clean_debug_files", code=307)

@app.route('/test_ollama_connection', methods=['GET'])
def test_ollama_connection_redirect():
    """重定向到系统API蓝图中的测试Ollama连接端点"""
    query_string = request.query_string.decode()
    if query_string:
        return redirect(f"/api/test_ollama_connection?{query_string}", code=307)
    else:
        return redirect("/api/test_ollama_connection", code=307)

@app.route('/test_lama_repair', methods=['GET'])
def test_lama_repair_redirect():
    """重定向到系统API蓝图中的测试LAMA修复端点"""
    query_string = request.query_string.decode()
    if query_string:
        return redirect(f"/api/test_lama_repair?{query_string}", code=307)
    else:
        return redirect("/api/test_lama_repair", code=307)

@app.route('/test_params', methods=['POST'])
def test_params_redirect():
    """重定向到系统API蓝图中的测试参数端点"""
    return redirect("/api/test_params", code=307)

@app.route('/test_sakura_connection', methods=['GET'])
def test_sakura_connection_redirect():
    """重定向到系统API蓝图中的测试Sakura连接端点"""
    query_string = request.query_string.decode()
    if query_string:
        return redirect(f"/api/test_sakura_connection?{query_string}", code=307)
    else:
        return redirect("/api/test_sakura_connection", code=307)

@app.route('/test_baidu_translate_connection', methods=['POST'])
def test_baidu_translate_connection_redirect():
    """重定向到系统API蓝图中的测试百度翻译API连接端点"""
    return redirect("/api/test_baidu_translate_connection", code=307)

# -----------------------------------------------

# 在应用启动时创建必要的文件夹
def create_required_directories():
    # 获取项目根目录
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 确保config目录及其子目录存在
    os.makedirs(os.path.join(base_path, 'config'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'config', 'plugin_configs'), exist_ok=True)
    
    # 确保data目录及其子目录存在
    os.makedirs(os.path.join(base_path, 'data', 'debug'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'data', 'sessions'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'data', 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'data', 'temp'), exist_ok=True)  # 新增: 临时目录
    
    # 确保logs目录存在
    os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)

# 在应用启动时调用
create_required_directories()

if __name__ == '__main__':
    # 禁用Flask的默认日志处理
    app.logger.handlers.clear()
    
    # 设置Flask日志
    app.logger.setLevel(logging.WARNING)
    
    # 优化werkzeug日志
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)  # 只显示警告及以上级别
    
    # 精确控制第三方库的日志级别
    silenced_modules = {
        'PIL': logging.WARNING,
        'matplotlib': logging.WARNING,
        'httpx': logging.WARNING,
        'urllib3': logging.WARNING,
        'torch': logging.WARNING,
        'transformers': logging.WARNING,
        'mangaocr': logging.WARNING,
        'manga_ocr': logging.WARNING,
        'paddleocr': logging.WARNING,  # 只控制PaddleOCR库的内部日志，不影响我们的PaddleOCR接口日志
    }
    
    for module, level in silenced_modules.items():
        logging.getLogger(module).setLevel(level)
    
    # 确保翻译模块的日志级别为INFO
    logging.getLogger('CoreTranslation').setLevel(logging.INFO)
    
    # 找到loguru日志库的处理器并禁用控制台输出
    try:
        from loguru import logger as loguru_logger
        loguru_logger.remove()  # 移除所有处理器
        # 只添加文件处理器
        loguru_logger.add(os.path.join(basedir, 'logs', f'loguru_{datetime.now().strftime("%Y-%m-%d")}.log'), 
                          level="INFO")
    except ImportError:
        pass  # loguru不是必需的库
    
    # 打开浏览器
    threading.Timer(1, open_browser).start()
    
    # 启动Sakura服务监控线程
    from src.app.api.system_api import start_service_monitor
    start_service_monitor()
    logger.info("服务监控线程已启动")
    
    # 预加载MangaOCR模型
    try:
        # 在导入MangaOCR之前先关闭其日志输出
        for manga_log in ['manga_ocr.ocr', 'manga_ocr']:
            manga_logger = logging.getLogger(manga_log)
            manga_logger.setLevel(logging.WARNING)
            # 移除控制台处理器
            for handler in list(manga_logger.handlers):
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    manga_logger.removeHandler(handler)
            
        from src.interfaces.manga_ocr_interface import preload_manga_ocr
        preload_manga_ocr()
        logger.info("已启动 MangaOCR 模型预加载线程")
    except Exception as e:
        logger.error(f"MangaOCR 预加载失败: {e}")
    
    logger.info("程序正在运行，请在浏览器中访问 http://127.0.0.1:5000/")
    
    # 启动Flask应用但不输出启动信息
    import logging
    log = logging.getLogger('werkzeug')
    log.disabled = True
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    app.run(debug=False, use_reloader=False)

    