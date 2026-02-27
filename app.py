import os
from flask import Flask, redirect, request, jsonify
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
# YOLO已被CTD替换，不再需要预加载
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
    
    # 简洁的启动信息
    logger.info(f"Saber-Translator 启动中... (Python {sys.version.split()[0]})")
    
    return logger

# 确定应用根目录 (app.py 所在的目录，即项目根目录)
basedir = os.path.abspath(os.path.dirname(__file__))


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# 本机单用户模式（默认开启）
LOCAL_MODE_ENABLED = _env_flag("SABER_LOCAL_MODE", True)
APP_PORT = int(os.getenv("SABER_PORT", "5000"))
DEFAULT_HOST = "127.0.0.1" if LOCAL_MODE_ENABLED else "0.0.0.0"
APP_HOST = os.getenv("SABER_BIND_HOST", DEFAULT_HOST)
LOCAL_API_TOKEN_HEADER = "X-Saber-Local-Token"
LOCAL_API_TOKEN = os.getenv("SABER_LOCAL_TOKEN") or secrets.token_urlsafe(32)
# 是否强制本地访问令牌（默认关闭，可按需开启）
LOCAL_TOKEN_REQUIRED = _env_flag("SABER_REQUIRE_LOCAL_TOKEN", False)

# 默认仅允许本机 Web UI 作为跨域来源
allowed_origins_env = os.getenv(
    "SABER_ALLOWED_ORIGINS",
    f"http://127.0.0.1:{APP_PORT},http://localhost:{APP_PORT}",
)
ALLOWED_CORS_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

# 创建日志记录器
logger = setup_logging()

# 准备应用程序
app = Flask(__name__,
           # 相对于 app.py (项目根目录) 的路径
           static_folder=os.path.join('src', 'app', 'static'),
           static_url_path='') # 保持 static_url_path 为空，以便 URL 保持 /style.css 等形式
CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_CORS_ORIGINS}},
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", LOCAL_API_TOKEN_HEADER],
)

app.config["LOCAL_MODE_ENABLED"] = LOCAL_MODE_ENABLED
app.config["LOCAL_API_TOKEN_HEADER"] = LOCAL_API_TOKEN_HEADER
app.config["LOCAL_API_TOKEN"] = LOCAL_API_TOKEN
app.config["LOCAL_TOKEN_REQUIRED"] = LOCAL_TOKEN_REQUIRED
app.config["APP_HOST"] = APP_HOST
app.config["APP_PORT"] = APP_PORT

if LOCAL_MODE_ENABLED:
    logger.info("本机单用户安全模式已启用")
    if LOCAL_TOKEN_REQUIRED:
        logger.info(f"本地 API Token 校验已启用，Header: {LOCAL_API_TOKEN_HEADER}")
    else:
        logger.info("本地 API Token 校验已关闭")


@app.before_request
def enforce_local_api_token():
    """
    本机模式下，对所有写操作 API 强制校验本地令牌。
    """
    if not app.config.get("LOCAL_MODE_ENABLED", True):
        return None
    if not app.config.get("LOCAL_TOKEN_REQUIRED", False):
        return None

    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None

    if not request.path.startswith("/api/"):
        return None

    # 获取本地 token 的接口自身不需要 token
    if request.path == "/api/local-token":
        return None

    expected = app.config.get("LOCAL_API_TOKEN", "")
    provided = request.headers.get(app.config.get("LOCAL_API_TOKEN_HEADER", LOCAL_API_TOKEN_HEADER), "")
    if not expected or not provided or not secrets.compare_digest(expected, provided):
        return jsonify({
            "success": False,
            "error": "未授权：缺少或无效的本地访问令牌"
        }), 401
    return None

# --- 初始化插件管理器 ---
try:
    plugin_manager = get_plugin_manager(app=app)
except Exception as e:
    logger.error(f"初始化插件管理器失败: {e}", exc_info=True)
# -----------------------

# --- 导入并注册蓝图 ---
try:
    from src.app import register_blueprints
    register_blueprints(app)
except ImportError as e:
    logger.error(f"导入或注册蓝图失败 - {e}")
    raise e
# -----------------

# 设置Flask应用的密钥
app.secret_key = secrets.token_hex(16)

# 初始化性能监控和错误处理器
from src.shared.performance import RequestTimer
RequestTimer.init_app(app)
from src.app.error_handlers import register_error_handlers
register_error_handlers(app)

def get_local_ip():
    """获取本机局域网 IP 地址"""
    import socket
    try:
        # 创建一个 UDP socket 并连接到外部地址（不会真的发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{APP_PORT}/")

# 注册重定向路由以保持向后兼容性
from src.app.route_redirects import register_redirects
register_redirects(app)

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
    os.makedirs(os.path.join(base_path, 'data', 'temp'), exist_ok=True)  # 临时目录
    
    # 确保logs目录存在
    os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)

# 在应用启动时调用
create_required_directories()

# 自动迁移书架数据（从旧格式升级到新格式）
def auto_migrate_bookshelf_data():
    """自动检测并迁移书架数据到新格式"""
    try:
        from src.core import bookshelf_manager
        result = bookshelf_manager.migrate_books_metadata()
        if result.get("migrated"):
            logger.info(f"📚 书架数据迁移完成: {result.get('message')}")
        # 如果已是新格式则不输出日志，静默跳过
    except Exception as e:
        logger.warning(f"书架数据迁移检查失败: {e}")

    # 新增：统一数据结构迁移（会话和Insight数据）
    try:
        from src.core.data_migration import check_and_migrate
        result = check_and_migrate()
        if result.get("migrated"):
            logger.info(f"📦 数据结构迁移完成: {result.get('message')}")
    except Exception as e:
        logger.warning(f"数据结构迁移检查失败: {e}")

auto_migrate_bookshelf_data()

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
        'transformers.utils': logging.ERROR,  # 抑制 transformers 内部警告
        'mangaocr': logging.WARNING,
        'manga_ocr': logging.WARNING,
        'paddleocr': logging.WARNING,
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
    from src.app.api.system.tests import start_service_monitor
    start_service_monitor()
    
    # 预加载MangaOCR模型
    try:
        # 在导入MangaOCR之前先设置日志级别
        # 允许MangaOCR接口的INFO日志，但限制库内部的日志
        for manga_log in ['manga_ocr.ocr', 'manga_ocr']:
            manga_logger = logging.getLogger(manga_log)
            manga_logger.setLevel(logging.WARNING)  # 限制库内部日志
            # 移除控制台处理器
            for handler in list(manga_logger.handlers):
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    manga_logger.removeHandler(handler)
        
        # 确保我们自己的MangaOCR接口日志可见
        logging.getLogger('MangaOCRInterface').setLevel(logging.INFO)
        logging.getLogger('CoreOCR').setLevel(logging.INFO)
            
        from src.interfaces.manga_ocr_interface import preload_manga_ocr
        preload_manga_ocr()
    except Exception as e:
        logger.warning(f"MangaOCR 预加载失败: {e}")
    
    # 获取局域网 IP
    local_ip = get_local_ip()
    
    # 美化启动信息
    local_url = f"http://127.0.0.1:{APP_PORT}/"
    lan_url = f"http://{local_ip}:{APP_PORT}/"
    
    # Saber-Translator ASCII Art Logo
    logo = f"""
{colorama.Fore.MAGENTA}  ____        _                 {colorama.Fore.CYAN}_____                    _       _             
{colorama.Fore.MAGENTA} / ___|  __ _| |__   ___ _ __  {colorama.Fore.CYAN}|_   _| __ __ _ _ __  ___| | __ _| |_ ___  _ __ 
{colorama.Fore.MAGENTA} \\___ \\ / _` | '_ \\ / _ \\ '__| {colorama.Fore.CYAN}  | || '__/ _` | '_ \\/ __| |/ _` | __/ _ \\| '__|
{colorama.Fore.MAGENTA}  ___) | (_| | |_) |  __/ |    {colorama.Fore.CYAN}  | || | | (_| | | | \\__ \\ | (_| | || (_) | |   
{colorama.Fore.MAGENTA} |____/ \\__,_|_.__/ \\___|_|    {colorama.Fore.CYAN}  |_||_|  \\__,_|_| |_|___/_|\\__,_|\\__\\___/|_|   
{colorama.Style.RESET_ALL}"""
    print(logo)
    print(f"{colorama.Fore.CYAN}╔{'═'*46}╗{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}  {colorama.Fore.GREEN}✔ 程序已启动{colorama.Style.RESET_ALL}{' '*32}{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}{' '*46}{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}  {colorama.Fore.YELLOW}本机:{colorama.Style.RESET_ALL}    {local_url:<35}{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}  {colorama.Fore.YELLOW}局域网:{colorama.Style.RESET_ALL}  {lan_url:<35}{colorama.Fore.CYAN}║{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}╚{'═'*46}╝{colorama.Style.RESET_ALL}\n")
    
    # 启动Flask应用但不输出启动信息
    import logging
    log = logging.getLogger('werkzeug')
    log.disabled = True
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    # 本机模式默认仅监听 127.0.0.1，可通过 SABER_BIND_HOST 覆盖
    app.run(host=APP_HOST, port=APP_PORT, debug=False, use_reloader=False, threaded=True)

    
