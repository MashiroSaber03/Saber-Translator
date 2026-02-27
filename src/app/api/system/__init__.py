"""
系统API模块

将原来的 system_api.py 拆分为多个子模块：
- plugins.py: 插件管理相关API
- fonts.py: 字体管理相关API
- downloads.py: 批量下载相关API
- tests.py: 连接测试相关API
- files.py: 文件处理相关API
"""

from flask import Blueprint, jsonify, current_app
import socket

# 创建系统API蓝图
system_bp = Blueprint('system_api', __name__, url_prefix='/api')


def get_local_ip():
    """获取本机局域网 IP 地址"""
    try:
        # 创建一个 UDP socket 并连接到外部地址（不会真的发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@system_bp.route('/server-info', methods=['GET'])
def get_server_info():
    """获取服务器信息，包括局域网访问地址"""
    local_ip = get_local_ip()
    port = int(current_app.config.get("APP_PORT", 5000))
    
    return jsonify({
        "success": True,
        "local_url": f"http://127.0.0.1:{port}/",
        "lan_url": f"http://{local_ip}:{port}/",
        "lan_ip": local_ip,
        "port": port
    })


@system_bp.route('/local-token', methods=['GET'])
def get_local_token():
    """
    获取本机模式 API 访问令牌。
    前端在启动后读取并通过自定义请求头发送。
    """
    enabled = bool(current_app.config.get("LOCAL_TOKEN_REQUIRED", False))
    return jsonify({
        "success": True,
        "enabled": enabled,
        "header": current_app.config.get("LOCAL_API_TOKEN_HEADER", "X-Saber-Local-Token"),
        "token": current_app.config.get("LOCAL_API_TOKEN", "") if enabled else "",
    })


# 导入各个子模块以注册路由
from . import plugins
from . import fonts
from . import downloads
from . import tests
from . import files
from . import mobi_handler  # MOBI/AZW 电子书解析
from . import pdf_handler   # PDF 文件解析
from . import gpu           # GPU 资源管理

__all__ = ['system_bp']
