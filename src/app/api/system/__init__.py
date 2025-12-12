"""
系统API模块

将原来的 system_api.py 拆分为多个子模块：
- plugins.py: 插件管理相关API
- fonts.py: 字体管理相关API
- downloads.py: 批量下载相关API
- tests.py: 连接测试相关API
- files.py: 文件处理相关API
"""

from flask import Blueprint, jsonify
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
    port = 5000  # 默认端口
    
    return jsonify({
        "success": True,
        "local_url": f"http://127.0.0.1:{port}/",
        "lan_url": f"http://{local_ip}:{port}/",
        "lan_ip": local_ip,
        "port": port
    })


# 导入各个子模块以注册路由
from . import plugins
from . import fonts
from . import downloads
from . import tests
from . import files
from . import mobi_handler  # MOBI/AZW 电子书解析

__all__ = ['system_bp']
