"""
路径处理辅助模块，提供文件路径处理相关的通用函数
"""

import os
import sys
import tempfile
import logging

logger = logging.getLogger("PathHelpers")
# 配置日志（如果需要独立测试或记录路径问题）
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def resource_path(relative_path):
    """
    获取资源的绝对路径，适用于开发环境和PyInstaller打包环境
    
    Args:
        relative_path: 相对路径
    
    Returns:
        资源的绝对路径
    """
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
        logger.info(f"打包环境中，基础路径: {base_path}")
    except Exception:
        # 开发环境中的路径 - 获取 path_helpers.py 所在的目录 (src/shared)，然后向上两级到项目根目录
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        logger.info(f"开发环境中，项目根目录: {base_path}")
    
    abs_path = os.path.join(base_path, relative_path)
    logger.debug(f"资源路径解析: '{relative_path}' -> '{abs_path}'")
    return abs_path


def get_debug_dir(subdirectory=None):
    """
    获取debug目录的绝对路径 (data/debug/)
    
    Args:
        subdirectory: 子目录名，可选
    
    Returns:
        debug目录的绝对路径
    """
    project_root = resource_path('') # 获取项目根目录
    debug_base = os.path.join(project_root, 'data', 'debug')
    
    try:
        os.makedirs(debug_base, exist_ok=True)
        if subdirectory:
            debug_path = os.path.join(debug_base, subdirectory)
            os.makedirs(debug_path, exist_ok=True)
            return debug_path
        return debug_base
    except Exception as e:
        logger.error(f"无法创建或访问调试目录: {debug_base} - {e}", exc_info=True)
        
        # 尝试在用户目录创建
        try:
            user_home = os.path.expanduser("~")
            user_debug_base = os.path.join(user_home, "Saber-Translator", "debug")
            os.makedirs(user_debug_base, exist_ok=True)
            
            if subdirectory:
                user_debug_path = os.path.join(user_debug_base, subdirectory)
                os.makedirs(user_debug_path, exist_ok=True)
                return user_debug_path
            return user_debug_base
        except Exception as e2:
            logger.error(f"无法在用户目录创建debug目录: {e2}", exc_info=True)
            
            # 回退到临时目录作为最后的手段
            temp_dir = tempfile.gettempdir()
            debug_temp = os.path.join(temp_dir, "Saber-Translator-debug")
            os.makedirs(debug_temp, exist_ok=True)
            if subdirectory:
                subdir_temp = os.path.join(debug_temp, subdirectory)
                os.makedirs(subdir_temp, exist_ok=True)
                return subdir_temp
            return debug_temp


def is_packaged():
    """
    检测是否在PyInstaller环境中运行
    
    Returns:
        布尔值，表示是否在打包环境中运行
    """
    return getattr(sys, 'frozen', False)


def get_font_path(font_path):
    """
    获取字体的绝对路径
    
    Args:
        font_path: 字体路径，可能是相对路径也可能是绝对路径
        
    Returns:
        字体的绝对路径
    """
    if not font_path:
        # 如果未提供字体，使用默认字体
        default_font_rel_path = os.path.join('src', 'app', 'static', 'fonts', 'STSONG.TTF')
        return resource_path(default_font_rel_path)
    
    # 处理不同格式的字体路径
    if font_path.startswith('static/fonts/'):
        # 从static/fonts/开头的路径
        rel_path = os.path.join('src', 'app', font_path)
        return resource_path(rel_path)
    elif font_path.startswith('static/'):
        # 从static/开头的路径
        rel_path = os.path.join('src', 'app', font_path)
        return resource_path(rel_path)
    elif font_path.startswith('fonts/'):
        # 从fonts/开头的路径
        rel_path = os.path.join('src', 'app', 'static', font_path)
        return resource_path(rel_path)
    elif os.path.exists(font_path):
        # 如果路径存在，直接返回
        return font_path
    else:
        # 尝试在当前路径下查找
        app_dir_path = resource_path(os.path.basename(font_path))
        if os.path.exists(app_dir_path):
            return app_dir_path
            
        # 尝试在fonts目录下查找
        fonts_dir_path = resource_path(os.path.join('src', 'app', 'static', 'fonts', os.path.basename(font_path)))
        if os.path.exists(fonts_dir_path):
            return fonts_dir_path
            
        # 尝试在static目录下查找
        static_dir_path = resource_path(os.path.join('src', 'app', 'static', os.path.basename(font_path)))
        if os.path.exists(static_dir_path):
            return static_dir_path
    
    # 如果所有尝试都失败，返回默认字体
    logger.warning(f"未找到字体 {font_path}，使用默认字体")
    default_font_rel_path = os.path.join('src', 'app', 'static', 'fonts', 'STSONG.TTF')
    return resource_path(default_font_rel_path)


# 测试代码
if __name__ == '__main__':
    # 启用日志输出以便测试
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("--- 测试路径函数 ---")
    print("项目根目录:", resource_path(''))
    print("调试目录:", get_debug_dir())
    print("气泡调试目录:", get_debug_dir('bubbles'))
    print("默认字体路径:", get_font_path(None))
    print("尝试获取 STXINGKA:", get_font_path('static/fonts/STXINGKA.TTF')) # 假设字体已移动
    print("尝试获取不存在字体:", get_font_path('nonexistentfont.ttf'))