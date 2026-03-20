"""
字体管理相关API

包含所有与字体管理相关的API端点：
- 获取字体列表
- 上传字体文件
"""

import os
import re
import logging
from typing import Dict, List, Any
from flask import request, jsonify

from . import system_bp
from src.shared.path_helpers import resource_path

logger = logging.getLogger("SystemAPI.Fonts")

# 默认字体映射
DEFAULT_FONTS_MAPPING = {
    '华文行楷': 'fonts/STXINGKA.TTF',
    '华文新魏': 'fonts/STXINWEI.TTF',
    '华文中宋': 'fonts/STZHONGS.TTF',
    '楷体': 'fonts/STKAITI.TTF',
    '隶书': 'fonts/STLITI.TTF',
    '宋体': 'fonts/STSONG.TTF',
    '微软雅黑': 'fonts/msyh.ttc',
    '微软雅黑粗体': 'fonts/msyhbd.ttc',
    '幼圆': 'fonts/SIMYOU.TTF',
    '仿宋': 'fonts/STFANGSO.TTF',
    '华文琥珀': 'fonts/STHUPO.TTF',
    '华文细黑': 'fonts/STXIHEI.TTF',
    '中易楷体': 'fonts/simkai.ttf',
    '中易仿宋': 'fonts/simfang.ttf',
    '中易黑体': 'fonts/simhei.ttf',
    '中易隶书': 'fonts/SIMLI.TTF'
}

# 支持的字体文件扩展名
SUPPORTED_FONT_EXTENSIONS = ('.ttf', '.ttc', '.otf')


def get_font_directory() -> str:
    """获取字体目录路径"""
    return resource_path(os.path.join('src', 'app', 'static', 'fonts'))


def get_windows_fonts_directory() -> str:
    """获取 Windows 系统字体目录（默认 C:\\Windows\\Fonts）"""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return os.path.join(windir, "Fonts")


def sanitize_filename(filename: str) -> str:
    """
    安全处理文件名，保留中文字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 仅保留字母、数字、下划线、横杠、点和中文字符
    safe_filename = re.sub(r'[^\w\.-\u4e00-\u9fff]', '_', filename)
    
    # 确保文件名不为空
    if not safe_filename:
        safe_filename = 'unnamed_font.ttf'
    
    return safe_filename


def generate_display_name(file_name: str, default_fonts: Dict[str, str]) -> tuple[str, bool]:
    """
    生成字体的友好显示名称
    
    Args:
        file_name: 字体文件名
        default_fonts: 默认字体映射字典
        
    Returns:
        (display_name, is_default) 元组
    """
    # 检查是否在默认映射中
    for name, path in default_fonts.items():
        if file_name == os.path.basename(path):
            return name, True
    
    # 生成友好名称（去除扩展名）
    display_name = os.path.splitext(file_name)[0]
    
    # 将大写文件名转为更友好的格式
    if display_name.isupper():
        display_name = ' '.join(display_name.split('_'))
        display_name = display_name.title()
    
    return display_name, False


def _scan_font_dir(font_dir: str, *, source: str) -> List[Dict[str, Any]]:
    """
    扫描字体目录，返回字体信息列表。

    Args:
        font_dir: 要扫描的目录路径
        source: 字体来源标记（local/system）

    Returns:
        字体信息列表
    """
    results: List[Dict[str, Any]] = []
    if not font_dir or not os.path.isdir(font_dir):
        return results

    for file_name in os.listdir(font_dir):
        if not file_name.lower().endswith(SUPPORTED_FONT_EXTENSIONS):
            continue

        abs_path = os.path.abspath(os.path.join(font_dir, file_name))
        if not os.path.isfile(abs_path):
            continue

        if source == "local":
            display_name, is_default = generate_display_name(file_name, DEFAULT_FONTS_MAPPING)
            path_value = f'fonts/{file_name}'
        else:
            # 系统字体：仅保证返回可用的绝对路径
            display_name = os.path.splitext(file_name)[0]
            is_default = False
            path_value = abs_path

        results.append({
            'file_name': file_name,
            'display_name': display_name,
            'path': path_value,
            'is_default': is_default,
            'source': source
        })

    return results


@system_bp.route('/get_font_list', methods=['GET'])
def get_font_list():
    """
    获取fonts目录中的所有字体文件
    
    Returns:
        JSON响应:
        {
            'fonts': [
                {
                    'file_name': 'xxx.ttf',
                    'display_name': '显示名称',
                    'path': 'fonts/xxx.ttf',
                    'is_default': bool
                }
            ],
            'default_fonts': {
                '微软雅黑': 'fonts/msyh.ttc',
                // ... 其他默认字体映射
            }
        }
    """
    try:
        font_dir = get_font_directory()

        # 1) 内置/上传字体（项目 fonts 目录）
        local_fonts = _scan_font_dir(font_dir, source="local")

        # 2) Windows 已安装字体（C:\Windows\Fonts）
        system_fonts: List[Dict[str, Any]] = []
        if os.name == "nt":
            system_font_dir = get_windows_fonts_directory()
            system_fonts = _scan_font_dir(system_font_dir, source="system")

        all_fonts: List[Dict[str, Any]] = local_fonts + system_fonts

        # 排序：本地优先；默认映射字体优先；再按名称排序
        all_fonts.sort(key=lambda x: (
            0 if x.get('source') == 'local' else 1,
            not x.get('is_default', False),
            x.get('display_name', '')
        ))
        
        logger.info(f"成功获取字体列表，共 {len(all_fonts)} 个字体")
        
        return jsonify({
            'fonts': all_fonts,
            'default_fonts': DEFAULT_FONTS_MAPPING
        })
        
    except Exception as e:
        logger.error(f"获取字体列表失败: {str(e)}", exc_info=True)
        return jsonify({'error': f"获取字体列表失败: {str(e)}"}), 500


@system_bp.route('/upload_font', methods=['POST'])
def upload_font():
    """
    上传字体文件
    
    接收表单数据中的字体文件并保存到fonts目录
    
    Returns:
        JSON响应:
        {
            'success': True,
            'file_name': 'xxx.ttf',
            'display_name': '显示名称',
            'path': 'fonts/xxx.ttf'
        }
    """
    try:
        # 检查是否有文件上传
        if 'font' not in request.files:
            logger.warning("上传字体请求中未找到文件")
            return jsonify({'error': '未找到字体文件'}), 400
        
        font_file = request.files['font']
        
        # 检查文件名
        if font_file.filename == '':
            logger.warning("上传的字体文件名为空")
            return jsonify({'error': '未选择字体文件'}), 400
        
        # 检查文件格式
        if not font_file.filename.lower().endswith(SUPPORTED_FONT_EXTENSIONS):
            logger.warning(f"不支持的字体格式: {font_file.filename}")
            return jsonify({
                'error': f'只支持 {", ".join(ext.upper() for ext in SUPPORTED_FONT_EXTENSIONS)} 格式的字体文件'
            }), 400
        
        # 安全处理文件名
        original_filename = font_file.filename
        safe_filename = sanitize_filename(original_filename)
        
        logger.info(f"开始上传字体: {original_filename} -> {safe_filename}")
        
        # 确保字体目录存在
        font_dir = get_font_directory()
        os.makedirs(font_dir, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(font_dir, safe_filename)
        
        # 检查文件是否已存在
        if os.path.exists(file_path):
            logger.warning(f"字体文件已存在: {safe_filename}")
            return jsonify({
                'error': f'字体文件 "{safe_filename}" 已存在，请先删除或重命名'
            }), 409  # Conflict
        
        font_file.save(file_path)
        logger.info(f"字体文件已保存: {file_path}")
        
        # 生成友好的显示名称
        display_name, _ = generate_display_name(safe_filename, DEFAULT_FONTS_MAPPING)
        
        return jsonify({
            'success': True,
            'file_name': safe_filename,
            'display_name': display_name,
            'fontPath': f'fonts/{safe_filename}',
            'message': f'字体 "{display_name}" 上传成功'
        })
        
    except Exception as e:
        logger.error(f"上传字体文件失败: {str(e)}", exc_info=True)
        return jsonify({'error': f"上传字体文件失败: {str(e)}"}), 500
