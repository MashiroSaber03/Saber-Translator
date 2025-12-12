"""
文件处理相关API

包含所有与文件处理相关的API端点：
- 清理调试文件
- 参数测试
"""

import os
import shutil
import logging
from flask import request, jsonify

from . import system_bp
from src.shared.path_helpers import get_debug_dir, resource_path

logger = logging.getLogger("SystemAPI.Files")


@system_bp.route('/clean_debug_files', methods=['POST'])
def clean_debug_files():
    """
    清理调试目录中的文件和临时下载文件
    
    返回:
        {
            'success': True,
            'message': '清理了 N 个调试文件，释放了 X MB 空间'
        }
    """
    try:
        debug_dir = get_debug_dir()
        success_messages = []
        
        # 清理调试目录
        if os.path.exists(debug_dir):
            files_count = 0
            total_size = 0
            for root, dirs, files in os.walk(debug_dir):
                files_count += len(files)
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            
            total_size_mb = total_size / (1024 * 1024)
            
            # 删除debug目录中的所有内容
            for item in os.listdir(debug_dir):
                item_path = os.path.join(debug_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    logger.error(f"删除 {item_path} 失败: {e}")
            
            success_messages.append(
                f"清理了 {files_count} 个调试文件，释放了 {total_size_mb:.2f} MB 空间"
            )
            logger.info(f"清理调试目录完成: {success_messages[-1]}")
        
        # 清理临时下载目录
        temp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            '..', '..', '..', '..', 'data', 'temp'
        )
        temp_dir = os.path.normpath(temp_dir)
        
        if os.path.exists(temp_dir):
            temp_files_count = 0
            temp_total_size = 0
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isdir(item_path):
                        temp_total_size += sum(
                            os.path.getsize(os.path.join(root, f))
                            for root, dirs, files in os.walk(item_path)
                            for f in files
                        )
                        shutil.rmtree(item_path)
                        temp_files_count += 1
                except Exception as e:
                    logger.error(f"删除临时目录 {item_path} 失败: {e}")
            
            temp_size_mb = temp_total_size / (1024 * 1024)
            if temp_files_count > 0:
                success_messages.append(
                    f"清理了 {temp_files_count} 个临时下载文件，释放了 {temp_size_mb:.2f} MB 空间"
                )
                logger.info(f"清理临时目录完成: {success_messages[-1]}")
        
        message = '; '.join(success_messages) if success_messages else '没有文件需要清理'
        logger.info("文件清理完成")
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"清理文件失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@system_bp.route('/test_params', methods=['POST'])
def test_params():
    """
    测试参数解析功能（用于调试）
    
    接收任意JSON数据并返回，用于验证参数传递是否正确
    
    返回:
        {
            'success': True,
            'received_params': {...},
            'message': '参数接收成功'
        }
    """
    try:
        data = request.get_json()
        logger.info(f"收到测试参数: {data}")
        
        return jsonify({
            'success': True,
            'received_params': data,
            'param_count': len(data) if data else 0,
            'message': '参数接收成功'
        })
    except Exception as e:
        logger.error(f"测试参数失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
