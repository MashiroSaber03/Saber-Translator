"""
批量下载相关API

包含所有与批量下载相关的API端点：
- 准备批量下载（ZIP/PDF/CBZ）
- 下载处理好的文件
- 清理过期临时文件
"""

import os
import re
import shutil
import base64
import io
import time
import uuid
import zipfile
import logging
from typing import List, Dict, Any
from flask import request, jsonify, send_file
from PIL import Image

from . import system_bp
from src.shared.path_helpers import resource_path
from src.shared.security import validate_relative_path, resolve_under_base

logger = logging.getLogger("SystemAPI.Downloads")


@system_bp.route('/download_start_session', methods=['POST'])
def download_start_session_api():
    """
    开始一个新的下载会话，创建临时目录
    
    请求体:
        {
            'total_images': int  # 总图片数量
        }
    
    返回:
        {
            'success': True,
            'session_id': 'uuid'
        }
    """
    try:
        data = request.json or {}
        total_images = data.get('total_images', 0)
        
        # 创建唯一的会话ID和临时目录
        session_id = str(uuid.uuid4())
        base_path = resource_path('')
        temp_dir = os.path.join(base_path, 'data', 'temp', session_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"创建下载会话: {session_id}, 预期 {total_images} 张图片")
        
        return jsonify({
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"创建下载会话失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'创建下载会话失败: {str(e)}'}), 500


@system_bp.route('/download_upload_image', methods=['POST'])
def download_upload_image_api():
    """
    上传单张图片到下载会话的临时目录
    
    请求体:
        {
            'session_id': 'uuid',
            'image_index': int,
            'image_data': 'base64_data'
        }
    
    返回:
        {
            'success': True,
            'saved_index': int
        }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
            
        session_id = data.get('session_id')
        image_index = data.get('image_index', 0)
        image_data = data.get('image_data')
        file_path = data.get('file_path')  # 可选：文件相对路径（用于保留文件夹结构）
        
        if not session_id or not image_data:
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 验证session_id，防止路径注入
        if '..' in session_id or '/' in session_id or '\\' in session_id:
            return jsonify({'error': '无效的会话ID'}), 400
            
        base_path = resource_path('')
        temp_dir = os.path.join(base_path, 'data', 'temp', session_id)
        
        if not os.path.exists(temp_dir):
            return jsonify({'error': '会话不存在或已过期'}), 404
            
        # 处理Base64数据
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
            
        # 解码Base64数据
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        # 确保是RGB模式，避免调色板模式(P)导致保存后颜色错误
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 确定保存路径
        if file_path:
            # 使用文件路径保存（保留文件夹结构）
            # 安全校验 + 归一化
            ok, normalized_or_error = validate_relative_path(file_path, allow_unicode=True)
            if not ok:
                return jsonify({'error': f'无效的文件路径: {normalized_or_error}'}), 400

            # 将原始扩展名替换为 .png
            name_without_ext = os.path.splitext(normalized_or_error)[0]
            target_rel_path = name_without_ext + '.png'

            # 构建完整路径（强制限制在 temp_dir 下）
            ok, filepath_or_error = resolve_under_base(temp_dir, target_rel_path)
            if not ok:
                return jsonify({'error': f'无效的文件路径: {filepath_or_error}'}), 400
            filepath = filepath_or_error
            
            # 处理同名文件：如果已存在则自动添加序号
            if os.path.exists(filepath):
                base_name = os.path.splitext(target_rel_path)[0]
                counter = 1
                while True:
                    candidate_rel = f"{base_name}_{counter}.png"
                    ok, candidate_or_error = resolve_under_base(temp_dir, candidate_rel)
                    if not ok:
                        return jsonify({'error': f'无效的文件路径: {candidate_or_error}'}), 400
                    if not os.path.exists(candidate_or_error):
                        filepath = candidate_or_error
                        break
                    counter += 1
            
            # 创建子目录（如果需要）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else:
            # 回退到扁平结构（向后兼容）
            filename = f"image_{image_index:03d}.png"
            filepath = os.path.join(temp_dir, filename)
        
        img.save(filepath, format="PNG")
        
        rel_path = os.path.relpath(filepath, temp_dir)
        logger.debug(f"会话 {session_id}: 已保存图片 {rel_path}")
        
        return jsonify({
            'success': True,
            'saved_index': image_index
        })
        
    except Exception as e:
        logger.error(f"上传图片失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'上传图片失败: {str(e)}'}), 500


@system_bp.route('/download_finalize', methods=['POST'])
def download_finalize_api():
    """
    完成下载会话，打包所有已上传的图片
    
    请求体:
        {
            'session_id': 'uuid',
            'format': 'zip' | 'pdf' | 'cbz'
        }
    
    返回:
        {
            'success': True,
            'file_id': 'uuid',
            'format': 'zip',
            'message': '已成功处理 N 张图片'
        }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
            
        session_id = data.get('session_id')
        format_type = data.get('format', 'zip')
        
        if not session_id:
            return jsonify({'error': '缺少会话ID'}), 400
            
        # 验证session_id，防止路径注入
        if '..' in session_id or '/' in session_id or '\\' in session_id:
            return jsonify({'error': '无效的会话ID'}), 400
            
        base_path = resource_path('')
        temp_dir = os.path.join(base_path, 'data', 'temp', session_id)
        
        if not os.path.exists(temp_dir):
            return jsonify({'error': '会话不存在或已过期'}), 404
            
        # 获取所有已保存的图片文件（支持文件夹结构，递归遍历）
        saved_files = []  # 列表元素: (绝对路径, 相对路径)
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith('.png'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, temp_dir).replace('\\', '/')
                    saved_files.append((full_path, rel_path))
        
        # 按相对路径自然排序（支持多文件夹场景）
        def _natural_sort_key(item):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', item[1])]
        saved_files.sort(key=_natural_sort_key)
        
        if not saved_files:
            logger.error("没有找到任何图片文件")
            return jsonify({'error': '没有找到任何图片文件'}), 500
            
        logger.info(f"会话 {session_id}: 准备打包 {len(saved_files)} 张图片，格式: {format_type}")
        
        # 根据请求的格式类型打包文件
        output_path = ""
        if format_type in ('zip', 'cbz'):
            ext = '.cbz' if format_type == 'cbz' else '.zip'
            output_path = os.path.join(temp_dir, f"comic_translator_images{ext}")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for full_path, rel_path in saved_files:
                    zipf.write(full_path, rel_path)  # 保留文件夹结构
                    
            logger.info(f"已创建{format_type.upper()}文件: {output_path}")
            
        elif format_type == 'pdf':
            output_path = os.path.join(temp_dir, "comic_translator_images.pdf")
            
            # 使用PIL创建PDF（PDF 不支持文件夹层级，按排序后的顺序排列页面）
            images = []
            for full_path, rel_path in saved_files:
                try:
                    img = Image.open(full_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.error(f"加载图片到PDF失败: {rel_path} - {str(e)}")
            
            if images:
                first_image = images[0]
                if len(images) > 1:
                    first_image.save(output_path, save_all=True, append_images=images[1:])
                else:
                    first_image.save(output_path)
                logger.info(f"已创建PDF文件: {output_path}")
            else:
                logger.error("无法创建PDF：没有有效图像")
                return jsonify({'error': '创建PDF失败：没有有效图像'}), 500
        else:
            logger.error(f"不支持的格式类型: {format_type}")
            return jsonify({'error': f'不支持的格式类型: {format_type}'}), 400
            
        # 检查是否包含文件夹结构
        has_folders = any('/' in rel for _, rel in saved_files)
        
        return jsonify({
            'success': True,
            'message': f'已成功处理 {len(saved_files)} 张图片' + ('（保留文件夹结构）' if has_folders else ''),
            'file_id': session_id,
            'format': format_type
        })
        
    except Exception as e:
        logger.error(f"打包下载文件失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'打包下载文件失败: {str(e)}'}), 500


@system_bp.route('/download_all_images', methods=['POST'])
def download_all_images_api():
    """
    [已废弃] 旧的一次性上传所有图片的API，保留向后兼容
    建议使用新的分步API: download_start_session -> download_upload_image -> download_finalize
    """
    logger.warning("使用了已废弃的 download_all_images API，建议升级到分步上传API")
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
            
        format_type = data.get('format', 'zip')
        image_data_list = data.get('images', [])
        
        if not image_data_list:
            return jsonify({'error': '没有提供图片数据'}), 400
            
        logger.info(f"准备处理 {len(image_data_list)} 张图片，格式: {format_type}")
            
        # 创建唯一的临时目录
        unique_id = str(uuid.uuid4())
        base_path = resource_path('')
        temp_dir = os.path.join(base_path, 'data', 'temp', unique_id)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"创建临时目录: {temp_dir}")
        
        # 保存所有图片到临时目录
        saved_files = []
        for i, img_data in enumerate(image_data_list):
            if not img_data:
                logger.warning(f"跳过索引 {i} 的空图片数据")
                continue
                
            try:
                # 处理Base64数据
                if ',' in img_data:
                    img_data = img_data.split(',', 1)[1]
                    
                # 解码Base64数据
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                # 确保是RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 保存图片到临时目录
                filename = f"image_{i:03d}.png"
                filepath = os.path.join(temp_dir, filename)
                img.save(filepath, format="PNG")
                saved_files.append(filepath)
                logger.debug(f"已保存图片 {i+1}/{len(image_data_list)}: {filename}")
            except Exception as e:
                logger.error(f"保存图片 {i} 失败: {str(e)}")
        
        if not saved_files:
            logger.error("没有成功保存任何图片")
            return jsonify({'error': '所有图片处理失败'}), 500
            
        logger.info(f"成功保存 {len(saved_files)}/{len(image_data_list)} 张图片")
        
        # 根据请求的格式类型打包文件
        output_path = ""
        if format_type in ('zip', 'cbz'):
            ext = '.cbz' if format_type == 'cbz' else '.zip'
            output_path = os.path.join(temp_dir, f"comic_translator_images{ext}")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in saved_files:
                    zipf.write(file, os.path.basename(file))
                    
            logger.info(f"已创建{format_type.upper()}文件: {output_path}")
            
        elif format_type == 'pdf':
            output_path = os.path.join(temp_dir, "comic_translator_images.pdf")
            
            # 使用PIL创建PDF
            images = []
            for file in saved_files:
                try:
                    img = Image.open(file).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.error(f"加载图片到PDF失败: {file} - {str(e)}")
            
            if images:
                first_image = images[0]
                if len(images) > 1:
                    first_image.save(output_path, save_all=True, append_images=images[1:])
                else:
                    first_image.save(output_path)
                logger.info(f"已创建PDF文件: {output_path}")
            else:
                logger.error("无法创建PDF：没有有效图像")
                return jsonify({'error': '创建PDF失败：没有有效图像'}), 500
        else:
            logger.error(f"不支持的格式类型: {format_type}")
            return jsonify({'error': f'不支持的格式类型: {format_type}'}), 400
            
        return jsonify({
            'success': True,
            'message': f'已成功处理 {len(saved_files)} 张图片',
            'file_id': unique_id,
            'format': format_type
        })
        
    except Exception as e:
        logger.error(f"批量下载处理失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'批量下载处理失败: {str(e)}'}), 500


@system_bp.route('/download_file/<file_id>', methods=['GET'])
def download_processed_file_api(file_id: str):
    """
    下载之前处理好的文件
    
    URL参数:
        file_id: 文件唯一标识符
        format: 文件格式 (zip/pdf/cbz)
    
    返回:
        文件流
    """
    format_type = request.args.get('format', 'zip')
    
    try:
        # 验证file_id，防止路径注入
        if not file_id or '..' in file_id or '/' in file_id or '\\' in file_id:
            logger.error(f"无效的file_id: {file_id}")
            return jsonify({'error': '无效的文件ID'}), 400
            
        # 构建文件路径
        base_path = resource_path('')
        temp_dir = os.path.join(base_path, 'data', 'temp', file_id)
        
        if not os.path.exists(temp_dir):
            logger.error(f"临时目录不存在: {temp_dir}")
            return jsonify({'error': '请求的文件不存在或已过期'}), 404
            
        # 根据格式类型确定文件名和MIME类型
        if format_type == 'zip':
            filename = "comic_translator_images.zip"
            file_path = os.path.join(temp_dir, filename)
            mime_type = "application/zip"
        elif format_type == 'cbz':
            filename = "comic_translator_images.cbz"
            file_path = os.path.join(temp_dir, filename)
            mime_type = "application/x-cbz"
        elif format_type == 'pdf':
            filename = "comic_translator_images.pdf"
            file_path = os.path.join(temp_dir, filename)
            mime_type = "application/pdf"
        else:
            logger.error(f"不支持的格式: {format_type}")
            return jsonify({'error': f'不支持的格式: {format_type}'}), 400
            
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return jsonify({'error': '请求的文件不存在或已过期'}), 404
            
        logger.info(f"发送文件: {filename}")
        
        # 返回文件
        return send_file(
            file_path, 
            mimetype=mime_type,
            as_attachment=True, 
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'下载文件失败: {str(e)}'}), 500


@system_bp.route('/clean_temp_files', methods=['POST'])
def clean_temp_files_api():
    """
    清理临时下载文件夹中的过期文件
    
    过期时间: 24小时
    
    返回:
        {
            'success': True,
            'message': '已清理 N 个过期临时目录'
        }
    """
    try:
        base_path = resource_path('')
        temp_base_dir = os.path.join(base_path, 'data', 'temp')
        
        if not os.path.exists(temp_base_dir):
            logger.info(f"临时目录不存在，无需清理: {temp_base_dir}")
            return jsonify({'success': True, 'message': '临时目录不存在，无需清理'})
            
        # 获取当前时间
        current_time = time.time()
        # 设置过期时间为24小时
        expiry_time = current_time - (24 * 60 * 60)
        removed_count = 0
        total_size = 0
        
        # 遍历所有子目录
        for dir_name in os.listdir(temp_base_dir):
            dir_path = os.path.join(temp_base_dir, dir_name)
            if os.path.isdir(dir_path):
                # 检查目录的创建时间
                dir_creation_time = os.path.getctime(dir_path)
                if dir_creation_time < expiry_time:
                    # 计算目录大小
                    try:
                        dir_size = sum(
                            os.path.getsize(os.path.join(root, f))
                            for root, dirs, files in os.walk(dir_path)
                            for f in files
                        )
                        total_size += dir_size
                        
                        # 删除过期目录
                        shutil.rmtree(dir_path)
                        removed_count += 1
                        logger.info(f"已删除过期临时目录: {dir_path}")
                    except Exception as e:
                        logger.error(f"删除临时目录失败: {dir_path} - {str(e)}")
        
        size_mb = total_size / (1024 * 1024)
        message = f'已清理 {removed_count} 个过期临时目录'
        if size_mb > 0:
            message += f'，释放了 {size_mb:.2f} MB 空间'
            
        logger.info(message)
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"清理临时文件失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
