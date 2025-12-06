# src/core/session_manager.py

import os
import json
import logging
import time
import shutil # 用于后续的删除操作
import base64  # 用于图片编解码
from src.shared.path_helpers import resource_path # 需要路径助手

logger = logging.getLogger("SessionManager")

# --- 基础配置 ---
SESSION_BASE_DIR_NAME = "sessions" # 会话保存的基础目录名
METADATA_FILENAME = "session_meta.json" # 会话元数据文件名
IMAGE_FILE_EXTENSION = ".png"  # 存储图片的文件扩展名（新格式）
LEGACY_B64_EXTENSION = ".b64"  # 旧版 Base64 文件扩展名（向后兼容）
IMAGE_DATA_EXTENSION = IMAGE_FILE_EXTENSION  # 当前使用的扩展名

# --- 辅助函数 ---

def _get_session_base_dir():
    """获取存储所有会话的基础目录的绝对路径 (e.g., project_root/data/sessions/)"""
    # 使用 resource_path 获取项目根目录，然后构建路径
    # 注意：我们将 sessions 放在 data 目录下，与 debug 同级
    base_path = resource_path(os.path.join("data", SESSION_BASE_DIR_NAME))
    try:
        os.makedirs(base_path, exist_ok=True) # 确保目录存在
        return base_path
    except OSError as e:
        logger.error(f"无法创建或访问会话基础目录: {base_path} - {e}", exc_info=True)
        # 极端情况下的回退（例如权限问题），可以考虑临时目录，但这里简化处理
        raise # 重新抛出错误，因为这是关键目录

def _get_session_path(session_name):
    """获取指定名称会话的文件夹绝对路径"""
    # 确保 session_name 是字符串类型
    session_name = str(session_name) if session_name is not None else None
    
    if not session_name or "/" in session_name or "\\" in session_name:
        logger.error(f"无效的会话名称: {session_name}")
        return None
    safe_session_name = "".join(c for c in session_name if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_session_name:
        logger.error(f"处理后会话名称为空: {session_name}")
        return None
    return os.path.join(_get_session_base_dir(), safe_session_name)

def _save_image_data(session_folder, image_index, image_type, base64_data):
    """
    将图像数据保存为真实的 PNG 图片文件。

    Args:
        session_folder (str): 此会话的文件夹路径。
        image_index (int): 图像在其列表中的索引。
        image_type (str): 图像类型 ('original', 'translated', 'clean')。
        base64_data (str): Base64 编码的图像数据 (不含 'data:image/...' 前缀)。

    Returns:
        bool: 保存是否成功。
    """
    if not base64_data:
        # logger.debug(f"图像 {image_index} 的 {image_type} 数据为空，跳过保存。")
        return True # 认为是"成功"跳过

    filename = f"image_{image_index}_{image_type}{IMAGE_FILE_EXTENSION}"
    filepath = os.path.join(session_folder, filename)
    try:
        # 将 Base64 解码为二进制数据，保存为真实图片文件
        image_bytes = base64.b64decode(base64_data)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        # logger.debug(f"成功保存图像文件: {filepath}")
        
        # 删除旧的 .b64 文件（如果存在）
        legacy_filepath = os.path.join(session_folder, f"image_{image_index}_{image_type}{LEGACY_B64_EXTENSION}")
        if os.path.exists(legacy_filepath):
            try:
                os.remove(legacy_filepath)
                logger.debug(f"已删除旧版 .b64 文件: {legacy_filepath}")
            except Exception:
                pass  # 删除失败不影响主流程
        
        return True
    except Exception as e:
        logger.error(f"保存图像文件失败: {filepath} - {e}", exc_info=True)
        return False

def _load_image_data(session_folder, image_index, image_type):
    """
    从文件加载图像数据，返回 Base64 编码的字符串。
    支持新版 PNG 文件和旧版 .b64 文件（向后兼容）。

    Args:
        session_folder (str): 此会话的文件夹路径。
        image_index (int): 图像在其列表中的索引。
        image_type (str): 图像类型 ('original', 'translated', 'clean')。

    Returns:
        str or None: Base64 编码的图像数据 (不含前缀)，如果文件不存在或读取失败则返回 None。
    """
    # 优先尝试读取新格式 PNG 文件
    png_filename = f"image_{image_index}_{image_type}{IMAGE_FILE_EXTENSION}"
    png_filepath = os.path.join(session_folder, png_filename)
    
    if os.path.exists(png_filepath):
        try:
            with open(png_filepath, 'rb') as f:
                image_bytes = f.read()
            # 将二进制数据编码为 Base64 字符串返回
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"读取图像文件失败: {png_filepath} - {e}", exc_info=True)
            return None
        
    # 向后兼容：尝试读取旧版 .b64 文件
    legacy_filename = f"image_{image_index}_{image_type}{LEGACY_B64_EXTENSION}"
    legacy_filepath = os.path.join(session_folder, legacy_filename)
    
    if os.path.exists(legacy_filepath):
        try:
            with open(legacy_filepath, 'r', encoding='utf-8') as f:
                logger.debug(f"从旧版 .b64 文件加载图像: {legacy_filepath}")
                return f.read()
        except Exception as e:
            logger.error(f"读取旧版图像数据文件失败: {legacy_filepath} - {e}", exc_info=True)
            return None
        
    # logger.debug(f"图像文件未找到: {png_filepath} 或 {legacy_filepath}")
    return None


def get_image_file_path(session_folder, image_index, image_type):
    """
    获取图像文件的完整路径（用于 URL 访问）。
    优先返回新格式 PNG 文件路径，其次是旧版 .b64 文件路径。

    Args:
        session_folder (str): 此会话的文件夹路径。
        image_index (int): 图像在其列表中的索引。
        image_type (str): 图像类型 ('original', 'translated', 'clean')。

    Returns:
        str or None: 文件的完整路径，如果文件不存在则返回 None。
    """
    # 优先检查新格式 PNG 文件
    png_filename = f"image_{image_index}_{image_type}{IMAGE_FILE_EXTENSION}"
    png_filepath = os.path.join(session_folder, png_filename)
    
    if os.path.exists(png_filepath):
        return png_filepath
        
    # 向后兼容：检查旧版 .b64 文件
    legacy_filename = f"image_{image_index}_{image_type}{LEGACY_B64_EXTENSION}"
    legacy_filepath = os.path.join(session_folder, legacy_filename)
    
    if os.path.exists(legacy_filepath):
        return legacy_filepath
        
    return None


def get_image_url_path(session_name, image_index, image_type):
    """
    获取图像的 URL 路径（相对路径）。

    Args:
        session_name (str): 会话名称。
        image_index (int): 图像在其列表中的索引。
        image_type (str): 图像类型 ('original', 'translated', 'clean')。

    Returns:
        str or None: 图像的 URL 路径，如果文件不存在则返回 None。
    """
    session_folder = _get_session_path(session_name)
    if not session_folder:
        return None
        
    filepath = get_image_file_path(session_folder, image_index, image_type)
    if not filepath:
        return None
        
    # 返回相对于 API 的路径
    filename = os.path.basename(filepath)
    return f"/api/sessions/image/{session_name}/{filename}"


# --- 主要保存函数 ---

def save_session(session_name, session_data):
    """
    保存完整的会话状态到磁盘。

    Args:
        session_name (str): 要保存的会话名称。
        session_data (dict): 包含要保存的所有状态数据的字典，结构应类似：
            {
                "ui_settings": {...}, # 包含模型、字体、修复等设置
                "images": [ # 图片列表
                    {
                        "fileName": "...",
                        "originalDataURL": "data:image/...;base64,...", # 原始图 Base64
                        "translatedDataURL": "data:image/...;base64,...", # 翻译图 Base64 (可能为 null)
                        "cleanImageData": "...", # 干净背景 Base64 (可能为 null)
                        "bubbleCoords": [[...], ...],
                        "originalTexts": ["...", ...],
                        "bubbleTexts": ["...", ...],
                        "textboxTexts": ["...", ...],
                        "bubbleStates": { # 气泡独立样式 (可能为 null)
                            "0": {...}, "1": {...}
                        },
                        # 其他图片相关状态...
                    },
                    ...
                ],
                "currentImageIndex": index
            }

    Returns:
        bool: 保存是否成功。
    """
    logger.info(f"开始保存会话: {session_name}")
    session_folder = _get_session_path(session_name)
    if not session_folder:
        return False

    try:
        # 1. 创建会话文件夹 (如果已存在，先删除旧内容还是覆盖？这里选择覆盖)
        if os.path.exists(session_folder):
            logger.warning(f"会话 '{session_name}' 已存在，将覆盖。")
            # 可以选择删除旧文件夹或只覆盖文件，这里简单覆盖
            # shutil.rmtree(session_folder) # 如果需要完全清空
        os.makedirs(session_folder, exist_ok=True)

        # 2. 准备元数据 (排除大的 Base64 字符串)
        metadata_to_save = {
            "metadata": {
                "name": session_name,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "translator_version": "2.1.3+" # 或者从某处获取版本号
            },
            "ui_settings": session_data.get("ui_settings", {}),
            "images_meta": [], # 存储图片元数据，不含 Base64
            "currentImageIndex": session_data.get("currentImageIndex", -1)
        }

        images_data = session_data.get("images", [])
        all_image_data_saved = True

        # 3. 遍历图片，保存 Base64 数据到文件，并准备图片元数据
        for idx, img_state in enumerate(images_data):
            image_meta = img_state.copy() # 复制一份元数据

            # 提取并保存 Original Image Data
            original_b64 = None
            if img_state.get('originalDataURL'):
                try:
                    original_b64 = img_state['originalDataURL'].split(',', 1)[1]
                except IndexError:
                    logger.warning(f"图像 {idx} 的 originalDataURL 格式无效，跳过保存。")
                if not _save_image_data(session_folder, idx, 'original', original_b64):
                    all_image_data_saved = False
            del image_meta['originalDataURL'] # 从元数据中移除

            # 提取并保存 Translated Image Data
            translated_b64 = None
            if img_state.get('translatedDataURL'):
                try:
                    translated_b64 = img_state['translatedDataURL'].split(',', 1)[1]
                except IndexError:
                    logger.warning(f"图像 {idx} 的 translatedDataURL 格式无效，跳过保存。")
                if not _save_image_data(session_folder, idx, 'translated', translated_b64):
                    all_image_data_saved = False
            del image_meta['translatedDataURL'] # 从元数据中移除

            # 提取并保存 Clean Image Data
            clean_b64 = img_state.get('cleanImageData') # 假设这个已经是纯 Base64
            if not _save_image_data(session_folder, idx, 'clean', clean_b64):
                all_image_data_saved = False
            if 'cleanImageData' in image_meta:
                 del image_meta['cleanImageData'] # 从元数据中移除

            # (可选) 可以在 image_meta 中添加标记，指示哪些文件被保存了
            image_meta['hasOriginalData'] = bool(original_b64)
            image_meta['hasTranslatedData'] = bool(translated_b64)
            image_meta['hasCleanData'] = bool(clean_b64)

            metadata_to_save["images_meta"].append(image_meta)

        if not all_image_data_saved:
            logger.warning(f"会话 '{session_name}': 部分图像数据保存失败。元数据仍会保存。")
            # 这里可以选择是否继续保存元数据，或者直接返回 False
            # return False # 如果希望任何图像保存失败都导致整个会话保存失败

        # 4. 保存元数据 JSON 文件
        metadata_filepath = os.path.join(session_folder, METADATA_FILENAME)
        try:
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"成功保存会话元数据: {metadata_filepath}")
        except IOError as e:
            logger.error(f"保存会话元数据文件失败: {metadata_filepath} - {e}", exc_info=True)
            return False # 元数据保存失败，则整个会话保存失败
        except Exception as e:
             logger.error(f"写入会话元数据时发生未知错误: {metadata_filepath} - {e}", exc_info=True)
             return False

        logger.info(f"会话 '{session_name}' 已成功保存到: {session_folder}")
        return True

    except Exception as e:
        logger.error(f"保存会话 '{session_name}' 时发生未知错误: {e}", exc_info=True)
        # 尝试清理可能已创建的文件夹/文件
        try:
            if os.path.exists(session_folder):
                shutil.rmtree(session_folder)
        except Exception as cleanup_e:
            logger.error(f"清理失败的会话文件夹时出错: {session_folder} - {cleanup_e}")
        return False

def load_session(session_name):
    """
    从磁盘加载指定的会话状态。

    Args:
        session_name (str): 要加载的会话名称。

    Returns:
        dict or None: 包含完整会话状态的字典 (结构与 save_session 接收的 session_data 类似)，
                      如果会话不存在或加载失败则返回 None。
    """
    logger.info(f"开始加载会话: {session_name}")
    session_folder = _get_session_path(session_name)
    if not session_folder or not os.path.isdir(session_folder):
        logger.error(f"会话文件夹未找到或不是有效目录: {session_folder}")
        return None

    metadata_filepath = os.path.join(session_folder, METADATA_FILENAME)
    if not os.path.exists(metadata_filepath):
        logger.error(f"会话元数据文件未找到: {metadata_filepath}")
        return None

    try:
        # 1. 加载元数据 JSON 文件
        with open(metadata_filepath, 'r', encoding='utf-8') as f:
            session_meta_data = json.load(f)
        logger.info(f"成功加载会话元数据: {metadata_filepath}")

        # 2. 准备要返回的完整会话数据结构
        session_data_to_return = {
            "ui_settings": session_meta_data.get("ui_settings", {}),
            "images": [], # 稍后填充
            "currentImageIndex": session_meta_data.get("currentImageIndex", -1)
            # 可以考虑把 metadata 也包含进去，供前端显示
            # "metadata": session_meta_data.get("metadata", {})
        }

        images_meta = session_meta_data.get("images_meta", [])
        all_images_loaded = True

        # 3. 遍历图片元数据，加载对应的 Base64 数据
        for idx, img_meta in enumerate(images_meta):
            loaded_img_state = img_meta.copy() # 复制元数据

            # 加载 Original Image Data (如果元数据标记存在)
            if img_meta.get('hasOriginalData'):
                original_b64 = _load_image_data(session_folder, idx, 'original')
                if original_b64 is not None:
                    loaded_img_state['originalDataURL'] = f"data:image/png;base64,{original_b64}" # 加上前缀
                else:
                    logger.warning(f"会话 '{session_name}', 图像 {idx}: 标记有原始数据但文件加载失败。")
                    loaded_img_state['originalDataURL'] = None # 明确设为 None
                    all_images_loaded = False
            else:
                loaded_img_state['originalDataURL'] = None

            # 加载 Translated Image Data
            if img_meta.get('hasTranslatedData'):
                translated_b64 = _load_image_data(session_folder, idx, 'translated')
                if translated_b64 is not None:
                    loaded_img_state['translatedDataURL'] = f"data:image/png;base64,{translated_b64}"
                else:
                    logger.warning(f"会话 '{session_name}', 图像 {idx}: 标记有翻译数据但文件加载失败。")
                    loaded_img_state['translatedDataURL'] = None
                    all_images_loaded = False
            else:
                 loaded_img_state['translatedDataURL'] = None

            # 加载 Clean Image Data
            if img_meta.get('hasCleanData'):
                clean_b64 = _load_image_data(session_folder, idx, 'clean')
                if clean_b64 is not None:
                    loaded_img_state['cleanImageData'] = clean_b64 # 这个已经是纯 Base64
                else:
                    logger.warning(f"会话 '{session_name}', 图像 {idx}: 标记有干净背景数据但文件加载失败。")
                    loaded_img_state['cleanImageData'] = None
                    all_images_loaded = False
            else:
                loaded_img_state['cleanImageData'] = None

            # 移除辅助标记
            loaded_img_state.pop('hasOriginalData', None)
            loaded_img_state.pop('hasTranslatedData', None)
            loaded_img_state.pop('hasCleanData', None)

            session_data_to_return["images"].append(loaded_img_state)

        if not all_images_loaded:
            logger.warning(f"会话 '{session_name}': 部分图像数据加载失败。")
            # 即使部分失败，仍然返回已加载的数据

        logger.info(f"会话 '{session_name}' 加载完成，共加载 {len(session_data_to_return['images'])} 张图片状态。")
        return session_data_to_return

    except json.JSONDecodeError as e:
        logger.error(f"解析会话元数据文件失败: {metadata_filepath} - {e}", exc_info=True)
        return None
    except IOError as e:
         logger.error(f"读取会话元数据文件时出错: {metadata_filepath} - {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"加载会话 '{session_name}' 时发生未知错误: {e}", exc_info=True)
        return None

# --- 添加列出和删除会话的函数 ---

def list_sessions():
    """
    列出所有已保存的会话名称和元数据。

    Returns:
        list: 包含每个会话元数据字典的列表，例如:
              [
                  {"name": "session1", "saved_at": "...", "image_count": 5},
                  {"name": "session2", "saved_at": "...", "image_count": 10},
                  ...
              ]
              如果出错或没有会话，返回空列表。
    """
    logger.info("开始列出已保存的会话...")
    session_base_dir = _get_session_base_dir()
    sessions_list = []

    try:
        if not os.path.isdir(session_base_dir):
            logger.info("会话基础目录不存在，没有已保存的会话。")
            return []

        for item_name in os.listdir(session_base_dir):
            item_path = os.path.join(session_base_dir, item_name)
            # 检查是否是目录，并且包含元数据文件
            if os.path.isdir(item_path):
                metadata_filepath = os.path.join(item_path, METADATA_FILENAME)
                if os.path.exists(metadata_filepath):
                    try:
                        with open(metadata_filepath, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        # 提取需要的信息
                        session_info = {
                            "name": meta_data.get("metadata", {}).get("name", item_name), # 优先用元数据里的名字
                            "saved_at": meta_data.get("metadata", {}).get("saved_at", "未知时间"),
                            "image_count": len(meta_data.get("images_meta", [])),
                            "version": meta_data.get("metadata", {}).get("translator_version", "未知版本")
                        }
                        sessions_list.append(session_info)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"无法读取或解析会话 '{item_name}' 的元数据文件 {metadata_filepath}: {e}")
                    except Exception as e:
                        logger.warning(f"处理会话 '{item_name}' 时发生未知错误: {e}")

        logger.info(f"找到 {len(sessions_list)} 个有效会话。")
        # 按保存时间降序排序（可选）
        sessions_list.sort(key=lambda s: s.get("saved_at", ""), reverse=True)
        return sessions_list

    except Exception as e:
        logger.error(f"列出保存的会话时出错: {e}", exc_info=True)
        return []

def delete_session(session_name):
    """
    删除指定名称的会话（包含所有文件和文件夹）。

    Args:
        session_name (str): 要删除的会话名称。

    Returns:
        bool: 操作是否成功。
    """
    logger.info(f"请求删除会话: {session_name}")
    session_folder = _get_session_path(session_name)
    if not session_folder or not os.path.isdir(session_folder):
        logger.error(f"删除失败: 找不到有效的会话文件夹：{session_name} (路径: {session_folder})")
        return False
    
    try:
        # 整个文件夹删除
        shutil.rmtree(session_folder)
        logger.info(f"成功删除会话文件夹: {session_folder}")
        return True
    except OSError as e:
        logger.error(f"删除会话文件夹失败: {session_folder} - {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"删除会话时发生未知错误: {e}", exc_info=True)
        return False

def rename_session(old_name, new_name):
    """
    重命名一个已保存的会话。

    Args:
        old_name (str): 当前的会话名称。
        new_name (str): 新的会话名称。

    Returns:
        bool: 重命名是否成功。
    """
    logger.info(f"请求重命名会话: 从 '{old_name}' 到 '{new_name}'")

    # 1. 验证新名称的有效性
    if not new_name or not isinstance(new_name, str) or "/" in new_name or "\\" in new_name:
        logger.error(f"重命名失败：无效的新会话名称 '{new_name}'")
        return False
    safe_new_name = "".join(c for c in new_name if c.isalnum() or c in ('_', '-')).rstrip()
    if not safe_new_name:
        logger.error(f"重命名失败：处理后的新会话名称为空 '{new_name}'")
        return False

    # 2. 获取旧的和新的文件夹路径
    old_folder = _get_session_path(old_name)
    new_folder = _get_session_path(new_name) # 使用处理过的 safe_new_name 来获取路径

    if not old_folder or not os.path.isdir(old_folder):
        logger.error(f"重命名失败：找不到旧会话文件夹 '{old_name}' (路径: {old_folder})")
        return False

    if not new_folder: # _get_session_path 内部已处理 new_name 无效的情况
        logger.error(f"重命名失败：无法生成有效的新会话路径 '{new_name}'")
        return False

    # 3. 检查新名称是否已存在
    if os.path.exists(new_folder):
        logger.error(f"重命名失败：新会话名称 '{new_name}' 已存在 (路径: {new_folder})")
        return False # 或者根据需求决定是否覆盖，但通常不允许重命名为已存在的

    try:
        # 4. 重命名文件夹
        os.rename(old_folder, new_folder)
        logger.info(f"成功重命名文件夹: 从 {old_folder} 到 {new_folder}")

        # 5. (重要) 更新元数据文件中的名称
        metadata_filepath = os.path.join(new_folder, METADATA_FILENAME)
        if os.path.exists(metadata_filepath):
            try:
                with open(metadata_filepath, 'r+', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    # 更新元数据中的 name 字段
                    if "metadata" in meta_data:
                        meta_data["metadata"]["name"] = new_name # 使用用户输入的原 new_name
                    else:
                        meta_data["metadata"] = {"name": new_name}
                    # 将文件指针移到开头，清空文件，然后写入新内容
                    f.seek(0)
                    f.truncate()
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                logger.info(f"成功更新元数据文件中的会话名称为 '{new_name}'")
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.error(f"重命名文件夹后，更新元数据文件失败: {metadata_filepath} - {e}", exc_info=True)
                # 可选：尝试将文件夹重命名回去以保持一致性？
                # os.rename(new_folder, old_folder)
                return False # 更新元数据失败也算失败
        else:
            logger.warning(f"重命名会话 '{new_name}' 时未找到元数据文件进行更新: {metadata_filepath}")
            # 即使元数据文件不存在，文件夹已重命名，可以认为部分成功或忽略

        return True # 文件夹重命名成功（即使元数据更新失败也可能返回 True）

    except OSError as e:
        logger.error(f"重命名会话文件夹失败: 从 '{old_folder}' 到 '{new_folder}' - {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"重命名会话时发生未知错误: {e}", exc_info=True)
        return False

# --- 按路径加载/保存会话（用于书籍/章节） ---

def load_session_by_path(session_path):
    """
    按路径加载会话数据（用于书籍/章节功能）。
    
    Args:
        session_path (str): 会话路径，相对于 data/sessions 目录
        
    Returns:
        dict or None: 包含完整会话状态的字典，如果加载失败则返回 None
    """
    logger.info(f"开始按路径加载会话: {session_path}")
    
    if not session_path:
        logger.error("无效的会话路径: 路径为空")
        return None
    
    # 构建完整路径
    base_dir = _get_session_base_dir()
    full_path = os.path.join(base_dir, session_path)
    
    # 安全检查：确保路径在 sessions 目录内
    real_base = os.path.realpath(base_dir)
    real_path = os.path.realpath(full_path)
    if not real_path.startswith(real_base):
        logger.error(f"安全检查失败：路径 {session_path} 超出会话目录范围")
        return None
    
    if not os.path.isdir(full_path):
        logger.warning(f"会话文件夹不存在: {full_path}")
        return None
    
    metadata_filepath = os.path.join(full_path, METADATA_FILENAME)
    if not os.path.exists(metadata_filepath):
        logger.warning(f"会话元数据文件未找到: {metadata_filepath}")
        return None
    
    try:
        with open(metadata_filepath, 'r', encoding='utf-8') as f:
            session_meta_data = json.load(f)
        
        session_data_to_return = {
            "ui_settings": session_meta_data.get("ui_settings", {}),
            "images": [],
            "currentImageIndex": session_meta_data.get("currentImageIndex", -1)
        }
        
        images_meta = session_meta_data.get("images_meta", [])
        
        for idx, img_meta in enumerate(images_meta):
            loaded_img_state = img_meta.copy()
            
            # 返回图片 URL 而不是 Base64 数据，避免大数据传输
            if img_meta.get('hasOriginalData'):
                loaded_img_state['originalDataURL'] = f"/api/sessions/image_by_path/{session_path}/image_{idx}_original.png"
            else:
                loaded_img_state['originalDataURL'] = None
            
            if img_meta.get('hasTranslatedData'):
                loaded_img_state['translatedDataURL'] = f"/api/sessions/image_by_path/{session_path}/image_{idx}_translated.png"
            else:
                loaded_img_state['translatedDataURL'] = None
            
            if img_meta.get('hasCleanData'):
                # cleanImageData 也改为 URL
                loaded_img_state['cleanImageData'] = f"/api/sessions/image_by_path/{session_path}/image_{idx}_clean.png"
            else:
                loaded_img_state['cleanImageData'] = None
            
            loaded_img_state.pop('hasOriginalData', None)
            loaded_img_state.pop('hasTranslatedData', None)
            loaded_img_state.pop('hasCleanData', None)
            
            session_data_to_return["images"].append(loaded_img_state)
        
        logger.info(f"会话路径 '{session_path}' 加载完成")
        return session_data_to_return
        
    except Exception as e:
        logger.error(f"按路径加载会话时发生错误: {e}", exc_info=True)
        return None


def save_session_by_path(session_path, session_data):
    """按路径保存会话数据（用于书籍/章节功能）"""
    logger.info(f"开始按路径保存会话: {session_path}")
    
    if not session_path:
        return False
    
    base_dir = _get_session_base_dir()
    full_path = os.path.join(base_dir, session_path)
    
    try:
        os.makedirs(full_path, exist_ok=True)
        
        images_data = session_data.get("images", [])
        images_meta = []
        
        for idx, img_state in enumerate(images_data):
            img_meta = {}
            
            original_url = img_state.get("originalDataURL")
            if original_url and isinstance(original_url, str) and "base64," in original_url:
                _, base64_part = original_url.split(",", 1)
                if _save_image_data(full_path, idx, 'original', base64_part):
                    img_meta['hasOriginalData'] = True
            
            translated_url = img_state.get("translatedDataURL")
            if translated_url and isinstance(translated_url, str) and "base64," in translated_url:
                _, base64_part = translated_url.split(",", 1)
                if _save_image_data(full_path, idx, 'translated', base64_part):
                    img_meta['hasTranslatedData'] = True
            
            clean_data = img_state.get("cleanImageData")
            if clean_data and isinstance(clean_data, str):
                if _save_image_data(full_path, idx, 'clean', clean_data):
                    img_meta['hasCleanData'] = True
            
            for key in img_state:
                if key not in ('originalDataURL', 'translatedDataURL', 'cleanImageData'):
                    img_meta[key] = img_state[key]
            
            images_meta.append(img_meta)
        
        metadata_to_save = {
            "metadata": {"name": session_path, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "image_count": len(images_meta)},
            "ui_settings": session_data.get("ui_settings", {}),
            "images_meta": images_meta,
            "currentImageIndex": session_data.get("currentImageIndex", -1)
        }
        
        with open(os.path.join(full_path, METADATA_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"会话路径 '{session_path}' 保存成功")
        return True
        
    except Exception as e:
        logger.error(f"按路径保存会话时发生错误: {e}", exc_info=True)
        return False


# --- 测试代码 (更新) ---
if __name__ == '__main__':
    # 创建一些模拟数据进行测试
    print("--- 测试 Session Manager (保存功能) ---")
    # 模拟前端发送的数据结构
    mock_session_data = {
        "ui_settings": {
            "modelProvider": "ollama",
            "modelName": "llama3",
            "fontSize": 28,
            "autoFontSize": False,
            "fontFamily": "fonts/msyh.ttc",
            "layoutDirection": "vertical",
            "useInpainting": "lama",
            "fillColor": "#EEEEEE",
            "textColor": "#111111",
            "rotationAngle": -5,
        },
        "images": [
            {
                "fileName": "page_01.png",
                "originalDataURL": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", # 1x1 Red Pixel
                "translatedDataURL": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", # 1x1 Red Pixel (mock)
                "cleanImageData": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPgvAQAD_gGLKKs5MgAAAABJRU5ErkJggg==", # 1x1 Black Pixel (mock)
                "bubbleCoords": [[10, 10, 50, 50]],
                "originalTexts": ["Hello"],
                "bubbleTexts": ["你好"],
                "textboxTexts": ["你好 (解释...)"],
                "bubbleStates": { "0": {"fontSize": 28, "autoFontSize": False, "fontFamily": "fonts/msyh.ttc", "textDirection": "vertical", "position_offset": {"x": 0,"y": 0}, "textColor": "#111111", "rotationAngle": -5}},
                "translationFailed": False,
                # ... 其他状态 ...
            },
            {
                "fileName": "page_02.jpg",
                "originalDataURL": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QAiRXhpZgAASUkqAAgAAAABABIBAwABAAAABgASAAAAAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKpgA//Z", # 1x1 White Pixel
                "translatedDataURL": None, # 未翻译
                "cleanImageData": None,
                "bubbleCoords": [[20, 30, 80, 90]],
                "originalTexts": ["World"],
                "bubbleTexts": [""],
                "textboxTexts": [""],
                "bubbleStates": None,
                "translationFailed": False,
                # ... 其他状态 ...
            }
        ],
        "currentImageIndex": 0
    }

    test_session_name = "my_first_test_session"
    print(f"尝试保存会话: {test_session_name}")
    success = save_session(test_session_name, mock_session_data)

    if success:
        print("保存成功！请检查以下目录和文件：")
        session_path = _get_session_path(test_session_name)
        print(f"  - 文件夹: {session_path}")
        print(f"  - 元数据文件: {os.path.join(session_path, METADATA_FILENAME)}")
        print(f"  - 图像文件 (示例):")
        print(f"    - {os.path.join(session_path, 'image_0_original' + IMAGE_DATA_EXTENSION)}")
        print(f"    - {os.path.join(session_path, 'image_0_translated' + IMAGE_DATA_EXTENSION)}")
        print(f"    - {os.path.join(session_path, 'image_0_clean' + IMAGE_DATA_EXTENSION)}")
        print(f"    - {os.path.join(session_path, 'image_1_original' + IMAGE_DATA_EXTENSION)}")
        # ... 应该没有 image_1_translated 和 image_1_clean
    else:
        print("保存失败。")

    # 测试无效名称
    print("\n尝试保存无效名称:")
    save_session("invalid/name", mock_session_data)
    save_session("", mock_session_data)

    print("\n--- 测试 Session Manager (加载、列出、删除功能) ---")

    # 1. 测试列出
    print("\n测试列出所有会话:")
    sessions = list_sessions()
    if sessions:
        print("找到以下会话:")
        for s in sessions:
            print(f"  - 名称: {s['name']}, 保存时间: {s['saved_at']}, 图片数: {s['image_count']}, 版本: {s['version']}")
    else:
        print("未找到任何已保存的会话。")

    # 2. 测试加载 (使用之前保存的名称)
    if test_session_name in [s['name'] for s in sessions]:
        print(f"\n测试加载会话: {test_session_name}")
        loaded_data = load_session(test_session_name)
        if loaded_data:
            print("加载成功！")
            print("  UI 设置:", loaded_data.get("ui_settings"))
            print(f"  图片数量: {len(loaded_data.get('images', []))}")
            # 检查第一张图片的 DataURL 是否已恢复
            if loaded_data.get('images') and loaded_data['images'][0].get('originalDataURL'):
                print("  第一张图片的 originalDataURL 已恢复。")
            else:
                print("  警告：第一张图片的 originalDataURL 未恢复。")
            print(f"  当前索引: {loaded_data.get('currentImageIndex')}")
        else:
            print("加载失败。")
    else:
        print(f"\n跳过加载测试，因为未找到会话 '{test_session_name}'。")

    # 3. 测试删除 (可选，会实际删除文件)
    # print(f"\n测试删除会话: {test_session_name}")
    # if test_session_name in [s['name'] for s in list_sessions()]: # 重新获取列表检查
    #     delete_confirm = input(f"真的要删除会话 '{test_session_name}' 吗? (yes/no): ")
    #     if delete_confirm.lower() == 'yes':
    #         deleted = delete_session(test_session_name)
    #         print(f"删除结果: {'成功' if deleted else '失败'}")
    #         # 再次列出检查
    #         print("\n删除后再次列出:")
    #         sessions_after_delete = list_sessions()
    #         if sessions_after_delete:
    #              for s in sessions_after_delete: print(f"  - {s['name']}")
    #         else: print("已无会话。")
    #     else:
    #         print("取消删除。")
    # else:
    #      print(f"无法删除，未找到会话 '{test_session_name}'。")


# ============================================================
# 分批保存功能 - 避免前端一次性传输大量 Base64 数据
# ============================================================

def _cleanup_orphan_image_files(session_folder, new_image_count):
    """
    清理超出当前图片数量的旧图片文件。
    用于处理用户删除图片后再保存的情况，避免留下孤立文件。
    
    Args:
        session_folder: 会话文件夹路径
        new_image_count: 新保存的图片数量
    """
    if not os.path.isdir(session_folder):
        return
    
    import re
    # 匹配 image_N_type.png 或 image_N_type.b64 格式的文件
    pattern = re.compile(r'^image_(\d+)_(original|translated|clean)\.(png|b64)$')
    
    deleted_count = 0
    for filename in os.listdir(session_folder):
        match = pattern.match(filename)
        if match:
            file_index = int(match.group(1))
            # 删除索引 >= 新图片数量的文件
            if file_index >= new_image_count:
                filepath = os.path.join(session_folder, filename)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.debug(f"已清理孤立图片文件: {filename}")
                except Exception as e:
                    logger.warning(f"清理孤立文件失败 {filename}: {e}")
    
    if deleted_count > 0:
        logger.info(f"已清理 {deleted_count} 个孤立图片文件")


def start_batch_save(session_path, metadata):
    """
    开始分批保存会话 - 第一步：创建目录并保存元数据（不含图片数据）。
    """
    logger.info(f"开始分批保存会话: {session_path}")
    
    try:
        if "/" in str(session_path) or "\\" in str(session_path):
            base_dir = _get_session_base_dir()
            session_folder = os.path.join(base_dir, session_path)
        else:
            session_folder = _get_session_path(session_path)
        
        if not session_folder:
            return {"success": False, "error": "无效的会话路径"}
        
        os.makedirs(session_folder, exist_ok=True)
        
        # 清理多余的旧图片文件（处理用户删除图片后再保存的情况）
        new_image_count = len(metadata.get("images_meta", []))
        _cleanup_orphan_image_files(session_folder, new_image_count)
        
        metadata_to_save = {
            "metadata": {
                "name": session_path,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "translator_version": "2.1.3+"
            },
            "ui_settings": metadata.get("ui_settings", {}),
            "images_meta": metadata.get("images_meta", []),
            "currentImageIndex": metadata.get("currentImageIndex", -1)
        }
        
        metadata_filepath = os.path.join(session_folder, METADATA_FILENAME)
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分批保存初始化成功: {session_folder}")
        return {"success": True, "session_folder": session_folder}
        
    except Exception as e:
        logger.error(f"分批保存初始化失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def save_single_image(session_folder, image_index, image_type, base64_data):
    """
    分批保存会话 - 第二步：保存单张图片。
    """
    try:
        if not base64_data:
            return {"success": True}
        
        if isinstance(base64_data, str) and ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        
        if _save_image_data(session_folder, image_index, image_type, base64_data):
            return {"success": True}
        else:
            return {"success": False, "error": f"保存图片 {image_index}_{image_type} 失败"}
            
    except Exception as e:
        logger.error(f"保存单张图片失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def complete_batch_save(session_folder, images_meta):
    """
    分批保存会话 - 第三步：完成保存，更新元数据。
    """
    try:
        metadata_filepath = os.path.join(session_folder, METADATA_FILENAME)
        
        if not os.path.exists(metadata_filepath):
            return {"success": False, "error": "元数据文件不存在"}
        
        with open(metadata_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata["images_meta"] = images_meta
        metadata["metadata"]["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["metadata"]["image_count"] = len(images_meta)
        
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分批保存完成: {session_folder}")
        return {"success": True}
        
    except Exception as e:
        logger.error(f"完成分批保存失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}