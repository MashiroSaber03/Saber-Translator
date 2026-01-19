# src/core/session_manager.py

import os
import json
import logging
import time
import shutil # 用于后续的删除操作
import base64  # 用于图片编解码
from src.shared.path_helpers import resource_path # 需要路径助手

logger = logging.getLogger("SessionManager")

# 注意：旧的保存相关的线程锁已移除，新的锁在 page_storage.py 中

# --- 基础配置 ---
SESSION_BASE_DIR_NAME = "sessions" # 会话保存的基础目录名
METADATA_FILENAME = "session_meta.json" # 会话元数据文件名
IMAGE_FILE_EXTENSION = ".png"  # 存储图片的文件扩展名（新格式）
LEGACY_B64_EXTENSION = ".b64"  # 旧版 Base64 文件扩展名（向后兼容）

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

# 注意：旧的 _save_image_data 函数已移除，保存功能请使用 src/core/page_storage.py


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


# --- 主要加载函数 ---
# 注意：旧的 save_session 函数已移除，保存功能请使用 src/core/page_storage.py


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
                        # 支持两种格式：新格式使用 total_pages，旧格式使用 images_meta
                        if "total_pages" in meta_data:
                            image_count = meta_data.get("total_pages", 0)
                        else:
                            image_count = len(meta_data.get("images_meta", []))
                        
                        session_info = {
                            "name": meta_data.get("metadata", {}).get("name", item_name), # 优先用元数据里的名字
                            "saved_at": meta_data.get("metadata", {}).get("saved_at", "未知时间"),
                            "image_count": image_count,
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
    支持两种存储格式：
    - 旧格式：image_N_type.png + images_meta
    - 新格式：images/N/type.png + page meta.json
    
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
        
        # 检查是否是新格式（有 total_pages 字段和 images 子目录）
        images_dir = os.path.join(full_path, "images")
        is_new_format = "total_pages" in session_meta_data and os.path.isdir(images_dir)
        
        if is_new_format:
            # === 新格式加载逻辑 ===
            logger.info(f"检测到新格式存档: {session_path}")
            return _load_new_format_session(session_path, full_path, session_meta_data)
        else:
            # === 旧格式加载逻辑 ===
            logger.info(f"检测到旧格式存档: {session_path}")
            return _load_old_format_session(session_path, session_meta_data)
        
    except Exception as e:
        logger.error(f"按路径加载会话时发生错误: {e}", exc_info=True)
        return None


def _load_new_format_session(session_path, full_path, session_meta_data):
    """
    加载新格式会话（images/{index}/ 子目录结构）
    """
    session_data_to_return = {
        "ui_settings": session_meta_data.get("ui_settings", {}),
        "images": [],
        "currentImageIndex": session_meta_data.get("currentImageIndex", 0)
    }
    
    total_pages = session_meta_data.get("total_pages", 0)
    images_dir = os.path.join(full_path, "images")
    
    for idx in range(total_pages):
        page_dir = os.path.join(images_dir, str(idx))
        page_meta_file = os.path.join(page_dir, "meta.json")
        
        # 加载页面元数据
        loaded_img_state = {}
        if os.path.exists(page_meta_file):
            try:
                with open(page_meta_file, 'r', encoding='utf-8') as f:
                    loaded_img_state = json.load(f)
            except Exception as e:
                logger.warning(f"加载页面元数据失败: {page_meta_file} - {e}")
        
        # 检查并设置图片 URL（新格式路径）
        original_file = os.path.join(page_dir, "original.png")
        if os.path.exists(original_file):
            loaded_img_state['originalDataURL'] = f"/api/sessions/page/{session_path}/{idx}/original"
        else:
            loaded_img_state['originalDataURL'] = None
        
        translated_file = os.path.join(page_dir, "translated.png")
        if os.path.exists(translated_file):
            loaded_img_state['translatedDataURL'] = f"/api/sessions/page/{session_path}/{idx}/translated"
        else:
            loaded_img_state['translatedDataURL'] = None
        
        clean_file = os.path.join(page_dir, "clean.png")
        if os.path.exists(clean_file):
            loaded_img_state['cleanImageData'] = f"/api/sessions/page/{session_path}/{idx}/clean"
        else:
            loaded_img_state['cleanImageData'] = None
        
        # 移除存储标记（前端不需要）
        loaded_img_state.pop('hasOriginal', None)
        loaded_img_state.pop('hasTranslated', None)
        loaded_img_state.pop('hasClean', None)
        
        session_data_to_return["images"].append(loaded_img_state)
    
    logger.info(f"新格式会话 '{session_path}' 加载完成，共 {total_pages} 页")
    return session_data_to_return


def _load_old_format_session(session_path, session_meta_data):
    """
    加载旧格式会话（image_N_type.png 平铺结构）
    """
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
    
    logger.info(f"旧格式会话 '{session_path}' 加载完成，共 {len(images_meta)} 页")
    return session_data_to_return



# 注意：旧的 save_session_by_path 函数已移除，保存功能请使用 src/core/page_storage.py

# ============================================================
# 注意：旧的保存功能已移除
# 新的单页保存逻辑请使用 src/core/page_storage.py
# 以下函数仅保留加载功能用于兼容旧存档：
# - load_session()
# - load_session_by_path()
# - _load_new_format_session()
# - _load_old_format_session()
# - _load_image_data()
# ============================================================
