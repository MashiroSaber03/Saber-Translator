"""
插件管理相关API

包含所有与插件管理相关的API端点：
- 获取插件列表
- 启用/禁用插件
- 删除插件
- 插件配置管理
- 插件默认状态管理
"""

import os
import shutil
import logging
from typing import Dict, Any
from flask import request, jsonify

from . import system_bp
from src.plugins.manager import get_plugin_manager
from src.plugins.base import PluginBase

logger = logging.getLogger("SystemAPI.Plugins")


@system_bp.route('/plugins', methods=['GET'])
def get_plugins_list():
    """获取所有已加载插件的列表及其状态"""
    try:
        plugin_mgr = get_plugin_manager()
        plugins_data = []
        for plugin_instance in plugin_mgr.get_all_plugins():
            meta = plugin_instance.get_metadata()
            meta['enabled'] = plugin_instance.is_enabled()
            plugins_data.append(meta)
        return jsonify({'success': True, 'plugins': plugins_data})
    except Exception as e:
        logger.error(f"获取插件列表失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '无法获取插件列表'}), 500


@system_bp.route('/plugins/<plugin_name>/enable', methods=['POST'])
def enable_plugin_api(plugin_name: str):
    """启用指定的插件"""
    try:
        plugin_mgr = get_plugin_manager()
        plugin = plugin_mgr.get_plugin(plugin_name)
        if plugin:
            plugin.enable()
            logger.info(f"插件 '{plugin_name}' 已通过 API 启用。")
            return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 已启用。"})
        else:
            logger.warning(f"尝试启用不存在的插件: {plugin_name}")
            return jsonify({'success': False, 'error': '插件未找到'}), 404
    except Exception as e:
        logger.error(f"启用插件 '{plugin_name}' 失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '启用插件时出错'}), 500


@system_bp.route('/plugins/<plugin_name>/disable', methods=['POST'])
def disable_plugin_api(plugin_name: str):
    """禁用指定的插件"""
    try:
        plugin_mgr = get_plugin_manager()
        plugin = plugin_mgr.get_plugin(plugin_name)
        if plugin:
            plugin.disable()
            logger.info(f"插件 '{plugin_name}' 已通过 API 禁用。")
            return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 已禁用。"})
        else:
            logger.warning(f"尝试禁用不存在的插件: {plugin_name}")
            return jsonify({'success': False, 'error': '插件未找到'}), 404
    except Exception as e:
        logger.error(f"禁用插件 '{plugin_name}' 失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '禁用插件时出错'}), 500


@system_bp.route('/plugins/<plugin_name>', methods=['DELETE'])
def delete_plugin_api(plugin_name: str):
    """删除指定的插件（物理删除文件夹）"""
    logger.warning(f"收到删除插件 '{plugin_name}' 的请求。")
    try:
        plugin_mgr = get_plugin_manager()
        plugin = plugin_mgr.get_plugin(plugin_name)

        if not plugin:
            logger.warning(f"尝试删除不存在的插件: {plugin_name}")
            return jsonify({'success': False, 'error': '插件未找到'}), 404

        # 确定插件目录路径
        plugin_path = None
        for p_dir in plugin_mgr.plugin_dirs:
            potential_path = os.path.join(p_dir, plugin_name)
            if os.path.isdir(potential_path):
                plugin_path = potential_path
                break

        if plugin_path and os.path.exists(plugin_path):
            logger.info(f"准备删除插件目录: {plugin_path}")
            try:
                shutil.rmtree(plugin_path)
                logger.info(f"插件目录 '{plugin_path}' 已成功删除。")
                
                # 从管理器中移除插件实例和元数据
                if plugin_name in plugin_mgr.plugins:
                    del plugin_mgr.plugins[plugin_name]
                if plugin_name in plugin_mgr.plugin_metadata:
                    del plugin_mgr.plugin_metadata[plugin_name]
                
                # 清理钩子
                plugin_mgr.unregister_plugin_hooks(plugin_name)

                return jsonify({
                    'success': True, 
                    'message': f"插件 '{plugin_name}' 已删除。建议重启应用以完全移除。"
                })
            except OSError as e:
                logger.error(f"删除插件目录 '{plugin_path}' 失败: {e}", exc_info=True)
                return jsonify({'success': False, 'error': f'删除插件文件失败: {e.strerror}'}), 500
        else:
            logger.error(f"找不到插件 '{plugin_name}' 的目录进行删除。")
            return jsonify({'success': False, 'error': '找不到插件目录'}), 404

    except Exception as e:
        logger.error(f"删除插件 '{plugin_name}' 时发生未知错误: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '删除插件时出错'}), 500


@system_bp.route('/plugins/<plugin_name>/config_schema', methods=['GET'])
def get_plugin_config_schema(plugin_name: str):
    """获取指定插件的配置规范"""
    plugin_mgr = get_plugin_manager()
    plugin = plugin_mgr.get_plugin(plugin_name)
    if plugin and isinstance(plugin, PluginBase):
        schema = plugin.get_config_spec()
        return jsonify({'success': True, 'schema': schema or []})
    else:
        return jsonify({'success': False, 'error': '插件未找到或无效'}), 404


@system_bp.route('/plugins/<plugin_name>/config', methods=['GET'])
def get_plugin_config(plugin_name: str):
    """获取指定插件的当前配置值"""
    plugin_mgr = get_plugin_manager()
    plugin = plugin_mgr.get_plugin(plugin_name)
    if plugin and isinstance(plugin, PluginBase):
        return jsonify({'success': True, 'config': plugin.config})
    else:
        return jsonify({'success': False, 'error': '插件未找到或无效'}), 404


@system_bp.route('/plugins/<plugin_name>/config', methods=['POST'])
def save_plugin_config_api(plugin_name: str):
    """保存指定插件的配置值"""
    data = request.get_json()
    if data is None:
        return jsonify({'success': False, 'error': '请求体必须是 JSON'}), 400

    plugin_mgr = get_plugin_manager()
    if plugin_mgr.save_plugin_config(plugin_name, data):
        return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 的配置已保存。"})
    else:
        if not plugin_mgr.get_plugin(plugin_name):
            return jsonify({'success': False, 'error': '插件未找到'}), 404
        else:
            return jsonify({'success': False, 'error': '保存插件配置失败'}), 500


@system_bp.route('/plugins/default_states', methods=['GET'])
def get_plugin_default_states():
    """获取所有插件的默认启用/禁用状态"""
    try:
        plugin_mgr = get_plugin_manager()
        return jsonify({'success': True, 'default_states': plugin_mgr.plugin_default_states})
    except Exception as e:
        logger.error(f"获取插件默认状态失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '无法获取插件默认状态'}), 500


@system_bp.route('/plugins/<plugin_name>/set_default_state', methods=['POST'])
def set_plugin_default_state_api(plugin_name: str):
    """设置指定插件的默认启用/禁用状态"""
    data = request.get_json()
    if data is None or 'enabled' not in data or not isinstance(data['enabled'], bool):
        return jsonify({
            'success': False, 
            'error': '请求体必须是 JSON 且包含布尔型的 "enabled" 字段'
        }), 400

    enabled = data['enabled']
    logger.info(f"API 请求：设置插件 '{plugin_name}' 默认状态为 {enabled}")

    try:
        plugin_mgr = get_plugin_manager()
        success = plugin_mgr.set_plugin_default_state(plugin_name, enabled)
        if success:
            return jsonify({'success': True, 'message': f"插件 '{plugin_name}' 的默认状态已更新。"})
        else:
            if plugin_mgr.get_plugin(plugin_name):
                return jsonify({'success': False, 'error': '保存插件默认状态失败'}), 500
            else:
                return jsonify({'success': False, 'error': '插件未找到'}), 404
    except Exception as e:
        logger.error(f"设置插件 '{plugin_name}' 默认状态时出错: {e}", exc_info=True)
        return jsonify({'success': False, 'error': '设置插件默认状态时发生内部错误'}), 500
