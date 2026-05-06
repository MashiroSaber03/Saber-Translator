"""翻译 Pipeline 生命周期路由：/api/pipeline/before|after。"""

import logging
from flask import jsonify, request

from src.plugins.http_helpers import (
    run_after_pipeline_hooks,
    run_before_pipeline_hooks,
)
from src.shared.exceptions import PluginException

from . import translate_bp

logger = logging.getLogger("TranslateAPI.Pipeline")


def _normalize_pipeline_payload(data):
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")

    pipeline_id = str(data.get("pipeline_id") or "").strip()
    if not pipeline_id:
        raise ValueError("缺少 pipeline_id")

    mode = str(data.get("mode") or "standard").strip() or "standard"
    scope = str(data.get("scope") or "all").strip() or "all"
    return pipeline_id, mode, scope


@translate_bp.route('/pipeline/before', methods=['POST'])
def pipeline_before():
    try:
        data = request.get_json() or {}
        pipeline_id, mode, scope = _normalize_pipeline_payload(data)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    payload = dict(data)
    payload["pipeline_id"] = pipeline_id
    payload["mode"] = mode
    payload["scope"] = scope

    try:
        updated = run_before_pipeline_hooks(
            payload,
            mode=mode,
            scope=scope,
            pipeline_id=pipeline_id,
        )
    except PluginException as exc:
        # _run_step_phase 会把插件抛的原始 PluginException 包成一条泛化消息，
        # 这里优先取出原始异常的 message 与 details，让插件作者配置的 cancel_message
        # 可以直接展示给用户。
        original = exc.__cause__ if isinstance(exc.__cause__, PluginException) else exc
        message = getattr(original, 'message', None) or str(original)
        plugin_details = dict(getattr(original, 'details', {}) or {})
        plugin_details['cancelled_by_plugin'] = True
        logger.warning(
            "before_pipeline 插件取消任务: pipeline_id=%s, error=%s",
            pipeline_id,
            message,
        )
        return jsonify({
            'success': False,
            'error': message,
            'details': plugin_details,
        }), 409
    except Exception as exc:
        logger.error(
            "before_pipeline 执行异常: pipeline_id=%s, error=%s",
            pipeline_id,
            exc,
            exc_info=True,
        )
        return jsonify({'success': False, 'error': '插件 before_pipeline 执行失败'}), 500

    logger.info(
        "before_pipeline 触发完成: pipeline_id=%s, mode=%s, scope=%s",
        pipeline_id,
        mode,
        scope,
    )
    return jsonify({
        'success': True,
        'pipeline_id': pipeline_id,
        'payload': updated,
    })


@translate_bp.route('/pipeline/after', methods=['POST'])
def pipeline_after():
    try:
        data = request.get_json() or {}
        pipeline_id, mode, scope = _normalize_pipeline_payload(data)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    result_payload = dict(data)
    result_payload["pipeline_id"] = pipeline_id
    result_payload["mode"] = mode
    result_payload["scope"] = scope

    try:
        updated = run_after_pipeline_hooks(
            result_payload,
            mode=mode,
            scope=scope,
            pipeline_id=pipeline_id,
        )
    except Exception as exc:
        # after 阶段不应阻断业务结果，仅记日志
        logger.error(
            "after_pipeline 执行异常（已忽略，不影响结果）: pipeline_id=%s, error=%s",
            pipeline_id,
            exc,
            exc_info=True,
        )
        return jsonify({'success': True, 'pipeline_id': pipeline_id})

    logger.info(
        "after_pipeline 触发完成: pipeline_id=%s, completed=%s, failed=%s",
        pipeline_id,
        result_payload.get("completed"),
        result_payload.get("failed"),
    )
    return jsonify({
        'success': True,
        'pipeline_id': pipeline_id,
        'payload': updated,
    })
