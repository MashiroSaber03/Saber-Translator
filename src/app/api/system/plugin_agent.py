"""
插件自动生成 Agent API

提供插件生成/修改会话、消息交互、执行控制和事件流接口。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from flask import Response, jsonify, request, stream_with_context

from . import system_bp
from src.core.plugin_agent import (
    PluginAgentRuntime,
    get_plugin_agent_settings_payload,
    get_plugin_builder_skill_markdown,
)
from src.plugins.manager import get_plugin_manager
from src.shared.ai_providers import (
    PLUGIN_AGENT_CAPABILITY,
    get_all_provider_manifests,
    normalize_provider_id,
)
from src.shared.openai_options import (
    create_openai_compatible_options,
    merge_openai_compatible_options,
    validate_openai_options_payload,
)
from src.shared.path_helpers import get_app_root

logger = logging.getLogger("SystemAPI.PluginAgent")

_plugin_agent_runtime: Optional[PluginAgentRuntime] = None


def _request_value(data: Dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return default


def get_plugin_agent_runtime() -> PluginAgentRuntime:
    global _plugin_agent_runtime
    if _plugin_agent_runtime is None:
        plugins_root = os.path.join(get_app_root(), "plugins")
        _plugin_agent_runtime = PluginAgentRuntime(
            plugins_root=plugins_root,
            finalize_refresh=lambda _target: get_plugin_manager().refresh_plugins(),
            skill_markdown=get_plugin_builder_skill_markdown(),
        )
    return _plugin_agent_runtime


def _build_provider_options() -> list[dict[str, str]]:
    manifests = get_all_provider_manifests()
    options = []
    for manifest in manifests.values():
        if PLUGIN_AGENT_CAPABILITY in manifest.capabilities:
            options.append({"value": manifest.id, "label": manifest.display_name})
    return sorted(options, key=lambda item: item["label"].lower())


def _parse_agent_config(data: Dict[str, Any]) -> Dict[str, Any]:
    provider = normalize_provider_id(_request_value(data, "provider", default=""))
    api_key = (_request_value(data, "api_key", "apiKey", default="") or "").strip()
    model_name = (_request_value(data, "model_name", "modelName", "model", default="") or "").strip()
    custom_base_url = (_request_value(data, "custom_base_url", "customBaseUrl", "base_url", "baseUrl", default="") or "").strip()

    payload = data.get("openai_options")
    invalid_keys = validate_openai_options_payload(payload)
    if invalid_keys:
        joined = ", ".join(invalid_keys)
        raise ValueError(
            f"openai_options 格式无效: {joined}。"
            "只支持 openai_options.request(force_json_output, temperature, extra_body) "
            "和 openai_options.execution(use_stream, rpm_limit, transport_retries, business_retries)。"
        )

    openai_options = merge_openai_compatible_options(
        payload,
        defaults=create_openai_compatible_options(
            force_json_output=False,
            use_stream=True,
            rpm_limit=7,
            transport_retries=10,
            business_retries=10,
        ),
        business_retries_maximum=10,
    )
    return {
        "provider": provider,
        "api_key": api_key,
        "model_name": model_name,
        "custom_base_url": custom_base_url or None,
        "openai_options": openai_options,
    }


def _session_error_status(error: ValueError) -> int:
    text = str(error)
    if "不存在或已过期" in text:
        return 404
    if "锁定" in text or "执行中" in text:
        return 409
    return 400


@system_bp.route("/plugins/agent/settings", methods=["GET"])
def plugin_agent_settings():
    try:
        payload = get_plugin_agent_settings_payload(
            plugin_records=get_plugin_manager().get_plugin_records(),
        )
        payload["providers"] = _build_provider_options()
        payload["session"] = None
        return jsonify(payload)
    except Exception as exc:
        logger.error("获取插件 Agent 设置失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "无法获取插件 Agent 设置"}), 500


@system_bp.route("/plugins/agent/sessions", methods=["POST"])
def create_plugin_agent_session():
    try:
        data = request.get_json() or {}
        mode = _request_value(data, "mode", default="create")
        plugin_id = _request_value(data, "plugin_id", "pluginId")
        runtime = get_plugin_agent_runtime()
        display_name = None
        if mode == "modify" and plugin_id:
            plugin = get_plugin_manager().get_plugin(plugin_id)
            display_name = getattr(plugin, "display_name", None) if plugin else None
        session = runtime.create_session_with_display_name(
            mode,
            plugin_id=plugin_id,
            display_name=display_name,
        )
        return jsonify({"success": True, "session": session.to_dict()})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.error("创建插件 Agent 会话失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "创建会话失败"}), 500


@system_bp.route("/plugins/agent/sessions/<session_id>", methods=["GET"])
def get_plugin_agent_session(session_id: str):
    runtime = get_plugin_agent_runtime()
    session = runtime.get_session(session_id)
    if not session:
        return jsonify({"success": False, "error": "会话不存在或已过期"}), 404
    return jsonify({"success": True, "session": session.to_dict()})


@system_bp.route("/plugins/agent/sessions/<session_id>", methods=["DELETE"])
def delete_plugin_agent_session(session_id: str):
    runtime = get_plugin_agent_runtime()
    deleted = runtime.delete_session(session_id)
    return jsonify({"success": True, "deleted": deleted})


@system_bp.route("/plugins/agent/sessions/<session_id>/messages", methods=["POST"])
def send_plugin_agent_message(session_id: str):
    try:
        data = request.get_json() or {}
        content = _request_value(data, "content", default="")
        agent_config = _parse_agent_config(data.get("agent_config") or {})
        runtime = get_plugin_agent_runtime()
        session = runtime.send_user_message(session_id, content, agent_config)
        return jsonify({"success": True, "session": session.to_dict()})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), _session_error_status(exc)
    except Exception as exc:
        logger.error("发送插件 Agent 消息失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "消息发送失败"}), 500


@system_bp.route("/plugins/agent/sessions/<session_id>/lock-target", methods=["POST"])
def lock_plugin_agent_target(session_id: str):
    try:
        data = request.get_json() or {}
        proposal = data.get("proposal") if isinstance(data.get("proposal"), dict) else data
        runtime = get_plugin_agent_runtime()
        session = runtime.lock_target(session_id, proposal)
        return jsonify({"success": True, "session": session.to_dict()})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), _session_error_status(exc)
    except Exception as exc:
        logger.error("锁定插件 Agent 目标失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "锁定目标失败"}), 500


@system_bp.route("/plugins/agent/sessions/<session_id>/start", methods=["POST"])
def start_plugin_agent_execution(session_id: str):
    try:
        data = request.get_json() or {}
        agent_config = _parse_agent_config(data.get("agent_config") or {})
        runtime = get_plugin_agent_runtime()
        session = runtime.start_execution(session_id, agent_config)
        return jsonify({"success": True, "session": session.to_dict()})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), _session_error_status(exc)
    except Exception as exc:
        logger.error("启动插件 Agent 执行失败: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "启动执行失败"}), 500


@system_bp.route("/plugins/agent/sessions/<session_id>/cancel", methods=["POST"])
def cancel_plugin_agent_execution(session_id: str):
    runtime = get_plugin_agent_runtime()
    cancelled = runtime.cancel_execution(session_id)
    return jsonify({"success": True, "cancelled": cancelled})


@system_bp.route("/plugins/agent/sessions/<session_id>/events", methods=["GET"])
def stream_plugin_agent_events(session_id: str):
    runtime = get_plugin_agent_runtime()
    session = runtime.get_session(session_id)
    if not session:
        return jsonify({"success": False, "error": "会话不存在或已过期"}), 404

    try:
        after_id = int(request.args.get("after_id", "0") or "0")
    except ValueError:
        after_id = 0

    def generate():
        current_after = after_id
        deadline = time.time() + 15.0
        while True:
            active_session = runtime.get_session(session_id)
            if not active_session:
                yield "event: error\ndata: {\"message\": \"会话不存在或已过期\"}\n\n"
                break

            events = runtime.get_events_since(session_id, current_after)
            for event in events:
                current_after = event.id
                yield f"event: {event.type}\ndata: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"

            if active_session.run_state != "running":
                break
            if time.time() >= deadline:
                break
            time.sleep(0.1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
