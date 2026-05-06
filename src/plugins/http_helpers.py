from typing import Any, Dict, Optional, Tuple

from .manager import (
    apply_after_pipeline_hooks,
    apply_after_step_hooks,
    apply_before_pipeline_hooks,
    apply_before_step_hooks,
)


def resolve_plugin_request_context(
    data: Dict[str, Any],
    *,
    default_mode: str,
    default_scope: str,
) -> Tuple[str, str]:
    mode = data.get("translation_mode") or data.get("translationMode") or default_mode
    scope = data.get("translation_scope") or data.get("translationScope") or default_scope
    return mode, scope


def prepare_plugin_payload(
    step: str,
    route: str,
    data: Dict[str, Any],
    *,
    default_mode: str,
    default_scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str, str]:
    mode, scope = resolve_plugin_request_context(
        data,
        default_mode=default_mode,
        default_scope=default_scope,
    )
    payload = run_before_step_hooks(
        step,
        route,
        data,
        mode=mode,
        scope=scope,
        metadata=metadata,
    )
    return payload, mode, scope


def run_before_step_hooks(
    step: str,
    route: str,
    data: Dict[str, Any],
    *,
    mode: str,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = apply_before_step_hooks(
        step,
        data,
        mode=mode,
        route=route,
        scope=scope,
        metadata=metadata,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"插件 before_{step} 必须返回对象")
    return payload


def finalize_plugin_result(
    step: str,
    route: str,
    result: Dict[str, Any],
    *,
    mode: str,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    updated = apply_after_step_hooks(
        step,
        result,
        mode=mode,
        route=route,
        scope=scope,
        metadata=metadata,
    )
    if not isinstance(updated, dict):
        raise ValueError(f"插件 after_{step} 必须返回对象")
    return updated


def run_before_pipeline_hooks(
    payload: Dict[str, Any],
    *,
    mode: str,
    scope: str,
    pipeline_id: str,
    route: str = "/api/pipeline/before",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    updated = apply_before_pipeline_hooks(
        payload,
        mode=mode,
        scope=scope,
        pipeline_id=pipeline_id,
        route=route,
        metadata=metadata,
    )
    if not isinstance(updated, dict):
        raise ValueError("插件 before_pipeline 必须返回对象")
    return updated


def run_after_pipeline_hooks(
    result: Dict[str, Any],
    *,
    mode: str,
    scope: str,
    pipeline_id: str,
    route: str = "/api/pipeline/after",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    updated = apply_after_pipeline_hooks(
        result,
        mode=mode,
        scope=scope,
        pipeline_id=pipeline_id,
        route=route,
        metadata=metadata,
    )
    if not isinstance(updated, dict):
        raise ValueError("插件 after_pipeline 必须返回对象")
    return updated
