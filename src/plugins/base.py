import logging
from typing import Any, Dict

from .hooks import (
    FAILURE_POLICIES,
    PLUGIN_MODES,
    PLUGIN_STEPS,
    PluginContext,
    normalize_plugin_mode,
    normalize_plugin_step,
)


class PluginBase:
    """
    插件 v2 基类。

    插件围绕原子步骤工作：detect / ocr / color / translate /
    ai_translate / inpaint / render，外加一对 pipeline 全局生命周期钩子。
    """

    plugin_id = "unnamed_plugin"
    display_name = "未命名插件"
    plugin_version = "0.1.0"
    plugin_author = "未知作者"
    plugin_description = "这是一个基础插件描述。"
    default_enabled = False
    supported_steps = PLUGIN_STEPS
    supported_modes = PLUGIN_MODES
    priority = 100
    failure_policy = "continue"

    def __init__(self, plugin_manager, app=None):
        self.plugin_manager = plugin_manager
        self.app = app
        self.logger = logging.getLogger(f"Plugin.{self.plugin_id}")
        self._enabled = False
        self.config: Dict[str, Any] = {}

    def setup(self) -> bool:
        self.logger.info(
            "插件 '%s' (%s) 正在设置...",
            self.display_name,
            self.plugin_id,
        )
        return True

    def enable(self) -> None:
        if not self._enabled:
            self._enabled = True
            self.logger.info("插件 '%s' 已启用。", self.display_name)
            self.on_enable()

    def disable(self) -> None:
        if self._enabled:
            self._enabled = False
            self.logger.info("插件 '%s' 已禁用。", self.display_name)
            self.on_disable()

    def is_enabled(self) -> bool:
        return self._enabled

    def get_metadata(self) -> Dict[str, Any]:
        config_schema = self.get_config_schema() or {}
        return {
            "id": self.plugin_id,
            "display_name": self.display_name,
            "version": self.plugin_version,
            "author": self.plugin_author,
            "description": self.plugin_description,
            "default_enabled": bool(self.default_enabled),
            "supported_steps": list(self.supported_steps or ()),
            "supported_modes": list(self.supported_modes or ()),
            "priority": int(self.priority),
            "failure_policy": self.failure_policy,
            "has_config": bool(config_schema),
        }

    def on_enable(self) -> None:
        pass

    def on_disable(self) -> None:
        pass

    def before_detect(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_detect(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_ocr(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_ocr(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_color(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_color(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_translate(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_translate(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_ai_translate(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_ai_translate(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_inpaint(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_inpaint(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_render(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_render(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def before_pipeline(self, context: PluginContext, payload: Dict[str, Any]):
        return None

    def after_pipeline(self, context: PluginContext, result: Dict[str, Any]):
        return None

    def get_config_schema(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def load_config(self, config_data: Dict[str, Any] | None) -> None:
        schema = self.get_config_schema() or {}
        raw_config = config_data or {}
        loaded_config: Dict[str, Any] = {}

        for key, field_spec in schema.items():
            default_value = field_spec.get("default")
            value_type = field_spec.get("type", "text")
            raw_value = raw_config.get(key, default_value)
            loaded_config[key] = self._coerce_config_value(
                key,
                raw_value,
                default_value,
                value_type,
            )

        self.config = loaded_config
        if loaded_config:
            self.logger.info("插件 '%s' 配置已加载: %s", self.display_name, self.config)

    def _coerce_config_value(
        self,
        key: str,
        raw_value: Any,
        default_value: Any,
        value_type: str,
    ) -> Any:
        try:
            if value_type == "number":
                return float(raw_value) if "." in str(raw_value) else int(raw_value)
            if value_type == "boolean":
                if isinstance(raw_value, bool):
                    return raw_value
                return str(raw_value).lower() in {"true", "1", "yes", "on"}
            if value_type == "select":
                return raw_value if raw_value is not None else default_value
            return str(raw_value) if raw_value is not None else default_value
        except (TypeError, ValueError):
            self.logger.warning(
                "配置项 '%s' 的值 '%s' 无效，回退到默认值 '%s'",
                key,
                raw_value,
                default_value,
            )
            return default_value

    def validate_metadata(self) -> None:
        if not self.plugin_id:
            raise ValueError("plugin_id 不能为空")
        self.plugin_id = str(self.plugin_id).strip()
        if not self.plugin_id:
            raise ValueError("plugin_id 不能为空")
        self.display_name = str(self.display_name).strip() or self.plugin_id
        self.plugin_description = str(self.plugin_description).strip()
        self.failure_policy = str(self.failure_policy).strip().lower()

        normalized_steps = tuple(dict.fromkeys(
            normalize_plugin_step(str(step).strip())
            for step in (self.supported_steps or ())
        ))
        if not normalized_steps:
            raise ValueError("supported_steps 不能为空")
        invalid_steps = [step for step in normalized_steps if step not in PLUGIN_STEPS]
        if invalid_steps:
            raise ValueError(f"存在无效的 supported_steps: {invalid_steps}")
        self.supported_steps = normalized_steps

        normalized_modes = tuple(dict.fromkeys(
            normalize_plugin_mode(str(mode).strip())
            for mode in (self.supported_modes or ())
        ))
        if not normalized_modes:
            raise ValueError("supported_modes 不能为空")
        invalid_modes = [mode for mode in normalized_modes if mode not in PLUGIN_MODES]
        if invalid_modes:
            raise ValueError(f"存在无效的 supported_modes: {invalid_modes}")
        self.supported_modes = normalized_modes

        try:
            self.priority = int(self.priority)
        except (TypeError, ValueError) as exc:
            raise ValueError("priority 必须是整数") from exc

        if self.failure_policy not in FAILURE_POLICIES:
            raise ValueError(
                f"failure_policy 必须是 {FAILURE_POLICIES} 之一，当前值: {self.failure_policy}"
            )
