import time
from typing import Any, Dict

from src.plugins.base import PluginBase
from src.shared.exceptions import PluginException


class PipelineLifecyclePlugin(PluginBase):
    plugin_id = "pipeline_lifecycle_plugin"
    display_name = "Pipeline 生命周期示例"
    plugin_version = "1.0.0"
    plugin_author = "Codex"
    plugin_description = "演示 before_pipeline / after_pipeline 全局钩子的使用，记录任务耗时与可选取消。"
    default_enabled = False
    supported_steps = ("pipeline",)
    supported_modes = ("standard", "hq", "proofread", "remove_text")
    priority = 10
    # before_pipeline 想通过 PluginException 取消任务时，必须使用 fail 策略，
    # 否则 manager 会按 continue 把异常吞掉。
    failure_policy = "fail"

    def __init__(self, plugin_manager, app=None):
        super().__init__(plugin_manager, app=app)
        self._start_times: Dict[str, float] = {}

    def get_config_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            "min_total_images": {
                "type": "number",
                "label": "最少图片数",
                "default": 0,
                "description": "若 total_images 小于此值，则在 before_pipeline 取消任务（0 表示从不取消）。",
            },
            "cancel_message": {
                "type": "text",
                "label": "取消提示",
                "default": "图片数量未达到插件配置的最低要求",
                "description": "被本插件取消时返回给前端的错误信息。",
            },
        }

    def before_pipeline(self, context, payload):
        pipeline_id = str(payload.get("pipeline_id") or "")
        total_images = int(payload.get("total_images") or 0)

        min_images_raw = self.config.get("min_total_images", 0) or 0
        try:
            min_images = int(min_images_raw)
        except (TypeError, ValueError):
            min_images = 0

        if min_images > 0 and total_images < min_images:
            message = str(self.config.get("cancel_message") or "任务被插件取消")
            raise PluginException(
                message,
                details={
                    "pipeline_id": pipeline_id,
                    "total_images": total_images,
                    "required": min_images,
                },
            )

        self._start_times[pipeline_id] = time.monotonic()
        self.logger.info(
            "[before_pipeline] pipeline_id=%s mode=%s scope=%s total_images=%s",
            pipeline_id,
            context.mode,
            context.scope,
            total_images,
        )
        return None

    def after_pipeline(self, context, result):
        pipeline_id = str(result.get("pipeline_id") or "")
        started_at = self._start_times.pop(pipeline_id, None)
        elapsed_ms = int((time.monotonic() - started_at) * 1000) if started_at else None

        self.logger.info(
            "[after_pipeline] pipeline_id=%s mode=%s scope=%s completed=%s failed=%s elapsed_ms=%s",
            pipeline_id,
            context.mode,
            context.scope,
            result.get("completed"),
            result.get("failed"),
            elapsed_ms,
        )
        return None
