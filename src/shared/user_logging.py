"""Small user-facing logging vocabulary shared by every Saber process.

The ordinary :mod:`logging` calls in the codebase are diagnostics.  Records
created here are the product log shown in the terminal and desktop window.
Keeping that distinction explicit prevents polling, heartbeat and library
messages from drowning out work the user actually asked Saber to perform.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import logging
import re
from typing import Any, Final

from src.shared.memory_errors import is_memory_allocation_error


USER_LOGGER_NAME: Final = "saber.user"
USER_LOG_MARKER: Final = "saber_user_log"
CATEGORY_FIELD: Final = "saber_log_category"
STREAM_ACTION_FIELD: Final = "saber_stream_action"
STREAM_ID_FIELD: Final = "saber_stream_id"
STREAM_CHUNK_FIELD: Final = "saber_stream_chunk"
STREAM_FRAME_PREFIX: Final = "@@SABER_STREAM@@"

JOB_LABELS: Final = {
    "translation": "漫画翻译",
    "remove_text": "批量去字",
    "detect": "文本检测",
    "style_apply": "样式应用",
    "text_import": "文本导入",
    "container_import": "文件导入",
    "export": "文件导出",
    "web_extract": "网页图片提取",
    "web_import_commit": "网页导入",
    "insight_analysis": "漫画分析",
    "derived_rebuild": "分析结果重建",
    "vector_rebuild": "语义索引重建",
    "insight_export": "分析报告导出",
    "continuation": "漫画续写",
    "plugin_agent": "插件任务",
}

STEP_LABELS: Final = {
    "detect": "文本检测",
    "ocr": "文字识别",
    "color": "颜色分析",
    "auto_terms": "术语提取",
    "translate": "文本翻译",
    "hq_translate": "高质量翻译",
    "proofread": "AI 校对",
    "repair": "文字修复",
    "render": "排版渲染",
    "save": "保存结果",
    "publish_clean": "保存去字图",
    "style_apply_document": "应用文字样式",
    "text_import_apply": "导入文本",
    "container_scan": "扫描导入文件",
    "container_import_page": "导入页面",
    "export_package": "生成下载文件",
    "web_extract_scan": "解析网页目录",
    "web_extract_page": "下载网页图片",
    "web_extract_finalize": "整理网页内容",
    "web_extract_auto_commit": "保存网页内容",
    "web_import_commit_page": "导入网页页面",
    "web_import_commit_finalize": "完成网页导入",
    "insight_analyze_batch": "批量漫画分析",
    "insight_analyze_page": "漫画分析（旧任务）",
    "insight_validate_run": "校验分析结果",
    "insight_publish_run": "发布分析结果",
    "insight_build_overview": "生成作品概览",
    "insight_build_compressed_context": "生成压缩上下文",
    "insight_build_timeline": "生成剧情时间线",
    "insight_build_vectors": "建立语义索引",
    "insight_stage_compressed_context": "整理全书上下文",
    "insight_stage_overview_no_spoiler": "生成无剧透概览",
    "insight_stage_overview_story_summary": "生成剧情总结",
    "insight_stage_timeline": "生成剧情时间线",
    "insight_stage_vectors": "建立语义索引",
    "continuation_generate_script": "生成续写脚本",
    "continuation_generate_page": "生成续写页面方案",
    "continuation_generate_image": "生成续写图片",
    "continuation_generate_character_sheet": "生成角色设定图",
    "continuation_export": "导出续写作品",
    "insight_export_report": "导出分析报告",
    "insight_qa_retrieve": "检索问答资料",
    "insight_qa_answer": "生成问答回复",
    "plugin_agent_execute": "执行插件任务",
    "bubble_ocr": "单气泡文字识别",
    "bubble_color": "单气泡颜色分析",
    "bubble_translate": "单气泡翻译",
    "page_detect": "当前页文本检测",
    "page_repair": "当前页文字修复",
    "live_render": "编辑结果渲染",
    "studio_generate": "生成角色设定",
    "studio_chat": "角色工作室对话",
    "studio_summary": "总结角色对话",
    "studio_agent": "角色卡助手",
}

CATEGORY_LABELS: Final = {
    "task": "任务",
    "step": "步骤",
    "result": "结果",
    "model": "模型",
    "stream": "流式",
    "system": "系统",
    "warning": "警告",
    "error": "错误",
}


@dataclass(frozen=True, slots=True)
class UserLogContext:
    job_id: str | None = None
    operation_id: str | None = None
    page_number: int | None = None
    step_kind: str | None = None
    step_ordinal: int | None = None


_CONTEXT: ContextVar[UserLogContext] = ContextVar(
    "saber_user_log_context",
    default=UserLogContext(),
)
_LOGGER = logging.getLogger(USER_LOGGER_NAME)
_DIAGNOSTIC_LOGGER = logging.getLogger("saber.diagnostics.user_logging")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@contextmanager
def user_log_context(**values: Any) -> Iterator[UserLogContext]:
    """Temporarily add durable task/operation identity to user log records."""

    current = _CONTEXT.get()
    unknown = set(values) - set(UserLogContext.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown user log context: {', '.join(sorted(unknown))}")
    updated = replace(current, **values)
    token = _CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _CONTEXT.reset(token)


def job_label(kind: object) -> str:
    value = str(kind or "").strip()
    return JOB_LABELS.get(value, value or "后台任务")


def step_label(kind: object, *, ordinal: int | None = None) -> str:
    value = str(kind or "").strip()
    if value.startswith("insight_build_layer_"):
        suffix = value.removeprefix("insight_build_layer_")
        if suffix.isascii() and suffix.isdigit():
            return f"汇总第 {int(suffix) + 1} 层分析"
    label = STEP_LABELS.get(value, value or "后台步骤")
    if value == "proofread" and ordinal is not None:
        return f"{label}（第 {ordinal} 轮）"
    return label


def _clean_text(value: object) -> str:
    try:
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    except Exception as error:
        if is_memory_allocation_error(error):
            raise
        text = f"（{type(value).__name__} 无法显示）"
    return _CONTROL_CHARACTERS.sub("", text).strip()


def _clean_detail_text(value: object) -> str:
    """Clean a detail block without destroying intentional indentation."""

    try:
        text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    except Exception as error:
        if is_memory_allocation_error(error):
            raise
        text = f"（{type(value).__name__} 无法显示）"
    return _CONTROL_CHARACTERS.sub("", text).strip("\n")


def inline_log_text(value: object) -> str:
    """Render one result value on a single readable product-log line."""

    text = _clean_text(value)
    return text.replace("\n", " ↵ ") if text else "（空）"


def _prefix(context: UserLogContext) -> str:
    parts: list[str] = []
    if context.job_id:
        parts.append(f"任务 {context.job_id[:8]}")
    elif context.operation_id:
        parts.append(f"操作 {context.operation_id[:8]}")
    if context.page_number is not None:
        parts.append(f"第 {context.page_number} 页")
    if context.step_kind:
        parts.append(step_label(context.step_kind, ordinal=context.step_ordinal))
    return " · ".join(parts)


def _discard_product_log(error: BaseException, diagnostic_message: str) -> None:
    if is_memory_allocation_error(error):
        raise error
    try:
        _DIAGNOSTIC_LOGGER.debug(
            diagnostic_message,
            exc_info=(type(error), error, error.__traceback__),
        )
    except Exception:
        pass


def _publish_product_record(
    category: str,
    body: str,
    *,
    level: int,
    extra: dict[str, object] | None = None,
) -> None:
    effective_level = max(
        level,
        logging.ERROR if category == "error"
        else logging.WARNING if category == "warning"
        else logging.DEBUG,
    )
    fields: dict[str, object] = {
        USER_LOG_MARKER: True,
        CATEGORY_FIELD: category,
    }
    if extra:
        fields.update(extra)
    try:
        _LOGGER.log(effective_level, body, extra=fields)
    except Exception as error:
        _discard_product_log(error, "product log record dropped after output failure")


def user_log(
    category: str,
    message: object,
    *,
    details: Sequence[object] = (),
    level: int = logging.INFO,
) -> None:
    """Publish one readable, secret-redacted product log record."""

    if category not in CATEGORY_LABELS:
        raise ValueError(f"unsupported user log category: {category}")
    try:
        context = _CONTEXT.get()
        clean_message = _clean_text(message)
        prefix = _prefix(context)
        body = f"{prefix}｜{clean_message}" if prefix else clean_message
        clean_details = [_clean_detail_text(value) for value in details]
        clean_details = [value for value in clean_details if value]
        if clean_details:
            rendered_details: list[str] = []
            for value in clean_details:
                lines = value.splitlines() or [""]
                rendered_details.append(f"│ {lines[0]}")
                rendered_details.extend(f"│   {line}" for line in lines[1:])
            body += "\n" + "\n".join(rendered_details)
        _publish_product_record(category, body, level=level)
    except Exception as error:
        _discard_product_log(error, "product log record dropped while formatting")


def _stream_log_record(
    *,
    action: str,
    stream_id: str,
    message: str,
    chunk: str = "",
    level: int = logging.INFO,
    category: str = "stream",
) -> None:
    """Publish one stream event without buffering or changing its chunk."""

    try:
        context_prefix = _prefix(_CONTEXT.get())
        body = f"{context_prefix}｜{message}" if context_prefix else message
    except Exception as error:
        _discard_product_log(error, "product stream record dropped while formatting")
        return
    _publish_product_record(
        category,
        body,
        level=level,
        extra={
            STREAM_ACTION_FIELD: action,
            STREAM_ID_FIELD: stream_id,
            STREAM_CHUNK_FIELD: chunk,
        },
    )


def log_task_started(*, job_id: str, kind: str, execution_mode: str) -> None:
    mode = "并行" if execution_mode == "parallel" else "顺序"
    with user_log_context(job_id=job_id):
        user_log("task", f"{job_label(kind)}开始｜{mode}模式")


def log_task_finished(
    *,
    job_id: str,
    kind: str,
    duration: float | None,
    status: str | None = None,
) -> None:
    status_labels = {
        "completed": "已完成",
        "completed_with_errors": "已完成，部分页面失败",
        "failed": "失败",
        "cancelled": "已取消",
        "paused": "已暂停",
        "interrupted": "已中断",
    }
    state = status_labels.get(str(status), "本轮结束")
    duration_text = f"｜历时 {duration:.2f} 秒" if duration is not None else ""
    with user_log_context(job_id=job_id):
        user_log("task", f"{job_label(kind)}{state}{duration_text}")


def log_task_failed(
    *,
    job_id: str,
    kind: str,
    duration: float,
    error: BaseException,
) -> None:
    message = _clean_text(error) or error.__class__.__name__
    with user_log_context(job_id=job_id):
        user_log(
            "error",
            f"{job_label(kind)}失败｜历时 {duration:.2f} 秒｜{message}",
        )


def log_step_started() -> None:
    user_log("step", "开始")


def log_step_finished(
    *,
    duration: float | None,
    status: str = "completed",
) -> None:
    """Close one step using its persisted outcome, without repeating results."""

    states = {
        "completed": "完成",
        "failed": "已记录失败",
        "skipped": "已跳过",
        "paused": "已暂停",
        "cancelled": "已取消",
    }
    try:
        state = states[status]
    except KeyError as exc:
        raise ValueError(f"unsupported step log status: {status}") from exc
    duration_text = f"｜耗时 {duration:.2f} 秒" if duration is not None else ""
    user_log("step", f"{state}{duration_text}")


def log_step_failed(error: BaseException, *, duration: float) -> None:
    message = _clean_text(error) or error.__class__.__name__
    try:
        _DIAGNOSTIC_LOGGER.debug(
            "step failed: %s",
            _prefix(_CONTEXT.get()) or "unknown step",
            exc_info=(type(error), error, error.__traceback__),
        )
    except Exception:
        pass
    user_log(
        "error",
        f"失败｜耗时 {duration:.2f} 秒｜{message}",
    )


def log_result(summary: object, details: Sequence[object] = ()) -> None:
    user_log("result", summary, details=details)


def log_model_request(
    *,
    provider: str,
    model: str | None,
    stream: bool,
    attempt: int,
    total_attempts: int,
) -> None:
    model_text = model.strip() if isinstance(model, str) and model.strip() else "默认模型"
    mode = "流式" if stream else "非流式"
    retry = (
        f"｜第 {attempt}/{total_attempts} 次请求"
        if total_attempts > 1
        else ""
    )
    user_log("model", f"请求 {provider} / {model_text}｜{mode}{retry}")


def log_model_input(label: str, details: Sequence[object]) -> None:
    clean_label = _clean_text(label)
    user_log("model", f"{clean_label}请求已准备｜{len(details)} 项输入")
    user_log(
        "model",
        f"{clean_label}请求内容",
        details=details,
        level=logging.DEBUG,
    )


def log_model_response(
    label: str,
    content: object,
    *,
    include_content: bool = True,
) -> None:
    text = _clean_text(content)
    user_log(
        "model",
        f"{label}返回 {len(text)} 个字符",
        details=(text.splitlines() or ("（空响应）",)) if include_content else (),
    )


def log_retry(label: str, attempt: int, total_attempts: int, error: object) -> None:
    user_log(
        "warning",
        f"{label}请求或返回结果处理失败，将重试 "
        f"{attempt}/{total_attempts}｜{_clean_text(error)}",
    )


class RetryLogEpisode:
    """Collapse repeated poller failures into one warning and one recovery log."""

    def __init__(self, label: str) -> None:
        self.label = _clean_text(label)
        self._failures = 0

    def record_failure(self, error: BaseException) -> bool:
        self._failures += 1
        if self._failures != 1:
            return False
        user_log(
            "warning",
            f"{self.label}异常，正在自动重试｜"
            f"异常类型 {type(error).__name__}｜详细原因见诊断日志",
        )
        return True

    def report_recovery(self) -> None:
        if self._failures == 0:
            return
        failures = self._failures
        self._failures = 0
        user_log("system", f"{self.label}已恢复｜自动重试 {failures} 次")


class StreamLog:
    """Forward every provider chunk immediately to true streaming sinks."""

    def __init__(self, label: str) -> None:
        self.label = _clean_text(label) or "模型"
        self._stream_id = f"{id(self):x}"
        self._started = False
        self._finished = False

    def __call__(self, chunk: str, _full_text: str) -> None:
        if not chunk:
            return
        try:
            clean_chunk = str(chunk).replace("\r\n", "\n").replace("\r", "\n")
        except Exception as error:
            if is_memory_allocation_error(error):
                raise
            clean_chunk = f"（{type(chunk).__name__} 无法显示）"
        clean_chunk = _CONTROL_CHARACTERS.sub("", clean_chunk)
        if not clean_chunk:
            return
        if not self._started:
            self._started = True
            _stream_log_record(
                action="start",
                stream_id=self._stream_id,
                message=f"{self.label}开始流式返回：",
            )
        _stream_log_record(
            action="chunk",
            stream_id=self._stream_id,
            message=f"{self.label}流式片段｜{clean_chunk}",
            chunk=clean_chunk,
        )

    def finish(self, *, completed: bool = True) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._started:
            return
        _stream_log_record(
            action="end",
            stream_id=self._stream_id,
            message=(
                f"{self.label}流式接收完成"
                if completed
                else f"{self.label}流式接收中断"
            ),
            level=logging.INFO if completed else logging.WARNING,
            category="stream" if completed else "warning",
        )


def json_details(value: object) -> list[str]:
    """Pretty JSON lines for result blocks without ASCII escaping."""

    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception as error:
        if is_memory_allocation_error(error):
            raise
        rendered = _clean_text(value)
    return rendered.splitlines()
