from __future__ import annotations

from io import StringIO
import json
import logging
from pathlib import Path
import sys
from unittest import TestCase

import colorama
from flask import Flask
import pytest

from src.backend_v2.api.app import _install_request_logging
from src.backend_v2.logging_config import (
    ProductConsoleHandler,
    ProductLogFormatter,
    ProductRotatingFileHandler,
    SecretSafeFormatter,
    configure_backend_logging,
    set_backend_console_level,
)
from src.backend_v2.operations.executor import (
    DurableRenderExecutor,
    WorkerOperationRunner,
)
from src.backend_v2.operations.repository import (
    OperationFence,
    OperationFenced,
    RenderFence,
)
from src.backend_v2.storage.schema import JOB_KINDS, OPERATION_KINDS
from src.shared.user_logging import (
    JOB_LABELS,
    RetryLogEpisode,
    STEP_LABELS,
    StreamLog,
    STREAM_FRAME_PREFIX,
    json_details,
    log_model_input,
    log_result,
    log_step_failed,
    step_label,
    user_log,
    user_log_context,
)


def test_secret_safe_formatter_redacts_messages_and_tracebacks() -> None:
    formatter = SecretSafeFormatter("%(levelname)s %(message)s")
    try:
        raise RuntimeError("upstream password=trace-sensitive")
    except RuntimeError:
        exception_info = sys.exc_info()
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "request failed: api_key=sk-sensitive Authorization: Bearer token-value",
        (),
        exception_info,
    )

    rendered = formatter.format(record)

    assert "sk-sensitive" not in rendered
    assert "token-value" not in rendered
    assert "trace-sensitive" not in rendered
    assert rendered.count("[REDACTED]") >= 3


def test_product_log_multiline_output_keeps_role_and_category_on_every_line() -> None:
    formatter = ProductLogFormatter("worker")
    record = logging.LogRecord(
        "saber.user",
        logging.INFO,
        __file__,
        1,
        "OCR 识别完成\n│ 01. 测试文本",
        (),
        None,
    )
    setattr(record, "saber_user_log", True)
    setattr(record, "saber_log_category", "result")

    lines = formatter.format(record).splitlines()

    assert len(lines) == 2
    assert all("[工作进程] [结果]" in line for line in lines)
    assert lines[0][:19] == lines[1][:19]


def test_backend_logging_keeps_product_and_diagnostics_in_separate_files(
    tmp_path: Path,
    capsys,
) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        log_path = configure_backend_logging(
            role="worker",
            data_root=tmp_path,
            console_level="INFO",
        )
        logger = logging.getLogger("saber.test")
        logger.debug("debug detail")
        logger.info("safe info api_key=sk-do-not-log")
        logger.warning("diagnostic warning")
        logger.error("diagnostic error")
        user_log("result", "OCR 识别完成", details=("01. 测试文本",))
        log_model_input(
            "高质量翻译",
            ("仅写入详细产品日志的提示词 api_key=sk-model-secret",),
        )
        for handler in root.handlers:
            handler.flush()

        console = capsys.readouterr().out
        product_text = log_path.read_text(encoding="utf-8")
        diagnostic_text = log_path.with_name(
            "saber-worker-diagnostic.log"
        ).read_text(encoding="utf-8")
        assert "[工作进程]" in console
        assert "[结果]" in console
        assert "OCR 识别完成" in console
        assert "01. 测试文本" in console
        assert colorama.Fore.GREEN in console
        assert colorama.Style.RESET_ALL in console
        assert "sk-do-not-log" not in console
        assert "debug detail" not in console
        assert "safe info" not in console
        assert "diagnostic warning" not in console
        assert "diagnostic error" not in console
        assert "高质量翻译请求已准备｜1 项输入" in console
        assert "仅写入详细产品日志的提示词" not in console
        assert "OCR 识别完成" in product_text
        assert "仅写入详细产品日志的提示词" in product_text
        assert "[调试] [模型]" in product_text
        assert "sk-model-secret" not in product_text
        assert "debug detail" not in product_text
        assert "safe info" not in product_text
        assert "diagnostic warning" not in product_text
        assert "diagnostic error" not in product_text
        assert "OCR 识别完成" not in diagnostic_text
        assert "debug detail" in diagnostic_text
        assert "safe info" in diagnostic_text
        assert "diagnostic warning" in diagnostic_text
        assert "diagnostic error" in diagnostic_text
        assert "sk-do-not-log" not in product_text
        assert "sk-do-not-log" not in diagnostic_text
        assert "\x1b[" not in product_text
        assert "\x1b[" not in diagnostic_text
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(original_level)


def test_debug_console_level_does_not_mix_diagnostics_into_product_output(
    tmp_path: Path,
    capsys,
) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_backend_logging(
            role="api",
            data_root=tmp_path,
            console_level="DEBUG",
        )
        logging.getLogger("saber.test").debug("internal scheduler detail")
        user_log("system", "产品日志仍然可见")
        for handler in root.handlers:
            handler.flush()

        console = capsys.readouterr().out
        assert "产品日志仍然可见" in console
        assert "internal scheduler detail" not in console
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(original_level)


def test_api_request_logging_keeps_normal_successes_at_debug_without_query_values(
    caplog,
) -> None:
    app = Flask(__name__)
    _install_request_logging(app)

    @app.get("/ping")
    def ping() -> str:
        return "pong"

    @app.post("/save")
    def save() -> str:
        return "saved"

    with caplog.at_level(logging.DEBUG, logger="saber.api.http"):
        response = app.test_client().get("/ping?api_key=sk-query-secret")
        post_response = app.test_client().post("/save")

    assert response.status_code == 200
    assert post_response.status_code == 200
    assert response.headers["X-Response-Time"].endswith("s")
    records = [
        record
        for record in caplog.records
        if record.name == "saber.api.http"
    ]
    messages = "\n".join(record.getMessage() for record in records)
    assert "HTTP GET /ping -> 200" in messages
    assert "HTTP POST /save -> 200" in messages
    assert "sk-query-secret" not in messages
    assert {record.levelno for record in records} == {logging.DEBUG}


def test_api_request_logging_omits_successful_health_checks(caplog) -> None:
    app = Flask(__name__)
    _install_request_logging(app)

    @app.get("/api/v2/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    with caplog.at_level(logging.DEBUG, logger="saber.api.http"):
        response = app.test_client().get("/api/v2/health")

    assert response.status_code == 200
    assert response.headers["X-Response-Time"].endswith("s")
    assert not [
        record for record in caplog.records if record.name == "saber.api.http"
    ]


def test_api_request_logging_does_not_turn_fenced_health_polling_into_noise(
    caplog,
) -> None:
    app = Flask(__name__)
    _install_request_logging(app)

    @app.get("/api/v2/health")
    def health() -> tuple[dict[str, str], int]:
        return {"status": "fenced"}, 503

    with caplog.at_level(logging.DEBUG):
        response = app.test_client().get("/api/v2/health")

    assert response.status_code == 503
    assert not [record for record in caplog.records if record.name == "saber.user"]
    assert len(
        [record for record in caplog.records if record.name == "saber.api.http"]
    ) == 1


def test_api_request_logging_keeps_failures_visible(caplog) -> None:
    app = Flask(__name__)
    _install_request_logging(app)

    @app.get("/unavailable")
    def unavailable() -> tuple[str, int]:
        return "unavailable", 503

    with caplog.at_level(logging.DEBUG):
        response = app.test_client().get("/unavailable")

    assert response.status_code == 503
    diagnostic_records = [
        record
        for record in caplog.records
        if record.name == "saber.api.http"
    ]
    assert len(diagnostic_records) == 1
    assert diagnostic_records[0].levelno == logging.DEBUG
    assert "HTTP GET /unavailable -> 503" in diagnostic_records[0].getMessage()
    product_records = [
        record
        for record in caplog.records
        if record.name == "saber.user"
    ]
    assert len(product_records) == 1
    assert product_records[0].levelno == logging.ERROR
    assert "接口请求失败｜GET /unavailable｜状态码 503" in (
        product_records[0].getMessage()
    )


def test_console_level_can_change_without_disabling_detailed_file_logging(
    tmp_path: Path,
) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_backend_logging(
            role="api",
            data_root=tmp_path,
            console_level="INFO",
        )
        owned_handlers = [
            handler
            for handler in root.handlers
            if getattr(handler, "_saber_v2_handler", False)
        ]
        console = next(
            handler
            for handler in owned_handlers
            if not isinstance(handler, logging.FileHandler)
        )
        file_handlers = [
            handler
            for handler in owned_handlers
            if isinstance(handler, logging.FileHandler)
        ]

        set_backend_console_level("WARNING")

        assert console.level == logging.WARNING
        assert {handler.level for handler in file_handlers} == {logging.DEBUG}
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(original_level)


def test_api_request_logging_ignores_normal_stream_disconnects(caplog) -> None:
    app = Flask(__name__)
    _install_request_logging(app)
    teardown = app.teardown_request_funcs[None][-1]

    with app.test_request_context("/api/v2/jobs/events"):
        with caplog.at_level(logging.ERROR, logger="saber.api.http"):
            teardown(GeneratorExit())

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "raised an unhandled exception" not in messages


def test_every_durable_kind_has_a_chinese_product_log_label() -> None:
    assert set(JOB_LABELS) == set(JOB_KINDS)
    assert set(OPERATION_KINDS) <= set(STEP_LABELS)
    assert all(
        any("\u4e00" <= character <= "\u9fff" for character in label)
        for label in (*JOB_LABELS.values(), *STEP_LABELS.values())
    )


def test_product_log_context_correlates_parallel_tasks_and_operations() -> None:
    job_logs = TestCase().assertLogs("saber.user", level="INFO")
    with job_logs as captured, user_log_context(
        job_id="12345678-job",
        page_number=7,
        step_kind="ocr",
    ):
        log_result("识别到 2 个气泡")

    assert "任务 12345678 · 第 7 页 · 文字识别｜识别到 2 个气泡" in captured.output[0]

    operation_logs = TestCase().assertLogs("saber.user", level="INFO")
    with operation_logs as captured, user_log_context(
        operation_id="87654321-operation",
        step_kind="studio_chat",
    ):
        log_result("回复已生成")

    assert "操作 87654321 · 角色工作室对话｜回复已生成" in captured.output[0]


def test_stream_log_publishes_every_provider_chunk_immediately() -> None:
    stream = StreamLog("高质量翻译")
    with TestCase().assertLogs("saber.user", level="INFO") as captured:
        stream("短", "短")
        stream("流", "短流")

    assert len(captured.records) == 3
    assert [record.saber_stream_action for record in captured.records] == [
        "start",
        "chunk",
        "chunk",
    ]
    assert captured.records[1].saber_stream_chunk == "短"
    assert captured.records[2].saber_stream_chunk == "流"


def test_console_handler_appends_stream_chunks_without_newlines() -> None:
    output = StringIO()
    handler = ProductConsoleHandler(output, framed=False)
    handler.setFormatter(ProductLogFormatter("worker"))
    logger = logging.getLogger("saber.user")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream = StreamLog("高质量翻译")
        stream("你", "你")
        assert output.getvalue().endswith("高质量翻译开始流式返回：你")
        stream("好", "你好")
        assert output.getvalue().endswith("高质量翻译开始流式返回：你好")
        stream.finish()
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        handler.close()

    lines = output.getvalue().splitlines()
    assert lines[0].endswith("高质量翻译开始流式返回：你好")
    assert lines[1].endswith("高质量翻译流式接收完成")


def test_console_handler_frames_each_chunk_for_desktop_forwarding() -> None:
    output = StringIO()
    handler = ProductConsoleHandler(output, framed=True)
    handler.setFormatter(ProductLogFormatter("worker"))
    logger = logging.getLogger("saber.user")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream = StreamLog("漫画分析")
        stream("第一段", "第一段")
        stream("第二段", "第一段第二段")
        stream.finish()
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        handler.close()

    frames = [
        json.loads(line.removeprefix(STREAM_FRAME_PREFIX))
        for line in output.getvalue().splitlines()
    ]
    assert [frame["action"] for frame in frames] == [
        "start",
        "chunk",
        "chunk",
        "end",
    ]
    assert [frame["chunk"] for frame in frames[1:3]] == ["第一段", "第二段"]


def test_product_file_persists_stream_chunks_before_completion(tmp_path: Path) -> None:
    path = tmp_path / "product.log"
    handler = ProductRotatingFileHandler(
        path,
        maxBytes=1024 * 1024,
        backupCount=1,
        encoding="utf-8",
    )
    handler.setFormatter(ProductLogFormatter("worker"))
    logger = logging.getLogger("saber.user")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream = StreamLog("AI校对")
        stream("实时内容", "实时内容")
        assert path.read_text(encoding="utf-8").endswith("AI校对开始流式返回：实时内容")
        stream.finish()
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        handler.close()

    assert "AI校对流式接收完成" in path.read_text(encoding="utf-8")


def test_json_log_details_never_break_business_values() -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular

    assert json_details({"path": Path("模型/文件.bin")}) == [
        "{",
        '  "path": "模型\\\\文件.bin"',
        "}",
    ]
    assert json_details(circular) == ["{'self': {...}}"]

    class BrokenString:
        def __str__(self) -> str:
            raise RuntimeError("string rendering failed")

    assert json_details(BrokenString()) == ["（BrokenString 无法显示）"]


def test_step_failure_log_survives_unprintable_exception() -> None:
    class BrokenException(Exception):
        def __str__(self) -> str:
            raise RuntimeError("cannot render exception")

    with TestCase().assertLogs("saber.user", level="ERROR") as captured:
        with user_log_context(step_kind="ocr"):
            log_step_failed(BrokenException(), duration=0.5)

    assert (
        "文字识别｜失败｜耗时 0.50 秒｜（BrokenException 无法显示）"
        in captured.output[0]
    )


def test_product_log_output_failure_never_changes_business_result(monkeypatch) -> None:
    def fail(*_args, **_kwargs) -> None:
        raise OSError("log sink unavailable")

    monkeypatch.setattr("src.shared.user_logging._LOGGER.log", fail)
    user_log("result", "业务已经完成")


def test_product_log_keeps_summary_when_one_detail_cannot_be_stringified() -> None:
    class BrokenDetail:
        def __str__(self) -> str:
            raise ValueError("broken detail")

    with TestCase().assertLogs("saber.user", level="INFO") as captured:
        user_log("result", "识别完成", details=(BrokenDetail(), "正常详情"))

    assert len(captured.records) == 1
    message = captured.records[0].getMessage()
    assert "识别完成" in message
    assert "（BrokenDetail 无法显示）" in message
    assert "正常详情" in message


def test_product_log_never_hides_memory_allocation_failure(monkeypatch) -> None:
    def fail(*_args, **_kwargs) -> None:
        raise MemoryError("out of memory")

    monkeypatch.setattr("src.shared.user_logging._LOGGER.log", fail)
    with pytest.raises(MemoryError, match="out of memory"):
        user_log("result", "业务已经完成")


def test_stream_log_state_advances_even_when_output_sink_fails(monkeypatch) -> None:
    stream = StreamLog("高质量翻译")

    def fail(*_args, **_kwargs) -> None:
        raise OSError("log sink unavailable")

    monkeypatch.setattr("src.shared.user_logging._LOGGER.log", fail)
    stream("已接收", "已接收")
    stream.finish()

    assert stream._started is True
    assert stream._finished is True


def test_dynamic_insight_layer_labels_are_one_based_for_people() -> None:
    assert step_label("insight_build_layer_0") == "汇总第 1 层分析"
    assert step_label("insight_build_layer_8") == "汇总第 9 层分析"


def test_fenced_operation_log_does_not_claim_the_executor_switched() -> None:
    fence = OperationFence(
        operation_id="operation-fenced",
        attempt_id="attempt",
        executor_epoch_id="epoch",
        executor_role="worker",
        owner_user_id="local-user",
    )

    class Repository:
        def claim_next(self, **_kwargs):
            return fence, {"kind": "page_detect"}

    runner = WorkerOperationRunner(
        Repository(),  # type: ignore[arg-type]
        worker_epoch_id="epoch",
        handlers={
            "page_detect": lambda _fence, _operation: (_ for _ in ()).throw(
                OperationFenced("page revision changed")
            )
        },
    )

    with TestCase().assertLogs("saber.user", level="INFO") as captured:
        assert runner.run_one() is True

    rendered = "\n".join(captured.output)
    assert "目标数据或执行状态已变化" in rendered
    assert captured.records[-1].saber_log_category == "system"
    assert "执行器已切换" not in rendered


def test_stored_worker_errors_keep_their_real_traceback_type() -> None:
    error = RuntimeError("后台线程失败")

    with TestCase().assertLogs(
        "saber.diagnostics.user_logging",
        level="DEBUG",
    ) as captured:
        log_step_failed(error, duration=0.25)

    rendered = "\n".join(captured.output)
    assert "RuntimeError: 后台线程失败" in rendered
    assert "NoneType: None" not in rendered


def test_live_render_keeps_success_detail_at_debug_and_failures_visible() -> None:
    class Repository:
        def __init__(self) -> None:
            self.failed: tuple[str, str] | None = None

        def complete(self, _fence, *, publisher) -> bool:
            publisher(object())
            return True

        def fail(self, _fence, *, code: str, message: str) -> None:
            self.failed = (code, message)

    fence = RenderFence(
        render_request_id="render12-request",
        page_id="page34-page",
        rendering_revision=7,
        attempt_id="attempt",
        api_epoch_id="epoch",
        owner_user_id="local-user",
    )
    repository = Repository()
    executor = DurableRenderExecutor(
        repository,  # type: ignore[arg-type]
        api_epoch_id="epoch",
        handler=lambda _fence: lambda _connection: None,
    )

    with TestCase().assertLogs("saber.user", level="DEBUG") as captured:
        executor._execute(fence)

    rendered = "\n".join(captured.output)
    assert "操作 render12 · 编辑结果渲染｜开始｜文档版本 7" in rendered
    assert "操作 render12 · 编辑结果渲染｜完成｜文档版本 7" in rendered

    failing = DurableRenderExecutor(
        repository,  # type: ignore[arg-type]
        api_epoch_id="epoch",
        handler=lambda _fence: (_ for _ in ()).throw(RuntimeError("字体不可用")),
    )
    with TestCase().assertLogs("saber.user", level="ERROR") as captured:
        failing._execute(fence)

    assert "编辑结果渲染｜失败" in captured.output[0]
    assert "字体不可用" in captured.output[0]
    assert repository.failed == ("RENDER_FAILED", "字体不可用")


def test_background_scheduler_faults_log_once_and_report_recovery() -> None:
    labels = ("即时操作调度", "编辑结果调度", "任务状态同步")
    with TestCase().assertLogs("saber.user", level="INFO") as captured:
        for label in labels:
            episode = RetryLogEpisode(label)
            error = RuntimeError("temporary scheduler failure")
            assert episode.record_failure(error) is True
            assert episode.record_failure(error) is False
            episode.report_recovery()

    rendered = "\n".join(captured.output)
    assert rendered.count("即时操作调度异常") == 1
    assert "即时操作调度已恢复｜自动重试 2 次" in rendered
    assert rendered.count("编辑结果调度异常") == 1
    assert "编辑结果调度已恢复｜自动重试 2 次" in rendered
    assert rendered.count("任务状态同步异常") == 1
    assert "任务状态同步已恢复｜自动重试 2 次" in rendered
