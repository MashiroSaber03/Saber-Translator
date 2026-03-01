"""
Manga Insight 最小回归检查脚本

运行方式:
    python tests_backend/manga_insight_regression.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import types
import uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 某些最小环境未安装 PyYAML，这里注入最小桩避免导入失败。
try:
    import yaml  # type: ignore # noqa: F401
except ModuleNotFoundError:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub


def validate_analysis_pages(pages, total_pages: int):
    if total_pages <= 0:
        raise ValueError("书籍没有可分析的图片")
    if not isinstance(pages, list) or not pages:
        raise ValueError("pages 不能为空")
    normalized_pages = []
    for page_num in pages:
        if not isinstance(page_num, int) or isinstance(page_num, bool):
            raise ValueError("页码必须是整数")
        if page_num <= 0:
            raise ValueError("页码必须大于 0")
        if page_num > total_pages:
            raise ValueError("页码越界")
        normalized_pages.append(page_num)
    return sorted(set(normalized_pages))


def validate_reanalyze_pages(pages, total_pages: int):
    if total_pages <= 0:
        raise ValueError("书籍没有可分析的图片")
    if not isinstance(pages, list) or not pages:
        raise ValueError("未指定页面")
    normalized_pages = []
    for page_num in pages:
        if not isinstance(page_num, int) or isinstance(page_num, bool):
            raise ValueError("页码必须是整数")
        if page_num <= 0:
            raise ValueError("页码必须大于 0")
        if page_num > total_pages:
            raise ValueError("页码越界")
        normalized_pages.append(page_num)
    return sorted(set(normalized_pages))


async def check_task_start_conflict() -> None:
    """检查同书冲突启动语义。"""
    try:
        from src.core.manga_insight.task_manager import get_task_manager
        from src.core.manga_insight.task_models import TaskType, TaskStatus
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    manager = get_task_manager()
    manager.tasks.clear()
    manager.running_tasks.clear()
    manager.book_tasks.clear()
    manager.progress_callbacks.clear()
    manager._pause_events.clear()
    manager._cancel_flags.clear()

    task_running = await manager.create_task(book_id="book_conflict", task_type=TaskType.FULL_BOOK)
    task_running.status = TaskStatus.RUNNING

    task_pending = await manager.create_task(book_id="book_conflict", task_type=TaskType.FULL_BOOK)
    start_result = await manager.start_task(task_pending.task_id)

    assert start_result.success is False, "冲突启动应失败"
    assert start_result.status_code == 409, "冲突启动应返回 409"
    assert start_result.error_code == "TASK_START_REJECTED", "冲突启动应返回 TASK_START_REJECTED"
    # 冲突启动不应残留 pending 任务，否则状态接口可能被“最新任务”误导
    assert task_pending.task_id not in manager.tasks, "冲突启动后不应保留 pending 任务"

    latest_task = await manager.get_latest_book_task("book_conflict")
    assert latest_task is not None, "应存在运行中的任务"
    assert latest_task.get("task_id") == task_running.task_id, "最新任务应仍指向真实运行任务"


async def check_storage_clear_for_pages() -> None:
    """检查按页清缓存是否覆盖 page/batch/segment/chapter 和全局缓存。"""
    try:
        from src.core.manga_insight.storage import AnalysisStorage, get_insight_storage_path
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"regression_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)

    try:
        await storage.save_page_analysis(1, {"page_summary": "p1"})
        await storage.save_page_analysis(2, {"page_summary": "p2"})
        await storage.save_batch_analysis(1, 2, {"batch_summary": "b12"})
        await storage.save_batch_analysis(3, 4, {"batch_summary": "b34"})
        await storage.save_segment_summary("seg_1", {"page_range": {"start": 1, "end": 2}, "summary": "s12"})
        await storage.save_segment_summary("seg_2", {"page_range": {"start": 3, "end": 4}, "summary": "s34"})
        await storage.save_chapter_analysis("ch_1", {"title": "ch1", "page_range": {"start": 1, "end": 2}})
        await storage.save_chapter_analysis("ch_2", {"title": "ch2", "page_range": {"start": 3, "end": 4}})
        await storage.save_overview({"summary": "overview"})
        await storage.save_timeline({"groups": []})
        await storage.save_compressed_context({"context": "ctx"})
        await storage.save_template_overview("story_summary", {"content": "template"})

        await storage.clear_cache_for_pages([1], chapter_ids=["ch_1"])

        assert not await storage.load_page_analysis(1), "命中页应被删除"
        assert not await storage.load_batch_analysis(1, 2), "命中批次应被删除"
        assert not await storage.load_segment_summary("seg_1"), "命中小总结应被删除"
        assert not await storage.load_chapter_analysis("ch_1"), "命中章节应被删除"

        assert await storage.load_page_analysis(2), "未命中页应保留"
        assert await storage.load_batch_analysis(3, 4), "未命中批次应保留"
        assert await storage.load_segment_summary("seg_2"), "未命中小总结应保留"
        assert await storage.load_chapter_analysis("ch_2"), "未命中章节应保留"

        assert not await storage.load_timeline(), "全局时间线缓存应被清除"
        assert await storage.load_overview() == {}, "全局概述缓存应被清除"
    finally:
        shutil.rmtree(get_insight_storage_path(book_id), ignore_errors=True)


async def check_page_validation() -> None:
    """检查非法页码输入校验。"""
    try:
        validate_analysis_pages([0], 10)
        raise AssertionError("页码 0 应被拒绝")
    except ValueError:
        pass

    try:
        validate_analysis_pages(["1"], 10)
        raise AssertionError("字符串页码应被拒绝")
    except ValueError:
        pass

    try:
        validate_reanalyze_pages([11], 10)
        raise AssertionError("越界页码应被拒绝")
    except ValueError:
        pass


async def check_manifest_resilience() -> None:
    """检查页面清单构建对单章节坏数据具备容错能力。"""
    try:
        import src.core as core_module
        import src.shared.path_helpers as path_helpers
        from src.core.manga_insight.book_pages import build_book_pages_manifest
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    temp_root = tempfile.mkdtemp(prefix="manifest_regression_")
    book_id = f"book_manifest_{uuid.uuid4().hex[:8]}"
    old_bookshelf = getattr(core_module, "bookshelf_manager", None)
    old_resource_path = path_helpers.resource_path

    try:
        chapters_root = os.path.join(temp_root, "data", "bookshelf", book_id, "chapters")
        os.makedirs(chapters_root, exist_ok=True)

        # 章节 1：坏数据（total_pages 非整数）
        ch1_session = os.path.join(chapters_root, "ch_1", "session")
        os.makedirs(ch1_session, exist_ok=True)
        with open(os.path.join(ch1_session, "session_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"total_pages": "oops"}, f)

        # 章节 2：正常数据（1 页）
        ch2_images = os.path.join(chapters_root, "ch_2", "session", "images", "0")
        os.makedirs(ch2_images, exist_ok=True)
        with open(os.path.join(chapters_root, "ch_2", "session", "session_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"total_pages": 1}, f)
        with open(os.path.join(ch2_images, "original.png"), "wb") as f:
            f.write(b"fake-image")
        with open(os.path.join(ch2_images, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"fileName": "001.png"}, f)

        core_module.bookshelf_manager = types.SimpleNamespace(
            get_book=lambda query_book_id: {
                "id": query_book_id,
                "title": "Regression Book",
                "cover": "",
                "chapters": [{"id": "ch_1"}, {"id": "ch_2"}],
            } if query_book_id == book_id else None
        )
        path_helpers.resource_path = lambda rel: os.path.join(temp_root, rel)

        manifest = build_book_pages_manifest(book_id)

        assert len(manifest.get("chapters", [])) == 2, "章节列表应保留"
        assert manifest.get("total_pages") == 1, "坏章节不应导致整本 total_pages 归零"
        assert len(manifest.get("all_images", [])) == 1, "应保留可用章节图片"
        assert manifest.get("all_images", [])[0].get("chapter_id") == "ch_2", "应来自正常章节"
    finally:
        if old_bookshelf is not None:
            core_module.bookshelf_manager = old_bookshelf
        path_helpers.resource_path = old_resource_path
        shutil.rmtree(temp_root, ignore_errors=True)


async def check_manifest_sort_respects_chapter_order() -> None:
    """检查同名文件跨章节时，清单排序仍保持章节连续。"""
    try:
        import src.core as core_module
        import src.shared.path_helpers as path_helpers
        from src.core.manga_insight.book_pages import build_book_pages_manifest
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    temp_root = tempfile.mkdtemp(prefix="manifest_sort_regression_")
    book_id = f"book_manifest_sort_{uuid.uuid4().hex[:8]}"
    old_bookshelf = getattr(core_module, "bookshelf_manager", None)
    old_resource_path = path_helpers.resource_path

    def _write_page(chapter_id: str, page_idx: int, file_name: str) -> None:
        image_dir = os.path.join(
            temp_root, "data", "bookshelf", book_id, "chapters", chapter_id, "session", "images", str(page_idx)
        )
        os.makedirs(image_dir, exist_ok=True)
        with open(os.path.join(image_dir, "original.png"), "wb") as f:
            f.write(b"fake-image")
        with open(os.path.join(image_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"fileName": file_name}, f)

    try:
        for chapter_id in ("ch_1", "ch_2"):
            session_dir = os.path.join(
                temp_root, "data", "bookshelf", book_id, "chapters", chapter_id, "session"
            )
            os.makedirs(session_dir, exist_ok=True)
            with open(os.path.join(session_dir, "session_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"total_pages": 2}, f)
            _write_page(chapter_id, 0, "image_000.png")
            _write_page(chapter_id, 1, "image_001.png")

        core_module.bookshelf_manager = types.SimpleNamespace(
            get_book=lambda query_book_id: {
                "id": query_book_id,
                "title": "Manifest Sort Book",
                "cover": "",
                "chapters": [{"id": "ch_1"}, {"id": "ch_2"}],
            } if query_book_id == book_id else None
        )
        path_helpers.resource_path = lambda rel: os.path.join(temp_root, rel)

        manifest = build_book_pages_manifest(book_id)
        chapter_sequence = [item.get("chapter_id") for item in manifest.get("all_images", [])]

        assert chapter_sequence == ["ch_1", "ch_1", "ch_2", "ch_2"], \
            f"页序应保持章节连续，实际: {chapter_sequence}"
    finally:
        if old_bookshelf is not None:
            core_module.bookshelf_manager = old_bookshelf
        path_helpers.resource_path = old_resource_path
        shutil.rmtree(temp_root, ignore_errors=True)


async def check_storage_list_pages_supports_large_page_numbers() -> None:
    """检查 list_pages 可正确解析 1000+ 页码文件名。"""
    try:
        from src.core.manga_insight.storage import AnalysisStorage, get_insight_storage_path
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"regression_large_pages_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)

    try:
        await storage.save_page_analysis(1, {"page_summary": "p1"})
        await storage.save_page_analysis(1000, {"page_summary": "p1000"})
        await storage.save_page_analysis(12345, {"page_summary": "p12345"})

        pages = await storage.list_pages()
        assert pages == [1, 1000, 12345], f"list_pages 应返回完整页码，实际: {pages}"
    finally:
        shutil.rmtree(get_insight_storage_path(book_id), ignore_errors=True)


async def check_batch_parse_error_pages_not_persisted() -> None:
    """检查批量结果中的 parse_error 页面不会作为单页分析落盘。"""
    try:
        from src.core.manga_insight.batch_analyzer import BatchAnalyzer
        from src.core.manga_insight.storage import AnalysisStorage, get_insight_storage_path
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubVlm:
        async def analyze_batch(self, _images, start_page, _context=None):
            return {
                "page_range": {"start": start_page, "end": start_page + 1},
                "pages": [
                    {"page_number": start_page, "parse_error": True},
                    {"page_number": start_page + 1, "page_summary": "ok"},
                ],
                "batch_summary": "batch",
                "parse_error": False,
            }

    book_id = f"regression_parse_error_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)
    analyzer = BatchAnalyzer(book_id, storage, _StubVlm())

    try:
        await analyzer.analyze_batch(page_nums=[1, 2], images=[b"img1", b"img2"], force=True, persist=True)
        pages = await storage.list_pages()
        assert pages == [2], f"parse_error 页面不应计入单页缓存，实际: {pages}"
    finally:
        shutil.rmtree(get_insight_storage_path(book_id), ignore_errors=True)


async def check_incremental_no_changes_skips_post_processing() -> None:
    """检查增量 no_changes（无清理）不会触发 embedding/overview 后处理。"""
    try:
        import src.core.manga_insight.incremental_analyzer as incremental_module
        from src.core.manga_insight.task_executor import TaskExecutor
        from src.core.manga_insight.task_models import AnalysisTask, TaskType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubIncrementalAnalyzer:
        def __init__(self, _book_id, _config):
            pass

        async def analyze_new_content(self, on_progress=None, should_stop=None):
            if on_progress:
                on_progress(0, 0)
            return {
                "status": "no_changes",
                "message": "无新增或修改页面，已是最新状态",
                "cache_cleanup": {},
            }

    class _StubAnalyzer:
        pass

    old_incremental_cls = incremental_module.IncrementalAnalyzer
    incremental_module.IncrementalAnalyzer = _StubIncrementalAnalyzer

    executor = TaskExecutor(
        check_pause_cancel_func=lambda _task_id: True,
        notify_progress_func=lambda _task_id, _progress: None
    )
    post_calls = {"count": 0}
    old_post_processing = executor._post_analysis_processing

    async def _tracked_post(*_args, **_kwargs):
        post_calls["count"] += 1
        return []

    task = AnalysisTask(book_id="book_incremental_no_changes", task_type=TaskType.INCREMENTAL)

    try:
        executor._post_analysis_processing = _tracked_post
        warnings = await executor.execute_incremental_analysis(task, _StubAnalyzer())
        assert warnings == [], f"no_changes 不应产生告警，实际: {warnings}"
        assert post_calls["count"] == 0, "no_changes 且无清理时不应触发后处理"
    finally:
        incremental_module.IncrementalAnalyzer = old_incremental_cls
        executor._post_analysis_processing = old_post_processing


class _DummyClosable:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


async def check_qa_close() -> None:
    """检查 QA close 是否关闭所有客户端。"""
    try:
        from src.core.manga_insight.qa import MangaQA
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    qa = MangaQA.__new__(MangaQA)
    qa.chat_client = _DummyClosable()
    qa.embedding_client = _DummyClosable()
    qa.reranker = _DummyClosable()

    await qa.close()

    assert qa.chat_client.closed is True, "chat_client 应被关闭"
    assert qa.embedding_client.closed is True, "embedding_client 应被关闭"
    assert qa.reranker.closed is True, "reranker 应被关闭"


def _count_insight_cache_files(base_path: str) -> int:
    total = 0
    for folder in ("pages", "batches", "segments", "chapters"):
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        for name in os.listdir(folder_path):
            if name.endswith(".json"):
                total += 1
    return total


async def check_reanalyze_sparse_pages_are_contiguous() -> None:
    """检查非连续页重分析会按连续段分批，不出现跨洞范围文件。"""
    try:
        from src.core.manga_insight.task_executor import TaskExecutor
        from src.core.manga_insight.storage import AnalysisStorage, get_insight_storage_path
        from src.core.manga_insight.task_models import AnalysisTask, TaskType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"regression_sparse_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)

    class _FakeAnalyzer:
        def __init__(self):
            self.storage = storage
            self.config = types.SimpleNamespace(
                analysis=types.SimpleNamespace(
                    batch=types.SimpleNamespace(pages_per_batch=3)
                )
            )
            self._book_info = {
                "all_images": [{"path": f"p{i}.png"} for i in range(1, 8)]
            }

        async def get_book_info(self):
            return self._book_info

        async def analyze_batch(self, page_nums, image_infos=None, force=False, **_kwargs):
            start_page = min(page_nums)
            end_page = max(page_nums)
            await storage.save_batch_analysis(start_page, end_page, {
                "pages": [{"page_number": p} for p in page_nums]
            })
            for page_num in page_nums:
                await storage.save_page_analysis(page_num, {"page_number": page_num})
            return {"page_range": {"start": start_page, "end": end_page}}

        async def build_embeddings(self):
            return {"success": True}

        async def generate_overview(self):
            return {"summary": "ok"}

    def _always_continue(_task_id: str) -> bool:
        return True

    def _ignore_progress(_task_id: str, _progress: dict) -> None:
        return None

    task = AnalysisTask(
        book_id=book_id,
        task_type=TaskType.REANALYZE,
        target_pages=[1, 2, 4, 5]
    )

    executor = TaskExecutor(
        check_pause_cancel_func=_always_continue,
        notify_progress_func=_ignore_progress
    )

    try:
        await executor.execute_reanalysis(task, _FakeAnalyzer())
        batches = await storage.list_batches()
        batch_ranges = [(b["start_page"], b["end_page"]) for b in batches]

        assert (1, 4) not in batch_ranges, "不应出现跨洞范围批次 1-4"
        assert batch_ranges == [(1, 2), (4, 5)], f"批次范围应为连续段，实际: {batch_ranges}"
    finally:
        shutil.rmtree(get_insight_storage_path(book_id), ignore_errors=True)


async def check_full_mode_forces_reanalyze() -> None:
    """检查 mode=full 时会强制 force_reanalyze=True。"""
    try:
        from flask import Flask
        from src.app.api.manga_insight import analysis_routes
        from src.core.manga_insight.task_models import AnalysisTask
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    app = Flask(__name__)
    captured = {}

    class _DummyTaskManager:
        async def create_task(self, **kwargs):
            captured.update(kwargs)
            return AnalysisTask(
                book_id=kwargs["book_id"],
                task_type=kwargs["task_type"],
                force_reanalyze=kwargs.get("force_reanalyze", False),
            )

        async def start_task(self, task_id):
            return types.SimpleNamespace(
                success=True,
                task_id=task_id,
                reason="ok",
                error_code=None,
                status_code=200,
                running_task_id=None
            )

    old_manifest = analysis_routes.build_book_pages_manifest
    old_get_task_manager = analysis_routes.get_task_manager
    try:
        analysis_routes.build_book_pages_manifest = lambda _book_id: {
            "total_pages": 12,
            "chapters": []
        }
        analysis_routes.get_task_manager = lambda: _DummyTaskManager()

        with app.test_request_context(json={"mode": "full", "force": False}):
            response = analysis_routes.start_analysis("book_force_full")
            payload = response.get_json()
            assert payload.get("success") is True, "启动 full 模式应成功"
            assert captured.get("force_reanalyze") is True, "full 模式必须强制重跑"
    finally:
        analysis_routes.build_book_pages_manifest = old_manifest
        analysis_routes.get_task_manager = old_get_task_manager


async def check_status_ignores_completed_latest_task() -> None:
    """检查状态接口不会把已完成 latest_task 当作 current_task 返回。"""
    try:
        from flask import Flask
        from src.app.api.manga_insight import analysis_routes
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    app = Flask(__name__)

    class _DummyTaskManager:
        async def get_latest_book_task(self, _book_id):
            return {
                "task_id": "task_done",
                "status": "completed",
                "progress": {"analyzed_pages": 3, "total_pages": 10},
            }

    class _DummyStorage:
        def __init__(self, _book_id):
            pass

        async def list_pages(self):
            return [1, 2, 3]

        async def load_overview(self):
            return {}

    old_manifest = analysis_routes.build_book_pages_manifest
    old_get_task_manager = analysis_routes.get_task_manager
    old_storage_cls = analysis_routes.AnalysisStorage
    try:
        analysis_routes.build_book_pages_manifest = lambda _book_id: {"total_pages": 10, "chapters": []}
        analysis_routes.get_task_manager = lambda: _DummyTaskManager()
        analysis_routes.AnalysisStorage = _DummyStorage

        with app.test_request_context():
            response = analysis_routes.get_analysis_status("book_status")
            payload = response.get_json()

        assert payload.get("success") is True, "状态接口应成功"
        assert payload.get("current_task") is None, "completed latest_task 不应作为 current_task 返回"
        assert payload.get("fully_analyzed") is False, "3/10 页面时不应判定 fully_analyzed=true"
    finally:
        analysis_routes.build_book_pages_manifest = old_manifest
        analysis_routes.get_task_manager = old_get_task_manager
        analysis_routes.AnalysisStorage = old_storage_cls


async def check_status_includes_failed_latest_task() -> None:
    """检查状态接口会返回失败中的 latest_task 供前端显示 failed 状态。"""
    try:
        from flask import Flask
        from src.app.api.manga_insight import analysis_routes
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    app = Flask(__name__)

    class _DummyTaskManager:
        async def get_latest_book_task(self, _book_id):
            return {
                "task_id": "task_failed",
                "status": "failed",
                "error_message": "mock failed",
            }

    class _DummyStorage:
        def __init__(self, _book_id):
            pass

        async def list_pages(self):
            return []

        async def load_overview(self):
            return {}

    old_manifest = analysis_routes.build_book_pages_manifest
    old_get_task_manager = analysis_routes.get_task_manager
    old_storage_cls = analysis_routes.AnalysisStorage
    try:
        analysis_routes.build_book_pages_manifest = lambda _book_id: {"total_pages": 10, "chapters": []}
        analysis_routes.get_task_manager = lambda: _DummyTaskManager()
        analysis_routes.AnalysisStorage = _DummyStorage

        with app.test_request_context():
            response = analysis_routes.get_analysis_status("book_status_failed")
            payload = response.get_json()

        assert payload.get("success") is True, "状态接口应成功"
        assert payload.get("current_task", {}).get("status") == "failed", "failed latest_task 应透出给前端"
    finally:
        analysis_routes.build_book_pages_manifest = old_manifest
        analysis_routes.get_task_manager = old_get_task_manager
        analysis_routes.AnalysisStorage = old_storage_cls


async def check_preview_no_side_effect() -> None:
    """检查 preview 接口调用前后缓存文件数量不变，且返回 persisted=false。"""
    try:
        from flask import Flask
        from src.app.api.manga_insight import analysis_routes
        import src.core.manga_insight.analyzer as analyzer_module
        from src.core.manga_insight.storage import AnalysisStorage, get_insight_storage_path
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    app = Flask(__name__)
    book_id = f"regression_preview_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)
    old_manifest = analysis_routes.build_book_pages_manifest
    old_analyzer_cls = analyzer_module.MangaAnalyzer

    class _PreviewAnalyzer:
        def __init__(self, _book_id, _config):
            self.closed = False

        async def analyze_batch(self, page_nums, force=False, persist=True, **_kwargs):
            assert persist is False, "preview 必须以 persist=False 调用"
            return {
                "page_range": {"start": min(page_nums), "end": max(page_nums)},
                "pages": [{"page_number": p} for p in page_nums]
            }

        async def close(self):
            self.closed = True

    try:
        # 准备既有缓存，调用 preview 前后应保持不变
        await storage.save_page_analysis(1, {"page_summary": "existing"})
        await storage.save_batch_analysis(1, 2, {"batch_summary": "existing"})
        before_count = _count_insight_cache_files(storage.base_path)

        analysis_routes.build_book_pages_manifest = lambda _book_id: {
            "total_pages": 10,
            "chapters": []
        }
        analyzer_module.MangaAnalyzer = _PreviewAnalyzer

        with app.test_request_context(json={"pages": [1, 2, 3]}):
            response = analysis_routes.preview_analysis(book_id)
            payload = response.get_json()
            assert payload.get("success") is True, "preview 调用应成功"
            assert payload.get("persisted") is False, "preview 必须显式返回 persisted=false"

        after_count = _count_insight_cache_files(storage.base_path)
        assert before_count == after_count, "preview 不应新增或删除分析缓存文件"
    finally:
        analysis_routes.build_book_pages_manifest = old_manifest
        analyzer_module.MangaAnalyzer = old_analyzer_cls
        shutil.rmtree(get_insight_storage_path(book_id), ignore_errors=True)


async def check_incremental_snapshot_diff_flow() -> None:
    """检查增量分析基于快照差异：重跑修改页并清理删除章节缓存。"""
    try:
        import src.core.manga_insight.analyzer as analyzer_module
        from src.core.manga_insight.incremental_analyzer import IncrementalAnalyzer
        from src.core.manga_insight.change_detector import ContentChange, ChangeType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubStorage:
        def __init__(self):
            self.clear_calls = []
            self.saved_snapshot = None

        async def load_content_snapshot(self):
            return {"chapters": [{"chapter_id": "ch_keep", "pages": [{"page_num": 1, "hash": "a"}]}]}

        async def clear_cache_for_pages(self, page_nums, chapter_ids=None):
            self.clear_calls.append((sorted(page_nums), sorted(chapter_ids or [])))
            return {"pages_deleted": len(page_nums)}

        async def list_pages(self):
            return [1, 2, 3]

        async def list_batches(self):
            return []

        async def load_batch_analysis(self, _start_page, _end_page):
            return None

        async def save_content_snapshot(self, snapshot):
            self.saved_snapshot = snapshot
            return True

    class _StubDetector:
        async def build_content_snapshot(self):
            return {
                "chapters": [
                    {"chapter_id": "ch_keep", "pages": [{"page_num": 2, "hash": "changed"}]}
                ]
            }

        async def detect_changes(self, _current, _previous):
            return [
                ContentChange(ChangeType.MODIFIED, chapter_id="ch_keep", page_numbers=[2]),
                ContentChange(ChangeType.DELETED, chapter_id="ch_deleted", page_numbers=[3, 4]),
            ]

    class _StubAnalyzer:
        instances = []

        def __init__(self, _book_id, _config):
            self.closed = False
            self.batch_calls = []
            _StubAnalyzer.instances.append(self)

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}, {"path": "p2"}, {"path": "p3"}, {"path": "p4"}]}

        async def analyze_batch(self, page_nums, image_infos=None, force=False, previous_results=None, **_kwargs):
            self.batch_calls.append(list(page_nums))
            return {"page_range": {"start": min(page_nums), "end": max(page_nums)}}

        async def close(self):
            self.closed = True

    old_analyzer_cls = analyzer_module.MangaAnalyzer
    analyzer_module.MangaAnalyzer = _StubAnalyzer
    try:
        config = types.SimpleNamespace(
            analysis=types.SimpleNamespace(
                batch=types.SimpleNamespace(
                    pages_per_batch=5,
                    context_batch_count=1
                )
            ),
            embedding=types.SimpleNamespace(api_key="")
        )
        incremental = IncrementalAnalyzer("book_incremental", config)
        incremental.storage = _StubStorage()
        incremental.change_detector = _StubDetector()

        result = await incremental.analyze_new_content()
        analyzer_instance = _StubAnalyzer.instances[-1]

        assert result.get("status") == "completed", f"增量分析应完成，实际: {result}"
        assert analyzer_instance.batch_calls == [[2]], f"仅应重跑修改页，实际: {analyzer_instance.batch_calls}"
        assert incremental.storage.clear_calls == [
            ([3, 4], ["ch_deleted"]),
            ([2], [])
        ], f"应先清理删除内容，再清理待重跑页面，实际: {incremental.storage.clear_calls}"
        assert incremental.storage.saved_snapshot is not None, "完成后应保存新快照"
        assert analyzer_instance.closed is True, "增量分析结束后应关闭 analyzer 客户端"
    finally:
        analyzer_module.MangaAnalyzer = old_analyzer_cls


async def check_incremental_no_snapshot_when_failed_pages() -> None:
    """检查增量分析存在失败页时不会保存快照。"""
    try:
        import src.core.manga_insight.analyzer as analyzer_module
        from src.core.manga_insight.incremental_analyzer import IncrementalAnalyzer
        from src.core.manga_insight.change_detector import ContentChange, ChangeType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubStorage:
        def __init__(self):
            self.saved_snapshot = None

        async def load_content_snapshot(self):
            return {"chapters": []}

        async def clear_cache_for_pages(self, page_nums, chapter_ids=None):
            return {}

        async def list_pages(self):
            return []

        async def list_batches(self):
            return []

        async def load_batch_analysis(self, _start_page, _end_page):
            return None

        async def save_content_snapshot(self, snapshot):
            self.saved_snapshot = snapshot
            return True

    class _StubDetector:
        async def build_content_snapshot(self):
            return {"chapters": [{"chapter_id": "ch_1", "pages": [{"page_num": 2, "hash": "h2"}]}]}

        async def detect_changes(self, _current, _previous):
            return [ContentChange(ChangeType.MODIFIED, chapter_id="ch_1", page_numbers=[2])]

    class _FailingAnalyzer:
        instances = []

        def __init__(self, _book_id, _config):
            self.closed = False
            _FailingAnalyzer.instances.append(self)

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}, {"path": "p2"}]}

        async def analyze_batch(self, page_nums, image_infos=None, force=False, previous_results=None, **_kwargs):
            raise RuntimeError("mock analyze failure")

        async def close(self):
            self.closed = True

    old_analyzer_cls = analyzer_module.MangaAnalyzer
    analyzer_module.MangaAnalyzer = _FailingAnalyzer
    try:
        config = types.SimpleNamespace(
            analysis=types.SimpleNamespace(
                batch=types.SimpleNamespace(
                    pages_per_batch=5,
                    context_batch_count=1
                )
            ),
            embedding=types.SimpleNamespace(api_key="")
        )
        incremental = IncrementalAnalyzer("book_incremental_fail", config)
        incremental.storage = _StubStorage()
        incremental.change_detector = _StubDetector()

        result = await incremental.analyze_new_content()
        analyzer_instance = _FailingAnalyzer.instances[-1]

        assert result.get("status") == "completed", f"应返回 completed，实际: {result}"
        assert result.get("pages_failed") == 1, f"失败页应为 1，实际: {result}"
        assert result.get("snapshot_saved") is False, "失败页存在时 snapshot_saved 应为 False"
        assert incremental.storage.saved_snapshot is None, "失败页存在时不应保存快照"
        assert analyzer_instance.closed is True, "异常后 analyzer 仍应被关闭"
    finally:
        analyzer_module.MangaAnalyzer = old_analyzer_cls


async def check_incremental_parse_error_no_snapshot() -> None:
    """检查增量分析遇到 parse_error 时视为失败且不保存快照。"""
    try:
        import src.core.manga_insight.analyzer as analyzer_module
        from src.core.manga_insight.incremental_analyzer import IncrementalAnalyzer
        from src.core.manga_insight.change_detector import ContentChange, ChangeType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubStorage:
        def __init__(self):
            self.saved_snapshot = None

        async def load_content_snapshot(self):
            return {"chapters": []}

        async def clear_cache_for_pages(self, page_nums, chapter_ids=None):
            return {}

        async def list_pages(self):
            return []

        async def list_batches(self):
            return []

        async def load_batch_analysis(self, _start_page, _end_page):
            return None

        async def save_content_snapshot(self, snapshot):
            self.saved_snapshot = snapshot
            return True

    class _StubDetector:
        async def build_content_snapshot(self):
            return {"chapters": [{"chapter_id": "ch_1", "pages": [{"page_num": 2, "hash": "h2"}]}]}

        async def detect_changes(self, _current, _previous):
            return [ContentChange(ChangeType.MODIFIED, chapter_id="ch_1", page_numbers=[2])]

    class _ParseErrorAnalyzer:
        instances = []

        def __init__(self, _book_id, _config):
            self.closed = False
            _ParseErrorAnalyzer.instances.append(self)

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}, {"path": "p2"}]}

        async def analyze_batch(self, page_nums, image_infos=None, force=False, previous_results=None, **_kwargs):
            return {
                "page_range": {"start": min(page_nums), "end": max(page_nums)},
                "pages": [{"page_number": p, "parse_error": True} for p in page_nums],
                "parse_error": True,
            }

        async def close(self):
            self.closed = True

    old_analyzer_cls = analyzer_module.MangaAnalyzer
    analyzer_module.MangaAnalyzer = _ParseErrorAnalyzer
    try:
        config = types.SimpleNamespace(
            analysis=types.SimpleNamespace(
                batch=types.SimpleNamespace(
                    pages_per_batch=5,
                    context_batch_count=1
                )
            ),
            embedding=types.SimpleNamespace(api_key="")
        )
        incremental = IncrementalAnalyzer("book_incremental_parse_error", config)
        incremental.storage = _StubStorage()
        incremental.change_detector = _StubDetector()

        result = await incremental.analyze_new_content()
        analyzer_instance = _ParseErrorAnalyzer.instances[-1]

        assert result.get("status") == "completed", f"应返回 completed，实际: {result}"
        assert result.get("pages_failed") == 1, f"parse_error 页应计入失败，实际: {result}"
        assert result.get("pages_analyzed") == 0, f"parse_error 页不应计入成功，实际: {result}"
        assert result.get("snapshot_saved") is False, "parse_error 失败存在时 snapshot_saved 应为 False"
        assert incremental.storage.saved_snapshot is None, "parse_error 失败存在时不应保存快照"
        assert analyzer_instance.closed is True, "结束后 analyzer 应关闭"
    finally:
        analyzer_module.MangaAnalyzer = old_analyzer_cls


async def check_incremental_cancelled_before_cleanup() -> None:
    """检查增量分析在取消时不会先清理缓存。"""
    try:
        import src.core.manga_insight.analyzer as analyzer_module
        from src.core.manga_insight.incremental_analyzer import IncrementalAnalyzer
        from src.core.manga_insight.change_detector import ContentChange, ChangeType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubStorage:
        def __init__(self):
            self.clear_called = False

        async def load_content_snapshot(self):
            return {"chapters": []}

        async def clear_cache_for_pages(self, page_nums, chapter_ids=None):
            self.clear_called = True
            return {}

        async def list_pages(self):
            return []

        async def list_batches(self):
            return []

        async def load_batch_analysis(self, _start_page, _end_page):
            return None

        async def save_content_snapshot(self, snapshot):
            return True

    class _StubDetector:
        async def build_content_snapshot(self):
            return {"chapters": [{"chapter_id": "ch_1", "pages": [{"page_num": 1, "hash": "h1"}]}]}

        async def detect_changes(self, _current, _previous):
            return [ContentChange(ChangeType.MODIFIED, chapter_id="ch_1", page_numbers=[1])]

    class _StubAnalyzer:
        instances = []

        def __init__(self, _book_id, _config):
            self.closed = False
            _StubAnalyzer.instances.append(self)

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}]}

        async def close(self):
            self.closed = True

    old_analyzer_cls = analyzer_module.MangaAnalyzer
    analyzer_module.MangaAnalyzer = _StubAnalyzer
    try:
        config = types.SimpleNamespace(
            analysis=types.SimpleNamespace(
                batch=types.SimpleNamespace(
                    pages_per_batch=5,
                    context_batch_count=1
                )
            ),
            embedding=types.SimpleNamespace(api_key="")
        )
        incremental = IncrementalAnalyzer("book_incremental_cancel", config)
        incremental.storage = _StubStorage()
        incremental.change_detector = _StubDetector()

        result = await incremental.analyze_new_content(should_stop=lambda: True)
        analyzer_instance = _StubAnalyzer.instances[-1]

        assert result.get("status") == "cancelled", f"应返回 cancelled，实际: {result}"
        assert incremental.storage.clear_called is False, "取消前不应执行缓存清理"
        assert analyzer_instance.closed is True, "取消后 analyzer 仍应被关闭"
    finally:
        analyzer_module.MangaAnalyzer = old_analyzer_cls


async def check_task_manager_closes_analyzer() -> None:
    """检查任务执行结束后会调用 analyzer.close()。"""
    try:
        import src.core.manga_insight.task_manager as task_manager_module
        import src.core.manga_insight.analyzer as analyzer_module
        import src.core.manga_insight.task_executor as task_executor_module
        from src.core.manga_insight.task_models import AnalysisTask, TaskType, TaskStatus
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _StubAnalyzer:
        instances = []

        def __init__(self, _book_id, _config):
            self.closed = False
            _StubAnalyzer.instances.append(self)

        async def close(self):
            self.closed = True

    class _StubExecutor:
        def __init__(self, check_pause_cancel_func, notify_progress_func):
            self._check = check_pause_cancel_func
            self._notify = notify_progress_func

        async def execute(self, task, analyzer):
            return []

        async def build_timeline_on_complete(self, book_id):
            return None

    old_load_config = task_manager_module.load_insight_config
    old_analyzer_cls = analyzer_module.MangaAnalyzer
    old_executor_cls = task_executor_module.TaskExecutor

    manager = task_manager_module.get_task_manager()
    manager.tasks.clear()
    manager.running_tasks.clear()
    manager.book_tasks.clear()
    manager.progress_callbacks.clear()
    manager._pause_events.clear()
    manager._cancel_flags.clear()

    task = AnalysisTask(book_id="book_close", task_type=TaskType.FULL_BOOK)
    task.status = TaskStatus.RUNNING
    manager.tasks[task.task_id] = task
    manager.running_tasks[task.task_id] = object()
    manager._cancel_flags[task.task_id] = False

    try:
        task_manager_module.load_insight_config = lambda: types.SimpleNamespace()
        analyzer_module.MangaAnalyzer = _StubAnalyzer
        task_executor_module.TaskExecutor = _StubExecutor

        await manager._execute_task(task)
        analyzer_instance = _StubAnalyzer.instances[-1]

        assert analyzer_instance.closed is True, "任务结束后 analyzer.close() 必须被调用"
        assert task.status == TaskStatus.COMPLETED, f"任务应完成，实际状态: {task.status}"
    finally:
        task_manager_module.load_insight_config = old_load_config
        analyzer_module.MangaAnalyzer = old_analyzer_cls
        task_executor_module.TaskExecutor = old_executor_cls


async def check_reanalysis_parse_error_marks_failed_pages() -> None:
    """检查重分析遇到 parse_error 会计入失败页且不增加成功进度。"""
    try:
        from src.core.manga_insight.task_executor import TaskExecutor
        from src.core.manga_insight.task_models import AnalysisTask, TaskType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _FakeStorage:
        async def clear_cache_for_pages(self, page_nums):
            return {}

    class _FakeAnalyzer:
        def __init__(self):
            self.storage = _FakeStorage()
            self.config = types.SimpleNamespace(
                analysis=types.SimpleNamespace(
                    batch=types.SimpleNamespace(pages_per_batch=5)
                )
            )

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}, {"path": "p2"}]}

        async def analyze_batch(self, page_nums, image_infos=None, force=False, **_kwargs):
            return {
                "page_range": {"start": min(page_nums), "end": max(page_nums)},
                "pages": [{"page_number": p, "parse_error": True} for p in page_nums],
                "parse_error": True,
            }

        async def build_embeddings(self):
            return {"success": True}

        async def generate_overview(self):
            return {"summary": "ok"}

    task = AnalysisTask(
        book_id="book_reanalyze_parse_error",
        task_type=TaskType.REANALYZE,
        target_pages=[1, 2]
    )

    executor = TaskExecutor(
        check_pause_cancel_func=lambda _task_id: True,
        notify_progress_func=lambda _task_id, _progress: None
    )
    warnings = await executor.execute_reanalysis(task, _FakeAnalyzer())

    assert task.progress.analyzed_pages == 0, "parse_error 批次不应计入成功进度"
    assert sorted(task.failed_pages) == [1, 2], f"失败页应包含目标页，实际: {task.failed_pages}"
    assert any("重分析失败" in warning for warning in warnings), f"应包含重分析失败告警，实际: {warnings}"


async def check_full_analysis_refreshes_snapshot() -> None:
    """检查 full 分析完成后会重建并保存 content snapshot。"""
    try:
        import src.core.manga_insight.task_executor as task_executor_module
        import src.core.manga_insight.change_detector as change_detector_module
        from src.core.manga_insight.task_models import AnalysisTask, TaskType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _VlmStub:
        @staticmethod
        def is_configured():
            return True

    class _StorageStub:
        def __init__(self):
            self.saved_snapshot = None

        async def clear_all(self):
            return True

        async def save_content_snapshot(self, snapshot):
            self.saved_snapshot = snapshot
            return True

    class _AnalyzerStub:
        def __init__(self):
            self.vlm = _VlmStub()
            self.storage = _StorageStub()
            self.config = types.SimpleNamespace(
                analysis=types.SimpleNamespace(
                    batch=types.SimpleNamespace(pages_per_batch=5)
                )
            )

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}], "chapters": []}

    class _DetectorStub:
        def __init__(self, _book_id):
            pass

        async def build_content_snapshot(self):
            return {"chapters": [{"chapter_id": "ch_1", "pages": [{"page_num": 1, "hash": "h1"}]}]}

    old_load_config = task_executor_module.load_insight_config
    old_detector_cls = change_detector_module.ContentChangeDetector
    executor = task_executor_module.TaskExecutor(
        check_pause_cancel_func=lambda _task_id: True,
        notify_progress_func=lambda _task_id, _progress: None
    )

    async def _noop_full_batch(*_args, **_kwargs):
        return [], False

    task = AnalysisTask(book_id="book_full_snapshot", task_type=TaskType.FULL_BOOK, force_reanalyze=True)
    analyzer = _AnalyzerStub()
    old_execute_full_batch = executor._execute_full_book_batch_analysis

    try:
        task_executor_module.load_insight_config = lambda: types.SimpleNamespace(
            analysis=types.SimpleNamespace(batch=types.SimpleNamespace(pages_per_batch=5)),
            vlm=types.SimpleNamespace(force_json=True, use_stream=False),
        )
        change_detector_module.ContentChangeDetector = _DetectorStub
        executor._execute_full_book_batch_analysis = _noop_full_batch

        warnings = await executor.execute_full_book_analysis(task, analyzer)

        assert warnings == [], f"不应产生警告，实际: {warnings}"
        assert analyzer.storage.saved_snapshot is not None, "full 分析后应保存 content snapshot"
    finally:
        task_executor_module.load_insight_config = old_load_config
        change_detector_module.ContentChangeDetector = old_detector_cls
        executor._execute_full_book_batch_analysis = old_execute_full_batch


async def check_full_analysis_parse_error_batch_skips_snapshot() -> None:
    """检查 full 分析中出现 parse_error 批次时不会刷新快照。"""
    try:
        import src.core.manga_insight.task_executor as task_executor_module
        from src.core.manga_insight.task_models import AnalysisTask, TaskType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _VlmStub:
        @staticmethod
        def is_configured():
            return True

    class _StorageStub:
        def __init__(self):
            self.saved_snapshot = None

        async def clear_all(self):
            return True

        async def save_content_snapshot(self, snapshot):
            self.saved_snapshot = snapshot
            return True

    class _AnalyzerStub:
        def __init__(self):
            self.vlm = _VlmStub()
            self.storage = _StorageStub()

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}], "chapters": []}

        async def build_embeddings(self):
            return {"success": True}

        async def generate_overview(self):
            return {"summary": "ok"}

    old_load_config = task_executor_module.load_insight_config
    executor = task_executor_module.TaskExecutor(
        check_pause_cancel_func=lambda _task_id: True,
        notify_progress_func=lambda _task_id, _progress: None
    )
    old_execute_batch_layer = executor._execute_batch_layer
    old_post_processing = executor._post_analysis_processing

    async def _parse_error_batch_layer(*_args, **_kwargs):
        return [{
            "page_range": {"start": 1, "end": 1},
            "pages": [{"page_number": 1, "parse_error": True}],
            "parse_error": True,
        }]

    async def _noop_post(*_args, **_kwargs):
        return []

    task = AnalysisTask(book_id="book_full_parse_error", task_type=TaskType.FULL_BOOK, force_reanalyze=True)
    analyzer = _AnalyzerStub()

    try:
        task_executor_module.load_insight_config = lambda: types.SimpleNamespace(
            analysis=types.SimpleNamespace(
                batch=types.SimpleNamespace(
                    pages_per_batch=5,
                    context_batch_count=1,
                    get_layers=lambda: [{"name": "批量分析", "units_per_group": 5, "align_to_chapter": False}]
                )
            ),
            vlm=types.SimpleNamespace(force_json=True, use_stream=False),
        )
        executor._execute_batch_layer = _parse_error_batch_layer
        executor._post_analysis_processing = _noop_post

        warnings = await executor.execute_full_book_analysis(task, analyzer)

        assert analyzer.storage.saved_snapshot is None, "parse_error 批次存在时不应保存 content snapshot"
        assert any("失败批次" in warning for warning in warnings), f"应包含失败批次告警，实际: {warnings}"
    finally:
        task_executor_module.load_insight_config = old_load_config
        executor._execute_batch_layer = old_execute_batch_layer
        executor._post_analysis_processing = old_post_processing


async def check_full_analysis_skips_snapshot_when_batch_failed() -> None:
    """检查 full 分析存在失败批次时不会刷新快照。"""
    try:
        import src.core.manga_insight.task_executor as task_executor_module
        from src.core.manga_insight.task_models import AnalysisTask, TaskType
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    class _VlmStub:
        @staticmethod
        def is_configured():
            return True

    class _StorageStub:
        def __init__(self):
            self.saved_snapshot = None

        async def clear_all(self):
            return True

        async def save_content_snapshot(self, snapshot):
            self.saved_snapshot = snapshot
            return True

    class _AnalyzerStub:
        def __init__(self):
            self.vlm = _VlmStub()
            self.storage = _StorageStub()
            self.config = types.SimpleNamespace(
                analysis=types.SimpleNamespace(
                    batch=types.SimpleNamespace(pages_per_batch=5)
                )
            )

        async def get_book_info(self):
            return {"all_images": [{"path": "p1"}], "chapters": []}

    old_load_config = task_executor_module.load_insight_config
    executor = task_executor_module.TaskExecutor(
        check_pause_cancel_func=lambda _task_id: True,
        notify_progress_func=lambda _task_id, _progress: None
    )

    async def _failed_batch_result(*_args, **_kwargs):
        return ["全书批量分析存在失败批次: 1/1"], True

    task = AnalysisTask(book_id="book_full_snapshot_fail", task_type=TaskType.FULL_BOOK, force_reanalyze=True)
    analyzer = _AnalyzerStub()
    old_execute_full_batch = executor._execute_full_book_batch_analysis

    try:
        task_executor_module.load_insight_config = lambda: types.SimpleNamespace(
            analysis=types.SimpleNamespace(batch=types.SimpleNamespace(pages_per_batch=5)),
            vlm=types.SimpleNamespace(force_json=True, use_stream=False),
        )
        executor._execute_full_book_batch_analysis = _failed_batch_result

        warnings = await executor.execute_full_book_analysis(task, analyzer)

        assert any("失败批次" in w for w in warnings), f"应包含失败批次警告，实际: {warnings}"
        assert analyzer.storage.saved_snapshot is None, "存在失败批次时不应保存 content snapshot"
    finally:
        task_executor_module.load_insight_config = old_load_config
        executor._execute_full_book_batch_analysis = old_execute_full_batch


async def main() -> int:
    checks = [
        ("task_start_conflict", check_task_start_conflict),
        ("storage_clear_for_pages", check_storage_clear_for_pages),
        ("page_validation", check_page_validation),
        ("manifest_resilience", check_manifest_resilience),
        ("manifest_sort_respects_chapter_order", check_manifest_sort_respects_chapter_order),
        ("storage_list_pages_supports_large_page_numbers", check_storage_list_pages_supports_large_page_numbers),
        ("batch_parse_error_pages_not_persisted", check_batch_parse_error_pages_not_persisted),
        ("incremental_no_changes_skips_post_processing", check_incremental_no_changes_skips_post_processing),
        ("qa_close", check_qa_close),
        ("reanalyze_sparse_pages_are_contiguous", check_reanalyze_sparse_pages_are_contiguous),
        ("full_mode_forces_reanalyze", check_full_mode_forces_reanalyze),
        ("status_ignores_completed_latest_task", check_status_ignores_completed_latest_task),
        ("status_includes_failed_latest_task", check_status_includes_failed_latest_task),
        ("preview_no_side_effect", check_preview_no_side_effect),
        ("incremental_snapshot_diff_flow", check_incremental_snapshot_diff_flow),
        ("incremental_no_snapshot_when_failed_pages", check_incremental_no_snapshot_when_failed_pages),
        ("incremental_parse_error_no_snapshot", check_incremental_parse_error_no_snapshot),
        ("incremental_cancelled_before_cleanup", check_incremental_cancelled_before_cleanup),
        ("task_manager_closes_analyzer", check_task_manager_closes_analyzer),
        ("reanalysis_parse_error_marks_failed_pages", check_reanalysis_parse_error_marks_failed_pages),
        ("full_analysis_refreshes_snapshot", check_full_analysis_refreshes_snapshot),
        ("full_analysis_parse_error_batch_skips_snapshot", check_full_analysis_parse_error_batch_skips_snapshot),
        ("full_analysis_skips_snapshot_when_batch_failed", check_full_analysis_skips_snapshot_when_batch_failed),
    ]

    failed = 0
    skipped = 0
    for name, func in checks:
        try:
            await func()
            print(f"[PASS] {name}")
        except RuntimeError as exc:
            if str(exc).startswith("MISSING_DEPENDENCY:"):
                skipped += 1
                print(f"[SKIP] {name}: 缺少依赖 {str(exc).split(':', 1)[1]}")
            else:
                failed += 1
                print(f"[FAIL] {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {exc}")

    if failed > 0:
        print(f"\n回归检查失败: {failed}/{len(checks)} (跳过 {skipped})")
        return 1

    print(f"\n回归检查通过: {len(checks) - skipped}/{len(checks)} (跳过 {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
