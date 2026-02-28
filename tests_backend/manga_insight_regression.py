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


async def main() -> int:
    checks = [
        ("task_start_conflict", check_task_start_conflict),
        ("storage_clear_for_pages", check_storage_clear_for_pages),
        ("page_validation", check_page_validation),
        ("manifest_resilience", check_manifest_resilience),
        ("qa_close", check_qa_close),
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
