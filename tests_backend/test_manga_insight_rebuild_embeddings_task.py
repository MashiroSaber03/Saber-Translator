import os
import sys
import types
import unittest
from unittest import mock
import importlib.util

from flask import Blueprint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = _OpenAI
    openai_stub.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_stub


class RebuildEmbeddingsTaskApiTests(unittest.TestCase):
    def _load_data_routes_module(self):
        package_name = "isolated_manga_insight_rebuild_pkg"
        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api", "manga_insight")

        package_module = types.ModuleType(package_name)
        package_module.__path__ = [package_dir]
        package_module.manga_insight_bp = Blueprint(
            "isolated_manga_insight_rebuild",
            __name__,
            url_prefix="/api/manga-insight",
        )
        sys.modules[package_name] = package_module

        for mod_name, filename in (
            (f"{package_name}.async_helpers", "async_helpers.py"),
            (f"{package_name}.response_builder", "response_builder.py"),
            (f"{package_name}.data_routes", "data_routes.py"),
        ):
            spec = importlib.util.spec_from_file_location(mod_name, os.path.join(package_dir, filename))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)

        return sys.modules[f"{package_name}.data_routes"]

    def test_rebuild_embeddings_starts_background_task(self) -> None:
        from flask import Flask
        from src.core.manga_insight.task_models import AnalysisTask, TaskType

        app = Flask(__name__)
        data_routes = self._load_data_routes_module()
        captured = {}

        class _DummyTaskManager:
            async def create_task(self, **kwargs):
                captured.update(kwargs)
                return AnalysisTask(
                    book_id=kwargs["book_id"],
                    task_type=kwargs["task_type"],
                )

            async def start_task(self, task_id):
                return types.SimpleNamespace(
                    success=True,
                    task_id=task_id,
                    reason="ok",
                    error_code=None,
                    status_code=200,
                    running_task_id=None,
                )

        config = types.SimpleNamespace(
            embedding=types.SimpleNamespace(provider="custom", model="embed-model", api_key="key")
        )

        with mock.patch("src.core.manga_insight.task_manager.get_task_manager", return_value=_DummyTaskManager()), \
             mock.patch("src.core.manga_insight.config_utils.load_insight_config", return_value=config):
            with app.test_request_context(method="POST", json={}):
                response = data_routes.rebuild_embeddings("book_embed_task")
                payload = response.get_json()

        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "started")
        self.assertEqual(captured["book_id"], "book_embed_task")
        self.assertEqual(captured["task_type"], TaskType.EMBEDDINGS_REBUILD)

    def test_rebuild_embeddings_status_returns_task_and_stats(self) -> None:
        from flask import Flask

        app = Flask(__name__)
        data_routes = self._load_data_routes_module()

        class _DummyTaskManager:
            async def get_task_status(self, _task_id):
                return {
                    "task_id": "task_embed",
                    "book_id": "book_status",
                    "task_type": "embeddings_rebuild",
                    "status": "running",
                    "progress": {"current_phase": "embedding_rebuild"},
                    "result_data": None,
                }

        class _DummyVectorStore:
            def __init__(self, _book_id):
                pass

            def get_stats(self):
                return {"available": True, "pages_count": 12, "events_count": 5}

        with mock.patch("src.core.manga_insight.task_manager.get_task_manager", return_value=_DummyTaskManager()), \
             mock.patch("src.core.manga_insight.vector_store.MangaVectorStore", _DummyVectorStore):
            with app.test_request_context("/?task_id=task_embed"):
                response = data_routes.rebuild_embeddings_status("book_status")
                payload = response.get_json()

        self.assertTrue(payload["success"])
        self.assertEqual(payload["task"]["task_type"], "embeddings_rebuild")
        self.assertEqual(payload["stats"]["pages_count"], 12)


class RebuildEmbeddingsTaskExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_embeddings_rebuild_branch_runs_builder_and_stores_result(self) -> None:
        from src.core.manga_insight.task_executor import TaskExecutor
        from src.core.manga_insight.task_models import AnalysisTask, TaskType

        progress_updates = []

        executor = TaskExecutor(
            check_pause_cancel_func=lambda _task_id: True,
            notify_progress_func=lambda _task_id, progress: progress_updates.append(progress),
        )

        task = AnalysisTask(book_id="book_exec", task_type=TaskType.EMBEDDINGS_REBUILD)

        class _FakeAnalyzer:
            async def build_embeddings(self, progress_callback=None):
                return {"success": True, "pages_count": 3, "events_count": 2}

        warnings = await executor.execute(task, _FakeAnalyzer())

        self.assertEqual(warnings, [])
        self.assertEqual(task.result_data["build_result"]["pages_count"], 3)
        self.assertEqual(task.progress.analyzed_pages, 1)
        self.assertTrue(progress_updates)

    async def test_execute_embeddings_rebuild_branch_updates_progress_from_builder_callback(self) -> None:
        from src.core.manga_insight.task_executor import TaskExecutor
        from src.core.manga_insight.task_models import AnalysisTask, TaskType

        progress_updates = []

        executor = TaskExecutor(
            check_pause_cancel_func=lambda _task_id: True,
            notify_progress_func=lambda _task_id, progress: progress_updates.append(progress.copy()),
        )

        task = AnalysisTask(book_id="book_exec", task_type=TaskType.EMBEDDINGS_REBUILD)

        class _FakeAnalyzer:
            async def build_embeddings(self, progress_callback=None):
                if progress_callback:
                    progress_callback(1, 3, "embedding_batches", "批次 1/3")
                    progress_callback(3, 3, "embedding_batches", "批次 3/3")
                return {"success": True, "pages_count": 3, "events_count": 2}

        await executor.execute(task, _FakeAnalyzer())

        self.assertTrue(any(update.get("current_phase") == "批次 1/3" for update in progress_updates))
        self.assertTrue(any(update.get("phase_progress") == 100.0 for update in progress_updates))


if __name__ == "__main__":
    unittest.main()
