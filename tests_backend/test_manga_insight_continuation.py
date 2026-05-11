import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub


class ContinuationStorageTests(unittest.TestCase):
    def test_clear_continuation_data_removes_all_generated_assets(self) -> None:
        from src.core.manga_insight.storage import AnalysisStorage

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                storage = AnalysisStorage("book-1")

                continuation_dir = os.path.join(storage.base_path, "continuation")
                characters_dir = os.path.join(continuation_dir, "characters", "hero")
                os.makedirs(characters_dir, exist_ok=True)

                for relative_path in (
                    "script.json",
                    "pages.json",
                    "config.json",
                    "characters.json",
                    "page001.png",
                    os.path.join("characters", "hero", "hero_form_1.png"),
                ):
                    full_path = os.path.join(continuation_dir, relative_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "wb") as handle:
                        handle.write(b"data")

                self.assertTrue(os.path.exists(continuation_dir))

                result = asyncio.run(storage.clear_continuation_data())

                self.assertTrue(result)
                self.assertFalse(os.path.exists(continuation_dir))

    def test_save_continuation_pages_persists_empty_lists_for_downstream_invalidation(self) -> None:
        from src.core.manga_insight.storage import AnalysisStorage

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                storage = AnalysisStorage("book-1")

                result = asyncio.run(storage.save_continuation_pages([]))
                loaded = asyncio.run(storage.load_continuation_pages())

                self.assertTrue(result)
                self.assertEqual(loaded, {"pages": [], "saved_at": mock.ANY})


class StoryGeneratorReferenceCountTests(unittest.TestCase):
    def test_generate_chapter_script_uses_requested_reference_image_count(self) -> None:
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_template_overview(self, _template_key: str):
                return {"content": "story summary"}

            async def load_timeline(self):
                return {"events": [{"event": "battle"}]}

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = None

        async def _get_recent_page_analyses(_count: int):
            return []

        requested_counts: list[int] = []

        async def _get_recent_manga_images(count: int):
            requested_counts.append(count)
            return [b"image-a", b"image-b"]

        async def _load_images_from_paths(_paths):
            return []

        async def _call_vlm_with_images(_images, _prompt: str):
            return "【第1话 - 测试】\n第1页：内容"

        generator._get_recent_page_analyses = _get_recent_page_analyses
        generator._get_recent_manga_images = _get_recent_manga_images
        generator._load_images_from_paths = _load_images_from_paths
        generator._call_vlm_with_images = _call_vlm_with_images
        generator._build_chapter_script_prompt = lambda **_kwargs: "prompt"
        generator._extract_chapter_title = lambda _text: "测试"
        generator.chat_client = types.SimpleNamespace(generate=mock.AsyncMock(return_value="fallback"))

        script = asyncio.run(
            generator.generate_chapter_script(
                user_direction="继续主线",
                page_count=8,
                reference_image_count=2,
            )
        )

        self.assertEqual(requested_counts, [2])
        self.assertEqual(script.page_count, 8)
        self.assertEqual(script.chapter_title, "测试")

