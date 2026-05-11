import json
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


class StoryGeneratorContinuationBehaviorTests(unittest.TestCase):
    def test_prepare_continuation_data_auto_generates_story_summary_when_missing(self) -> None:
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            def __init__(self) -> None:
                self.saved_summary = None

            async def load_template_overview(self, template_key: str):
                if template_key == "story_summary":
                    return None
                return None

            async def load_timeline(self):
                return {"events": [{"event": "开端"}]}

            async def save_template_overview(self, template_key: str, data):
                self.saved_summary = (template_key, data)
                return True

        class _SummaryGeneratorStub:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            async def generate_with_template(self, template_key: str = "story_summary", skip_cache: bool = False):
                return {
                    "template_key": template_key,
                    "template_name": "故事概要",
                    "template_icon": "📖",
                    "content": "自动补齐的故事概要",
                    "source": "llm",
                    "generated_at": "2026-05-11T00:00:00",
                }

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.config = types.SimpleNamespace(prompts=types.SimpleNamespace())
        generator.char_manager = types.SimpleNamespace(
            format_characters_for_prompt=lambda: "（暂无角色信息）"
        )

        with mock.patch(
            "src.core.manga_insight.continuation.story_generator.HierarchicalSummaryGenerator",
            _SummaryGeneratorStub,
        ), mock.patch(
            "src.core.manga_insight.continuation.story_generator.create_chat_client",
            return_value=types.SimpleNamespace(close=mock.AsyncMock()),
        ):
            result = asyncio.run(generator.prepare_continuation_data())

        self.assertTrue(result["ready"])
        self.assertEqual(result["message"], "数据准备完成")
        self.assertIsNotNone(generator.storage.saved_summary)
        self.assertEqual(generator.storage.saved_summary[0], "story_summary")
        self.assertEqual(generator.storage.saved_summary[1]["content"], "自动补齐的故事概要")

    def test_generate_page_details_normalizes_form_name_to_form_id(self) -> None:
        from src.core.manga_insight.continuation.models import (
            ChapterScript,
            CharacterForm,
            CharacterProfile,
            ContinuationCharacters,
        )
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {
                    "characters": [
                        {"name": "Hero"},
                    ]
                }

        characters = ContinuationCharacters(
            book_id="book-1",
            characters=[
                CharacterProfile(
                    name="Hero",
                    aliases=[],
                    description="主角",
                    forms=[
                        CharacterForm(
                            form_id="battle",
                            form_name="Battle Form",
                            description="战斗形态",
                            reference_image="/tmp/hero-battle.png",
                        )
                    ],
                )
            ],
        )

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            load_characters=lambda: characters,
            format_characters_for_prompt=lambda: "角色档案",
        )
        generator.chat_client = types.SimpleNamespace(
            generate=mock.AsyncMock(
                return_value=json.dumps(
                    {
                        "page_number": 1,
                        "characters": ["Hero"],
                        "character_forms": [
                            {
                                "character": "Hero",
                                "form_id": "Battle Form",
                            }
                        ],
                        "description": "主角出场",
                        "dialogues": [{"character": "Hero", "text": "出发"}],
                    },
                    ensure_ascii=False,
                )
            )
        )

        page = asyncio.run(
            generator.generate_page_details(
                ChapterScript(
                    chapter_title="测试章节",
                    page_count=1,
                    script_text="第1页：主角出场",
                ),
                1,
            )
        )

        self.assertEqual(page.character_forms[0]["form_id"], "battle")
        self.assertEqual(page.character_forms[0]["form_name"], "Battle Form")


class ReferenceTokenResolverTests(unittest.TestCase):
    def test_build_continuation_candidates_include_placeholder_pages(self) -> None:
        from src.core.manga_insight.continuation.reference_tokens import build_continuation_reference_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            generated_path = os.path.join(temp_dir, "cont-1.png")
            with open(generated_path, "wb") as handle:
                handle.write(b"png")

            candidates = build_continuation_reference_candidates(
                total_original_pages=78,
                pages=[
                    {"page_number": 1, "image_url": generated_path, "status": "generated"},
                    {"page_number": 2, "image_url": "", "status": "pending"},
                ],
            )

            self.assertEqual(candidates[0]["token"], "continuation:1")
            self.assertEqual(candidates[0]["page_number"], 79)
            self.assertTrue(candidates[0]["has_image"])
            self.assertEqual(candidates[1]["token"], "continuation:2")
            self.assertEqual(candidates[1]["page_number"], 80)
            self.assertFalse(candidates[1]["has_image"])

    def test_resolve_reference_tokens_skips_placeholder_until_image_exists(self) -> None:
        from src.core.manga_insight.continuation.reference_tokens import resolve_reference_tokens

        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = os.path.join(temp_dir, "original-1.png")
            continuation_path = os.path.join(temp_dir, "continuation-1.png")
            with open(original_path, "wb") as handle:
                handle.write(b"png")

            candidates = [
                {
                    "token": "original:1",
                    "kind": "original",
                    "page_number": 1,
                    "path": original_path,
                    "has_image": True,
                },
                {
                    "token": "continuation:1",
                    "kind": "continuation",
                    "page_number": 79,
                    "path": "",
                    "has_image": False,
                },
            ]

            resolved = resolve_reference_tokens(
                ["original:1", "continuation:1"],
                candidates,
                current_page_number=80,
            )
            self.assertEqual([ref["path"] for ref in resolved], [original_path])

            with open(continuation_path, "wb") as handle:
                handle.write(b"png")
            candidates[1]["path"] = continuation_path
            candidates[1]["has_image"] = True
            resolved = resolve_reference_tokens(
                ["original:1", "continuation:1"],
                candidates,
                current_page_number=80,
            )
            self.assertEqual(
                [ref["path"] for ref in resolved],
                [original_path, continuation_path],
            )
