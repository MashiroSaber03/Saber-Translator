import json
import asyncio
import os
import sys
import tempfile
import types
import unittest
import importlib.util
from unittest import mock
from flask import Flask, Blueprint


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
        self.assertEqual(result["message"], "分析数据同步完成")
        self.assertIsNotNone(generator.storage.saved_summary)
        self.assertEqual(generator.storage.saved_summary[0], "story_summary")
        self.assertEqual(generator.storage.saved_summary[1]["content"], "自动补齐的故事概要")

    def test_prepare_continuation_data_reports_sync_metadata_and_non_destructive_character_merge(self) -> None:
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {"events": [{"event": "开端"}]}

        sync_result = {
            "characters": object(),
            "characters_added": 2,
            "total_characters": 5,
        }

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            sync_characters_from_timeline=mock.AsyncMock(return_value=sync_result)
        )
        generator._ensure_story_summary = mock.AsyncMock(return_value={"content": "已有故事概要"})

        result = asyncio.run(generator.prepare_continuation_data())

        self.assertTrue(result["ready"])
        self.assertTrue(result["story_summary_ready"])
        self.assertTrue(result["timeline_ready"])
        self.assertEqual(result["characters_added"], 2)
        self.assertEqual(result["total_characters"], 5)
        self.assertIn("synced_at", result)
        generator.char_manager.sync_characters_from_timeline.assert_awaited_once()

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
        generator._get_previous_story_summary = mock.AsyncMock(return_value="上一页承接")
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
                        "story_text": "主角出场并准备出发",
                        "dialogue_text": "Hero：出发",
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
        self.assertEqual(page.continuity_text, "上一页承接")
        self.assertEqual(page.story_text, "主角出场并准备出发")
        self.assertEqual(page.dialogue_text, "Hero：出发")

    def test_generate_page_details_sanitizes_camera_language_from_story_text(self) -> None:
        from src.core.manga_insight.continuation.models import ChapterScript
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {"characters": [{"name": "Hero"}]}

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            load_characters=lambda: None,
            format_characters_for_prompt=lambda: "（暂无角色信息）",
        )
        generator._get_previous_story_summary = mock.AsyncMock(return_value="原作末页摘要")
        generator.chat_client = types.SimpleNamespace(
            generate=mock.AsyncMock(
                return_value=json.dumps(
                    {
                        "page_number": 1,
                        "characters": ["Hero"],
                        "story_text": "俯视角度，男主冲向门口，特写表情，夜色中的走廊",
                        "dialogue_text": "Hero：等等我",
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
                    script_text="第1页：男主冲向门口",
                ),
                1,
            )
        )

        self.assertEqual(page.continuity_text, "原作末页摘要")
        self.assertNotIn("俯视", page.story_text)
        self.assertNotIn("特写", page.story_text)
        self.assertIn("男主冲向门口", page.story_text)
        self.assertIn("夜色中的走廊", page.story_text)

    def test_generate_page_details_keeps_core_action_when_camera_language_is_inline(self) -> None:
        from src.core.manga_insight.continuation.models import ChapterScript
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {"characters": [{"name": "Hero"}]}

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            load_characters=lambda: None,
            format_characters_for_prompt=lambda: "（暂无角色信息）",
        )
        generator._get_previous_story_summary = mock.AsyncMock(return_value="上一页剧情")
        generator.chat_client = types.SimpleNamespace(
            generate=mock.AsyncMock(
                return_value=json.dumps(
                    {
                        "page_number": 1,
                        "characters": ["Hero"],
                        "story_text": "俯视角度下男主冲向门口，女主在后方追上来",
                        "dialogue_text": "",
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
                    script_text="第1页：男主冲向门口，女主追上来",
                ),
                1,
            )
        )

        self.assertNotIn("俯视", page.story_text)
        self.assertIn("男主冲向门口", page.story_text)
        self.assertIn("女主在后方追上来", page.story_text)

    def test_generate_page_details_raises_on_invalid_json_response(self) -> None:
        from src.core.manga_insight.continuation.models import ChapterScript
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {"characters": []}

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            load_characters=lambda: None,
            format_characters_for_prompt=lambda: "（暂无角色信息）",
        )
        generator._get_previous_story_summary = mock.AsyncMock(return_value="上一页剧情")
        generator.chat_client = types.SimpleNamespace(
            generate=mock.AsyncMock(return_value="这不是合法的 JSON")
        )

        with self.assertRaises(ValueError):
            asyncio.run(
                generator.generate_page_details(
                    ChapterScript(
                        chapter_title="测试章节",
                        page_count=1,
                        script_text="第1页：主角出场",
                    ),
                    1,
                )
            )

    def test_generate_page_details_uses_previous_generated_story_for_continuity_after_first_page(self) -> None:
        from src.core.manga_insight.continuation.models import ChapterScript
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {"characters": [{"name": "Hero"}]}

            async def load_continuation_pages(self):
                return {
                    "pages": [
                        {
                            "page_number": 1,
                            "continuity_text": "原作末页摘要",
                            "story_text": "第一页里主角准备离开教室",
                            "dialogue_text": "Hero：我走了",
                            "characters": ["Hero"],
                            "character_forms": [],
                            "final_prompt": "",
                            "image_url": "",
                            "previous_url": "",
                            "status": "pending",
                        }
                    ]
                }

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            load_characters=lambda: None,
            format_characters_for_prompt=lambda: "（暂无角色信息）",
        )
        generator.chat_client = types.SimpleNamespace(
            generate=mock.AsyncMock(
                return_value=json.dumps(
                    {
                        "page_number": 2,
                        "characters": ["Hero"],
                        "story_text": "第二页里主角走到走廊，遇到同学",
                        "dialogue_text": "Hero：咦？",
                    },
                    ensure_ascii=False,
                )
            )
        )

        page = asyncio.run(
            generator.generate_page_details(
                ChapterScript(
                    chapter_title="测试章节",
                    page_count=2,
                    script_text="第1页：离开教室\n第2页：来到走廊",
                ),
                2,
            )
        )

        self.assertEqual(page.continuity_text, "第一页里主角准备离开教室")
        self.assertEqual(page.story_text, "第二页里主角走到走廊，遇到同学")

    def test_generate_page_details_accepts_markdown_json_via_structured_client(self) -> None:
        from src.core.manga_insight.continuation.models import ChapterScript
        from src.core.manga_insight.continuation.story_generator import StoryGenerator

        class _StorageStub:
            async def load_timeline(self):
                return {"characters": [{"name": "Hero"}]}

        async def _generate_parsed(_prompt, parser, **_kwargs):
            return parser(
                """```json
{
  "page_number": 1,
  "characters": ["Hero"],
  "story_text": "主角在雨夜走向路口",
  "dialogue_text": "Hero：我到了"
}
```"""
            )

        generator = StoryGenerator.__new__(StoryGenerator)
        generator.book_id = "book-1"
        generator.storage = _StorageStub()
        generator.char_manager = types.SimpleNamespace(
            load_characters=lambda: None,
            format_characters_for_prompt=lambda: "（暂无角色信息）",
        )
        generator._get_previous_story_summary = mock.AsyncMock(return_value="上一页剧情")
        generator.chat_client = types.SimpleNamespace(
            generate_parsed=mock.AsyncMock(side_effect=_generate_parsed)
        )

        page = asyncio.run(
            generator.generate_page_details(
                ChapterScript(
                    chapter_title="测试章节",
                    page_count=1,
                    script_text="第1页：主角在雨夜走向路口",
                ),
                1,
            )
        )

        self.assertEqual(page.continuity_text, "上一页剧情")
        self.assertEqual(page.story_text, "主角在雨夜走向路口")
        self.assertEqual(page.dialogue_text, "Hero：我到了")


class ContinuationStoryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self._isolated_module_names = [
            "isolated_manga_insight_pkg",
            "isolated_manga_insight_pkg.async_helpers",
            "isolated_manga_insight_pkg.response_builder",
            "isolated_manga_insight_pkg.continuation",
            "isolated_manga_insight_pkg.continuation.story_routes",
        ]
        self._original_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in self._isolated_module_names
        }

        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api", "manga_insight")
        continuation_dir = os.path.join(package_dir, "continuation")

        package_module = types.ModuleType("isolated_manga_insight_pkg")
        package_module.__path__ = [package_dir]
        package_module.manga_insight_bp = Blueprint(
            "isolated_manga_insight",
            __name__,
            url_prefix="/api/manga-insight",
        )
        sys.modules["isolated_manga_insight_pkg"] = package_module

        continuation_package = types.ModuleType("isolated_manga_insight_pkg.continuation")
        continuation_package.__path__ = [continuation_dir]
        sys.modules["isolated_manga_insight_pkg.continuation"] = continuation_package

        for module_name, file_path in (
            ("isolated_manga_insight_pkg.async_helpers", os.path.join(package_dir, "async_helpers.py")),
            ("isolated_manga_insight_pkg.response_builder", os.path.join(package_dir, "response_builder.py")),
            (
                "isolated_manga_insight_pkg.continuation.story_routes",
                os.path.join(continuation_dir, "story_routes.py"),
            ),
        ):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        self.story_routes = sys.modules["isolated_manga_insight_pkg.continuation.story_routes"]

    def tearDown(self) -> None:
        for module_name, original in self._original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_generate_single_page_details_returns_502_for_invalid_model_payload(self) -> None:
        async def _raise_invalid_payload(*_args, **_kwargs):
            raise ValueError("LLM 未返回有效的页面详情 JSON")

        with mock.patch.object(self.story_routes, "StoryGenerator") as story_generator_cls:
            story_generator_cls.return_value.generate_page_details = _raise_invalid_payload

            with self.app.test_request_context(
                "/api/manga-insight/book-1/continuation/pages/1",
                method="POST",
                json={
                    "script": {
                        "chapter_title": "测试章节",
                        "page_count": 3,
                        "script_text": "第1页：主角出场",
                    }
                },
            ):
                response = self.story_routes.generate_single_page_details("book-1", 1)

        flask_response, status_code = response
        self.assertEqual(status_code, 502)
        payload = flask_response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "LLM 未返回有效的页面详情 JSON")
        self.assertEqual(payload["error_code"], "INVALID_PAGE_DETAILS_RESPONSE")


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


class CharacterManagerSyncTests(unittest.TestCase):
    def test_sync_characters_from_timeline_appends_only_missing_characters(self) -> None:
        from src.core.manga_insight.continuation.character_manager import CharacterManager
        from src.core.manga_insight.continuation.models import (
            CharacterProfile,
            CharacterForm,
            ContinuationCharacters,
        )

        existing = ContinuationCharacters(
            book_id="book-1",
            characters=[
                CharacterProfile(
                    name="Hero",
                    aliases=["勇者"],
                    description="用户编辑后的描述",
                    forms=[
                        CharacterForm(
                            form_id="battle",
                            form_name="战斗形态",
                            description="保留原形态",
                            reference_image="/tmp/hero.png",
                        )
                    ],
                    enabled=False,
                )
            ],
        )
        timeline_characters = [
            CharacterProfile(
                name="Hero",
                aliases=["主角"],
                description="时间线里的角色描述",
                forms=[],
            ),
            CharacterProfile(
                name="Partner",
                aliases=["伙伴"],
                description="新角色",
                forms=[],
            ),
        ]

        manager = CharacterManager.__new__(CharacterManager)
        manager.book_id = "book-1"
        manager.load_characters = mock.Mock(return_value=existing)
        manager.load_characters_from_timeline = mock.AsyncMock(return_value=timeline_characters)
        manager.save_characters = mock.Mock(return_value=True)

        result = asyncio.run(manager.sync_characters_from_timeline())

        self.assertEqual(result["characters_added"], 1)
        self.assertEqual(result["total_characters"], 2)
        merged = result["characters"]
        self.assertEqual(len(merged.characters), 2)
        hero = merged.get_character("Hero")
        self.assertIsNotNone(hero)
        self.assertEqual(hero.description, "用户编辑后的描述")
        self.assertEqual(hero.enabled, False)
        self.assertEqual(len(hero.forms), 1)
        self.assertEqual(hero.forms[0].reference_image, "/tmp/hero.png")
        self.assertIsNotNone(merged.get_character("Partner"))
        manager.save_characters.assert_called_once()


class ContinuationCharacterRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self._isolated_module_names = [
            "isolated_manga_insight_character_pkg",
            "isolated_manga_insight_character_pkg.async_helpers",
            "isolated_manga_insight_character_pkg.response_builder",
            "isolated_manga_insight_character_pkg.continuation",
            "isolated_manga_insight_character_pkg.continuation.character_routes",
        ]
        self._original_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in self._isolated_module_names
        }

        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api", "manga_insight")
        continuation_dir = os.path.join(package_dir, "continuation")

        package_module = types.ModuleType("isolated_manga_insight_character_pkg")
        package_module.__path__ = [package_dir]
        package_module.manga_insight_bp = Blueprint(
            "isolated_manga_insight_character",
            __name__,
            url_prefix="/api/manga-insight",
        )
        sys.modules["isolated_manga_insight_character_pkg"] = package_module

        continuation_package = types.ModuleType("isolated_manga_insight_character_pkg.continuation")
        continuation_package.__path__ = [continuation_dir]
        sys.modules["isolated_manga_insight_character_pkg.continuation"] = continuation_package

        for module_name, file_path in (
            ("isolated_manga_insight_character_pkg.async_helpers", os.path.join(package_dir, "async_helpers.py")),
            ("isolated_manga_insight_character_pkg.response_builder", os.path.join(package_dir, "response_builder.py")),
            (
                "isolated_manga_insight_character_pkg.continuation.character_routes",
                os.path.join(continuation_dir, "character_routes.py"),
            ),
        ):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        self.character_routes = sys.modules["isolated_manga_insight_character_pkg.continuation.character_routes"]

    def tearDown(self) -> None:
        for module_name, original in self._original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_sync_continuation_analysis_returns_sync_payload(self) -> None:
        with mock.patch("src.core.manga_insight.continuation.StoryGenerator") as story_generator_cls:
            story_generator_cls.return_value.prepare_continuation_data = mock.AsyncMock(
                return_value={
                    "ready": True,
                    "message": "分析数据同步完成",
                    "story_summary_ready": True,
                    "timeline_ready": True,
                    "characters_added": 1,
                    "total_characters": 3,
                    "synced_at": "2026-05-21T20:00:00",
                }
            )

            with self.app.test_request_context(
                "/api/manga-insight/book-1/continuation/sync",
                method="POST",
            ):
                response = self.character_routes.sync_continuation_analysis("book-1")

        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["ready"])
        self.assertTrue(payload["story_summary_ready"])
        self.assertTrue(payload["timeline_ready"])
        self.assertEqual(payload["characters_added"], 1)
        self.assertEqual(payload["total_characters"], 3)
