import asyncio
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
import uuid
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

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub


def _demo_document(name: str = "测试角色") -> dict:
    return {
        "id": "doc_alpha",
        "bookId": "book-demo",
        "origin": {
            "type": "manual",
            "source_character": None,
            "source_pages": [],
        },
        "status": {
            "is_favorite": False,
            "frozen_sections": [],
            "last_validated_at": None,
        },
        "meta": {
            "title": name,
            "tags": ["测试"],
            "created_at": "2026-05-15T00:00:00",
            "updated_at": "2026-05-15T00:00:00",
        },
        "avatar": {
            "mode": "none",
            "asset_path": None,
            "source_page": None,
        },
        "identity": {
            "name": name,
            "aliases": [],
            "description": f"{name} 的简介",
            "personality": "冷静克制，目标明确。",
            "scenario": "故事持续推进，冲突尚未解决。",
        },
        "coreMessages": {
            "first_message": "我是测试角色，我们先谈正事。",
            "message_example": "<START>\n{{user}}: 现在怎么办？\n{{char}}: 先稳住局面。",
            "alternate_greetings": ["今天轮到我们推进计划了。"],
            "system_prompt": "保持角色设定稳定。",
            "post_history_instructions": "维持叙事连续性。",
            "creator_notes": "自动化测试文档",
            "character_version": "2.0.0",
        },
        "lorebook": {
            "name": f"{name} 世界书",
            "entries": [
                {
                    "id": "entry_1",
                    "comment": "角色基底",
                    "keys": [name],
                    "secondary_keys": [],
                    "content": f"{name} 会优先保护关键同伴。",
                    "enabled": True,
                    "constant": False,
                    "selective": True,
                    "priority": 100,
                    "position": "before_char",
                    "depth": 4,
                    "children": [],
                }
            ],
        },
        "regexScripts": [
            {
                "id": "regex_1",
                "scriptName": "状态隐藏",
                "findRegex": "<state>[\\s\\S]*?</state>",
                "replaceString": "",
                "placement": [2],
                "markdownOnly": False,
                "promptOnly": False,
                "runOnEdit": True,
                "disabled": False,
            }
        ],
        "stateTasks": [
            {
                "id": "task_1",
                "name": "初始化状态",
                "triggerTiming": "initialization",
                "interval": 0,
                "commands": "<<taskjs>>\nawait STscript('/setvar key=trust_score 20');\n<</taskjs>>",
                "disabled": False,
            }
        ],
        "chatPreset": {
            "opening_mode": "first_message",
        },
        "previewState": {
            "variables": {
                "trust_score": 20,
            },
            "messages": [],
        },
        "grounding": {
            "timeline_mode": "enhanced",
            "sample_pages": [1],
            "relationships": [],
            "key_moments": [],
        },
        "exportArtifacts": {},
    }


class CharacterStudioStorageTests(unittest.TestCase):
    def test_document_store_persists_index_and_document(self) -> None:
        from src.core.manga_insight.character_studio.store import CharacterStudioStore

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                store = CharacterStudioStore("book-demo")
                document = _demo_document()

                saved = asyncio.run(store.save_document(document))
                index_data = asyncio.run(store.load_index())
                loaded = asyncio.run(store.load_document(document["id"]))

                self.assertTrue(saved)
                self.assertEqual(index_data["documents"][0]["id"], document["id"])
                self.assertEqual(loaded["identity"]["name"], document["identity"]["name"])


class CharacterStudioExportAdapterTests(unittest.TestCase):
    def test_export_adapter_builds_v3_and_v2_payloads(self) -> None:
        from src.core.manga_insight.character_studio.adapters import build_export_bundle

        bundle = build_export_bundle(_demo_document())

        self.assertEqual(bundle["v3"]["spec"], "chara_card_v3")
        self.assertEqual(bundle["v3"]["data"]["name"], "测试角色")
        self.assertEqual(bundle["v2"]["spec"], "chara_card_v2")
        self.assertEqual(bundle["v2"]["data"]["name"], "测试角色")
        self.assertIsInstance(bundle["worldbook"]["entries"], list)

    def test_png_codec_roundtrip_preserves_v3_payload(self) -> None:
        from src.core.manga_insight.character_studio.adapters import build_export_bundle
        from src.core.manga_insight.character_studio.png_codec import CharacterStudioPngCodec

        bundle = build_export_bundle(_demo_document())
        png_bytes = CharacterStudioPngCodec.write_card_png(bundle["v3"])
        decoded = CharacterStudioPngCodec.read_card_png(png_bytes)

        self.assertEqual(decoded["spec"], "chara_card_v3")
        self.assertEqual(decoded["data"]["name"], "测试角色")

    def test_import_v3_payload_converts_lorebook_entries_to_internal_shape(self) -> None:
        from src.core.manga_insight.character_studio.adapters import import_document_payload

        payload = {
            "spec": "chara_card_v3",
            "data": {
                "name": "导入角色",
                "description": "描述",
                "personality": "性格",
                "scenario": "场景",
                "first_mes": "主问候",
                "mes_example": "<START>",
                "creator_notes": "备注",
                "system_prompt": "系统提示词",
                "post_history_instructions": "追加说明",
                "character_version": "3.0.0",
                "alternate_greetings": ["问候1"],
                "tags": ["测试"],
                "extensions": {
                    "regex_scripts": [{"scriptName": "脚本A", "findRegex": "abc", "replaceString": "def"}],
                    "xiaobaix-tasks": {"tasks": [{"name": "任务A", "commands": "<<taskjs>>\n<</taskjs>>"}]},
                },
                "character_book": {
                    "name": "导入世界书",
                    "entries": [
                        {
                            "id": "v3_1",
                            "keys": ["主角"],
                            "secondary_keys": ["别称"],
                            "comment": "人物条目",
                            "content": "条目内容",
                            "enabled": True,
                            "constant": False,
                            "selective": True,
                            "insertion_order": 200,
                            "position": "before_char",
                            "extensions": {"depth": 5, "probability": 88, "prevent_recursion": False},
                        }
                    ],
                },
            },
        }

        document = import_document_payload("book-demo", payload)
        entry = document["lorebook"]["entries"][0]

        self.assertEqual(entry["id"], "v3_1")
        self.assertEqual(entry["keys"], ["主角"])
        self.assertEqual(entry["secondary_keys"], ["别称"])
        self.assertEqual(entry["priority"], 200)
        self.assertEqual(entry["depth"], 5)
        self.assertEqual(entry["probability"], 88)
        self.assertFalse(entry["prevent_recursion"])


class CharacterStudioPreviewTests(unittest.TestCase):
    def test_initialize_preview_session_includes_doc_id_for_frontend_contract(self) -> None:
        from src.core.manga_insight.character_studio.preview import initialize_preview_session

        session = initialize_preview_session(_demo_document())

        self.assertIn("doc_id", session)
        self.assertEqual(session["doc_id"], "doc_alpha")

    def test_generate_review_persists_last_review_payload(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                asyncio.run(service.store.save_document(document))

                with mock.patch.object(service, "_create_chat_client", return_value=None):
                    updated = asyncio.run(service.generate_section(document["id"], "review"))

                self.assertIn("last_review", updated["exportArtifacts"])
                self.assertIn("summary", updated["exportArtifacts"]["last_review"])
                reloaded = asyncio.run(service.store.load_document(document["id"]))
                self.assertIn("last_review", reloaded["exportArtifacts"])

    def test_preview_chat_applies_runtime_rules_for_tasks_regex_and_lorebook(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                document["coreMessages"]["first_message"] = "开场白"
                document["regexScripts"] = [
                    {
                        "id": "regex_apply",
                        "scriptName": "问候替换",
                        "findRegex": "hello",
                        "replaceString": "hi",
                        "placement": [1],
                        "markdownOnly": False,
                        "promptOnly": False,
                        "runOnEdit": True,
                        "disabled": False,
                    },
                    {
                        "id": "regex_skip",
                        "scriptName": "停用编辑时运行",
                        "findRegex": "hero",
                        "replaceString": "champion",
                        "placement": [1],
                        "markdownOnly": False,
                        "promptOnly": False,
                        "runOnEdit": False,
                        "disabled": False,
                    },
                ]
                document["lorebook"]["entries"] = [
                    {
                        "id": "entry_secondary",
                        "comment": "次级关键词命中",
                        "keys": ["primary"],
                        "secondary_keys": ["ally"],
                        "content": "secondary content",
                        "enabled": True,
                        "constant": False,
                        "selective": False,
                        "priority": 100,
                        "position": "before_char",
                        "depth": 4,
                        "probability": 100,
                        "prevent_recursion": False,
                        "children": [],
                    },
                    {
                        "id": "entry_regex",
                        "comment": "正则关键词命中",
                        "keys": ["fo+d"],
                        "secondary_keys": [],
                        "content": "regex content",
                        "enabled": True,
                        "constant": False,
                        "selective": False,
                        "priority": 100,
                        "position": "before_char",
                        "depth": 4,
                        "probability": 100,
                        "prevent_recursion": False,
                        "use_regex": True,
                        "children": [],
                    },
                    {
                        "id": "entry_prob_zero",
                        "comment": "不会命中",
                        "keys": ["hero"],
                        "secondary_keys": [],
                        "content": "should not appear",
                        "enabled": True,
                        "constant": False,
                        "selective": False,
                        "priority": 100,
                        "position": "before_char",
                        "depth": 4,
                        "probability": 0,
                        "prevent_recursion": False,
                        "children": [],
                    },
                    {
                        "id": "entry_once",
                        "comment": "仅首次注入",
                        "keys": ["hero"],
                        "secondary_keys": [],
                        "content": "inject once",
                        "enabled": True,
                        "constant": False,
                        "selective": False,
                        "priority": 100,
                        "position": "before_char",
                        "depth": 4,
                        "probability": 100,
                        "prevent_recursion": True,
                        "children": [],
                    },
                ]
                document["stateTasks"] = [
                    {
                        "id": "task_init",
                        "name": "初始化状态",
                        "triggerTiming": "initialization",
                        "interval": 0,
                        "commands": "<<taskjs>>\nawait STscript('/setvar key=boot ready');\n<</taskjs>>",
                        "disabled": False,
                    },
                    {
                        "id": "task_receive_every_two",
                        "name": "双轮接收",
                        "triggerTiming": "message_received",
                        "interval": 2,
                        "commands": "<<taskjs>>\nawait STscript('/setvar key=received twice');\n<</taskjs>>",
                        "disabled": False,
                    },
                    {
                        "id": "task_after_reply",
                        "name": "回复后任务",
                        "triggerTiming": "message_sent",
                        "interval": 1,
                        "commands": "<<taskjs>>\nawait STscript('/setvar key=replied yes');\n<</taskjs>>",
                        "disabled": False,
                    },
                ]
                asyncio.run(service.store.save_document(document))

                with mock.patch.object(service, "_create_chat_client", return_value=None):
                    session1 = asyncio.run(service.preview_chat(document["id"], "hello ally hero food"))
                    session2 = asyncio.run(service.preview_chat(document["id"], "hello hero"))

                self.assertEqual(session1["messages"][1]["content"], "hi ally hero food")
                self.assertEqual(session1["variables"]["boot"], "ready")
                self.assertEqual(session1["variables"]["replied"], "yes")
                self.assertNotIn("received", session1["variables"])
                self.assertIn("received", session2["variables"])

                lorebook_comments = [
                    item.get("comment")
                    for item in session2["log"]
                    if item.get("type") == "lorebook"
                ]
                self.assertIn("次级关键词命中", lorebook_comments)
                self.assertIn("正则关键词命中", lorebook_comments)
                self.assertNotIn("不会命中", lorebook_comments)
                self.assertEqual(lorebook_comments.count("仅首次注入"), 1)


class CharacterStudioRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self._isolated_module_names = [
            "isolated_character_studio_pkg",
            "isolated_character_studio_pkg.async_helpers",
            "isolated_character_studio_pkg.response_builder",
            "isolated_character_studio_pkg.character_studio_routes",
        ]
        self._original_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in self._isolated_module_names
        }

        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api", "manga_insight")

        package_module = types.ModuleType("isolated_character_studio_pkg")
        package_module.__path__ = [package_dir]
        package_module.manga_insight_bp = Blueprint(
            "isolated_character_studio",
            __name__,
            url_prefix="/api/manga-insight",
        )
        sys.modules["isolated_character_studio_pkg"] = package_module

        for module_name, file_path in (
            ("isolated_character_studio_pkg.async_helpers", os.path.join(package_dir, "async_helpers.py")),
            ("isolated_character_studio_pkg.response_builder", os.path.join(package_dir, "response_builder.py")),
            ("isolated_character_studio_pkg.character_studio_routes", os.path.join(package_dir, "character_studio_routes.py")),
        ):
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        self.bp = package_module.manga_insight_bp
        self.app.register_blueprint(self.bp)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        for module_name, original in self._original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_character_studio_index_route_exists(self) -> None:
        with mock.patch("isolated_character_studio_pkg.character_studio_routes.CharacterStudioService") as service_cls:
            service_cls.return_value.get_index_payload = mock.AsyncMock(return_value={
                "book_id": "book-demo",
                "documents": [],
                "candidates": [],
                "count": 0,
            })
            response = self.client.get("/api/manga-insight/book-demo/character-studio/index")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["book_id"], "book-demo")

    def test_character_studio_document_route_includes_preview_session(self) -> None:
        with mock.patch("isolated_character_studio_pkg.character_studio_routes.CharacterStudioService") as service_cls:
            service = service_cls.return_value
            service.store.load_document = mock.AsyncMock(return_value=_demo_document())
            service.store.load_preview_session = mock.AsyncMock(return_value={
                "doc_id": "doc_alpha",
                "messages": [{"role": "assistant", "content": "已恢复预览"}],
                "variables": {"trust_score": 20},
                "log": [{"type": "task", "name": "初始化状态"}],
            })
            response = self.client.get("/api/manga-insight/book-demo/character-studio/documents/doc_alpha")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["preview_session"]["messages"][0]["content"], "已恢复预览")


class CharacterStudioSpaRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._isolated_module_names = [
            "isolated_main_routes_pkg",
            "isolated_main_routes_pkg.routes",
        ]
        self._original_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in self._isolated_module_names
        }

        package_dir = os.path.join(PROJECT_ROOT, "src", "app")
        package_module = types.ModuleType("isolated_main_routes_pkg")
        package_module.__path__ = [package_dir]
        package_module.main_bp = Blueprint("isolated_main_bp", __name__)
        sys.modules["isolated_main_routes_pkg"] = package_module

        spec = importlib.util.spec_from_file_location(
            "isolated_main_routes_pkg.routes",
            os.path.join(package_dir, "routes.py"),
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["isolated_main_routes_pkg.routes"] = module
        spec.loader.exec_module(module)

        self.app = Flask(
            __name__,
            static_folder=os.path.join(PROJECT_ROOT, "src", "app", "static"),
            static_url_path="",
        )
        self.app.register_blueprint(package_module.main_bp)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        for module_name, original in self._original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_character_studio_spa_path_serves_vue_app(self) -> None:
        response = self.client.get("/insight/character-studio?book=book-demo")
        try:
            self.assertEqual(response.status_code, 200)
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
