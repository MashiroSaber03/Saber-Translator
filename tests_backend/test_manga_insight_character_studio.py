import asyncio
import base64
import json
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
        "grounding": {
            "timeline_mode": "enhanced",
            "sample_pages": [1],
            "relationships": [],
            "key_moments": [],
        },
        "exportArtifacts": {},
    }


def _tiny_png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0uoAAAAASUVORK5CYII="
    )


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

    def test_generate_review_persists_last_review_payload_using_compressed_context(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def generate(self, prompt: str, temperature: float = 0.5):
                self.prompts.append(prompt)
                return json.dumps({
                    "summary": "当前角色卡基本贴合剧情，但世界书覆盖还不够。",
                    "issues": ["缺少对中盘冲突升级的明确刻画"],
                    "suggestions": ["补一条与情报统筹职责相关的世界书条目"],
                }, ensure_ascii=False)

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要强调测试角色在中盘承担统筹与风险评估职责，并多次主导关键抉择。",
                    "source": "chapters",
                    "group_count": 2,
                    "char_count": 40,
                }))

                fake_client = FakeClient()
                with mock.patch.object(service, "_create_chat_client", return_value=fake_client):
                    updated = asyncio.run(service.generate_section(document["id"], "review"))

                self.assertIn("last_review", updated["exportArtifacts"])
                self.assertIn("summary", updated["exportArtifacts"]["last_review"])
                self.assertEqual(updated["exportArtifacts"]["last_review"]["summary"], "当前角色卡基本贴合剧情，但世界书覆盖还不够。")
                reloaded = asyncio.run(service.store.load_document(document["id"]))
                self.assertIn("last_review", reloaded["exportArtifacts"])
                self.assertTrue(any("压缩摘要强调测试角色在中盘承担统筹与风险评估职责" in prompt for prompt in fake_client.prompts))
                self.assertTrue(any("目标角色名" in prompt for prompt in fake_client.prompts))
                self.assertTrue(any("当前角色卡内容（审查对象）" in prompt for prompt in fake_client.prompts))

    def test_create_document_from_candidate_only_prefills_name(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                asyncio.run(service.storage.save_timeline({
                    "mode": "enhanced",
                    "characters": [
                        {
                            "name": "候选角色",
                            "aliases": ["别称"],
                            "description": "旧简介",
                            "arc": "旧成长线",
                            "relationships": [{"character": "同伴", "relation": "队友"}],
                            "key_moments": [{"page": 3, "event": "登场"}],
                        }
                    ],
                }))
                asyncio.run(service.storage.save_overview({
                    "book_summary": "这是一本书的概要。",
                }))

                created = asyncio.run(service.create_document(candidate_name="候选角色"))

                self.assertEqual(created["origin"]["type"], "analysis")
                self.assertEqual(created["origin"]["source_character"], "候选角色")
                self.assertEqual(created["identity"]["name"], "候选角色")
                self.assertEqual(created["meta"]["title"], "候选角色")
                self.assertEqual(created["identity"]["aliases"], [])
                self.assertEqual(created["identity"]["description"], "")
                self.assertEqual(created["identity"]["personality"], "")
                self.assertEqual(created["identity"]["scenario"], "")
                self.assertEqual(created["meta"]["tags"], [])
                self.assertEqual(created["coreMessages"]["first_message"], "")
                self.assertEqual(created["coreMessages"]["alternate_greetings"], [])
                self.assertEqual(created["lorebook"]["entries"], [])
                self.assertEqual(created["regexScripts"], [])
                self.assertEqual(created["stateTasks"], [])
                self.assertEqual(created["grounding"]["timeline_mode"], "")
                self.assertEqual(created["grounding"]["sample_pages"], [])
                self.assertEqual(created["grounding"]["relationships"], [])
                self.assertEqual(created["grounding"]["key_moments"], [])

    def test_generate_full_uses_compressed_context_and_updates_all_sections(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def generate(self, prompt: str, temperature: float = 0.5):
                self.prompts.append(prompt)
                return json.dumps({
                    "identity": {
                        "name": "候选角色",
                        "description": "基于压缩摘要整理出的角色简介",
                        "personality": "谨慎、聪明、带一点疏离感",
                        "scenario": "故事进入中盘，冲突不断升级。",
                    },
                    "coreMessages": {
                        "first_message": "我是候选角色。先同步一下局势。",
                        "message_example": "<START>\n{{user}}: 现在情况如何？\n{{char}}: 先别慌，我们按计划来。",
                        "alternate_greetings": [
                            "你终于来了，我们直接进入正题。",
                            "先确认情报，再决定下一步。"
                        ],
                        "system_prompt": "保持角色克制而清醒。",
                        "post_history_instructions": "回应时维持剧情连续性。",
                        "creator_notes": "来自压缩摘要的整卡补全。",
                        "character_version": "2.1.0",
                    },
                    "lorebook": {
                        "entries": [
                            {
                                "id": "entry_full",
                                "comment": "角色基底",
                                "keys": ["候选角色"],
                                "secondary_keys": [],
                                "content": "他总是先评估风险再行动。",
                                "enabled": True,
                                "constant": False,
                                "selective": True,
                                "priority": 120,
                                "position": "before_char",
                                "depth": 4,
                                "children": [],
                            }
                        ]
                    },
                    "regexScripts": [
                        {
                            "id": "regex_full",
                            "scriptName": "隐藏状态块",
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
                            "id": "task_full",
                            "name": "初始化关系值",
                            "triggerTiming": "initialization",
                            "interval": 0,
                            "commands": "<<taskjs>>\nawait STscript('/setvar key=trust_score 35');\n<</taskjs>>",
                            "disabled": False,
                        }
                    ],
                }, ensure_ascii=False)

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document("候选角色")
                document["identity"]["description"] = ""
                document["identity"]["personality"] = ""
                document["identity"]["scenario"] = ""
                document["coreMessages"]["first_message"] = ""
                document["coreMessages"]["alternate_greetings"] = []
                document["lorebook"]["entries"] = []
                document["regexScripts"] = []
                document["stateTasks"] = []
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要里明确提到候选角色在中盘承担情报统筹职责，并多次在危机中保持冷静。",
                    "source": "chapters",
                    "group_count": 3,
                    "char_count": 42,
                }))

                fake_client = FakeClient()
                with mock.patch.object(service, "_create_chat_client", return_value=fake_client):
                    updated = asyncio.run(service.generate_section(document["id"], "full"))

                self.assertEqual(updated["identity"]["description"], "基于压缩摘要整理出的角色简介")
                self.assertEqual(updated["coreMessages"]["first_message"], "我是候选角色。先同步一下局势。")
                self.assertEqual(updated["lorebook"]["entries"][0]["comment"], "角色基底")
                self.assertEqual(updated["regexScripts"][0]["scriptName"], "隐藏状态块")
                self.assertEqual(updated["stateTasks"][0]["name"], "初始化关系值")
                self.assertTrue(any("压缩摘要里明确提到候选角色" in prompt for prompt in fake_client.prompts))

    def test_generate_full_raises_when_llm_returns_invalid_json_and_document_stays_unchanged(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            async def generate(self, prompt: str, temperature: float = 0.5):
                return "这不是 JSON"

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document("候选角色")
                document["identity"]["description"] = ""
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要内容",
                    "source": "chapters",
                    "group_count": 1,
                    "char_count": 6,
                }))

                with mock.patch.object(service, "_create_chat_client", return_value=FakeClient()):
                    with self.assertRaises(ValueError):
                        asyncio.run(service.generate_section(document["id"], "full"))

                reloaded = asyncio.run(service.store.load_document(document["id"]))
                self.assertEqual(reloaded["identity"]["description"], "")

    def test_generate_full_accepts_markdown_json_via_structured_client(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def generate_parsed(self, prompt, parser, temperature: float = 0.5):
                self.prompts.append(prompt)
                return parser(
                    """```json
{
  "identity": {
    "name": "候选角色",
    "description": "Markdown JSON 中的角色简介",
    "personality": "冷静、克制",
    "scenario": "冲突进入升级阶段。"
  },
  "coreMessages": {
    "first_message": "我是候选角色，我们直接同步情报。",
    "message_example": "<START>\\n{{user}}: 说重点\\n{{char}}: 先确认风险。",
    "alternate_greetings": ["先看局势。"],
    "system_prompt": "保持冷静和判断力。",
    "post_history_instructions": "",
    "creator_notes": "",
    "character_version": "2.0.0"
  },
  "lorebook": {
    "entries": [
      {
        "comment": "角色基底",
        "keys": ["候选角色"],
        "secondary_keys": [],
        "content": "她总会先评估局势。",
        "enabled": true,
        "constant": false,
        "selective": true,
        "priority": 100,
        "position": "before_char",
        "depth": 4,
        "children": []
      }
    ]
  },
  "regexScripts": [],
  "stateTasks": []
}
```"""
                )

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document("候选角色")
                document["identity"]["description"] = ""
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要内容",
                    "source": "chapters",
                    "group_count": 1,
                    "char_count": 6,
                }))

                fake_client = FakeClient()
                with mock.patch.object(service, "_create_chat_client", return_value=fake_client):
                    updated = asyncio.run(service.generate_section(document["id"], "full"))

                self.assertEqual(updated["identity"]["description"], "Markdown JSON 中的角色简介")
                self.assertEqual(updated["coreMessages"]["first_message"], "我是候选角色，我们直接同步情报。")
                self.assertEqual(updated["lorebook"]["entries"][0]["comment"], "角色基底")
                self.assertTrue(any("目标角色名" in prompt for prompt in fake_client.prompts))

    def test_generate_full_retries_invalid_generated_payload_before_saving(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService
        from src.shared.openai_execution import OpenAICompatibleBusinessRetryableError

        class FakeClient:
            async def generate_parsed(self, prompt, parser, temperature: float = 0.5):
                attempts = [
                    json.dumps({
                        "identity": {
                            "name": "候选角色",
                            "description": "第一次生成",
                            "personality": "谨慎",
                            "scenario": "冲突升级。",
                        },
                        "coreMessages": {
                            "first_message": "第一次问候",
                            "message_example": "<START>",
                            "alternate_greetings": [],
                            "system_prompt": "",
                            "post_history_instructions": "",
                            "creator_notes": "",
                            "character_version": "2.0.0",
                        },
                        "lorebook": {
                            "entries": []
                        },
                        "regexScripts": [
                            {
                                "scriptName": "坏脚本",
                                "findRegex": "",
                                "replaceString": "",
                                "placement": [2],
                                "markdownOnly": False,
                                "promptOnly": False,
                                "runOnEdit": True,
                                "disabled": False,
                            }
                        ],
                        "stateTasks": [],
                    }, ensure_ascii=False),
                    json.dumps({
                        "identity": {
                            "name": "候选角色",
                            "description": "第二次生成",
                            "personality": "谨慎",
                            "scenario": "冲突升级。",
                        },
                        "coreMessages": {
                            "first_message": "第二次问候",
                            "message_example": "<START>",
                            "alternate_greetings": [],
                            "system_prompt": "",
                            "post_history_instructions": "",
                            "creator_notes": "",
                            "character_version": "2.0.0",
                        },
                        "lorebook": {
                            "entries": [
                                {
                                    "comment": "角色基底",
                                    "keys": ["候选角色"],
                                    "secondary_keys": [],
                                    "content": "她总会先评估局势。",
                                    "enabled": True,
                                    "constant": False,
                                    "selective": True,
                                    "priority": 100,
                                    "position": "before_char",
                                    "depth": 4,
                                    "children": [],
                                }
                            ]
                        },
                        "regexScripts": [],
                        "stateTasks": [],
                    }, ensure_ascii=False),
                ]
                last_error = None
                for raw in attempts:
                    try:
                        return parser(raw)
                    except OpenAICompatibleBusinessRetryableError as exc:
                        last_error = exc
                raise AssertionError(f"parser should have accepted the final payload, last_error={last_error}")

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document("候选角色")
                document["identity"]["description"] = ""
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要内容",
                    "source": "chapters",
                    "group_count": 1,
                    "char_count": 6,
                }))

                with mock.patch.object(service, "_create_chat_client", return_value=FakeClient()):
                    updated = asyncio.run(service.generate_section(document["id"], "full"))

                self.assertEqual(updated["identity"]["description"], "第二次生成")
                self.assertEqual(updated["coreMessages"]["first_message"], "第二次问候")
                self.assertEqual(updated["regexScripts"], [])

    def test_generate_full_normalizes_common_alias_based_payload_shapes(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            async def generate_parsed(self, prompt, parser, temperature: float = 0.5):
                return parser(json.dumps({
                    "identity": {
                        "name": "候选角色",
                        "description": "使用了别名字段的整卡。",
                        "personality": "认真直接",
                        "scenario": "关键剧情节点。",
                    },
                    "coreMessages": {
                        "first_message": "我是候选角色。",
                        "message_example": "<START>",
                        "alternate_greetings": ["你好。"],
                        "system_prompt": "",
                        "post_history_instructions": "",
                        "creator_notes": "",
                        "character_version": "2.0.0",
                    },
                    "lorebook": {
                        "name": "候选角色 世界书",
                        "entries": [
                            {
                                "name": "五胞胎身份",
                                "content": "角色基础信息",
                                "keys": ["五胞胎", "中野家"],
                            },
                            {
                                "key": "教师梦想",
                                "content": "梦想成为教师",
                            }
                        ],
                    },
                    "regexScripts": [
                        {
                            "name": "食物兴奋反应",
                            "regex": "(肉包|汉堡)",
                            "replacement": "喜欢的食物：$1",
                            "placement": ["2"],
                            "markdownOnly": "false",
                            "promptOnly": "false",
                            "runOnEdit": "true",
                            "condition": "random(0.3)",
                        }
                    ],
                    "stateTasks": [
                        {
                            "name": "好感度追踪",
                            "description": "根据对话推进关系",
                            "trigger": "conversation_end",
                            "action": "update_affection",
                        }
                    ],
                }, ensure_ascii=False))

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document("候选角色")
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要内容",
                    "source": "chapters",
                    "group_count": 1,
                    "char_count": 6,
                }))

                with mock.patch.object(service, "_create_chat_client", return_value=FakeClient()):
                    updated = asyncio.run(service.generate_section(document["id"], "full"))

                self.assertEqual(updated["lorebook"]["entries"][0]["comment"], "五胞胎身份")
                self.assertEqual(updated["lorebook"]["entries"][0]["keys"], ["五胞胎", "中野家"])
                self.assertEqual(updated["lorebook"]["entries"][1]["comment"], "教师梦想")
                self.assertEqual(updated["lorebook"]["entries"][1]["keys"], ["教师梦想"])
                self.assertEqual(updated["regexScripts"][0]["scriptName"], "食物兴奋反应")
                self.assertEqual(updated["regexScripts"][0]["findRegex"], "(肉包|汉堡)")
                self.assertEqual(updated["regexScripts"][0]["placement"], [2])
                self.assertFalse(updated["regexScripts"][0]["markdownOnly"])
                self.assertFalse(updated["regexScripts"][0]["promptOnly"])
                self.assertTrue(updated["regexScripts"][0]["runOnEdit"])
                self.assertTrue(updated["regexScripts"][0]["disabled"])
                self.assertEqual(updated["stateTasks"][0]["triggerTiming"], "message_sent")
                self.assertTrue(updated["stateTasks"][0]["disabled"])
                self.assertIn("AI generated placeholder task metadata", updated["stateTasks"][0]["commands"])

    def test_translate_section_keeps_meta_title_in_sync_with_identity_name(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            async def generate_parsed(self, prompt, parser, temperature: float = 0.5):
                return parser(json.dumps({
                    "identity": {
                        "name": "中野五月",
                        "description": "翻译后的简介",
                        "personality": "翻译后的人设",
                        "scenario": "翻译后的场景",
                    },
                    "coreMessages": {
                        "first_message": "翻译后的首句",
                        "message_example": "<START>",
                        "alternate_greetings": ["你好。"],
                        "system_prompt": "",
                        "post_history_instructions": "",
                        "creator_notes": "",
                        "character_version": "2.0.0",
                    },
                }, ensure_ascii=False))

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document("Itsuki")
                document["meta"]["title"] = "Itsuki"
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要内容",
                    "source": "chapters",
                    "group_count": 1,
                    "char_count": 6,
                }))

                with mock.patch.object(service, "_create_chat_client", return_value=FakeClient()):
                    updated = asyncio.run(service.generate_section(document["id"], "translate"))

                self.assertEqual(updated["identity"]["name"], "中野五月")
                self.assertEqual(updated["meta"]["title"], "中野五月")

    def test_run_agent_uses_compressed_context_with_card_and_preview_runtime_context(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def generate(self, prompt: str, temperature: float = 0.6):
                self.prompts.append(prompt)
                return "建议补一条世界书，并输出一个 patch。\n```json:patch\n{\"set\":{\"identity.description\":\"补强后的简介\"}}\n```"

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                asyncio.run(service.store.save_document(document))
                asyncio.run(service.storage.save_compressed_context({
                    "context": "压缩摘要指出测试角色在关键危机中总是先评估风险，再统一调度同伴行动。",
                    "source": "chapters",
                    "group_count": 2,
                    "char_count": 39,
                }))
                state = asyncio.run(service.create_new_chat_session(document["id"]))
                session = state["active_session"]
                session["messages"] = [
                    {
                        "message_id": "msg_user",
                        "role": "user",
                        "content": "现在局势怎么样？",
                        "attachments": [],
                        "runtime_log": [],
                        "variables_snapshot": {"trust_score": 44},
                        "generation_meta": {},
                        "created_at": "2026-05-15T00:00:00",
                        "updated_at": "2026-05-15T00:00:00",
                    },
                    {
                        "message_id": "msg_assistant",
                        "role": "assistant",
                        "content": "先别急，我正在梳理风险。",
                        "attachments": [],
                        "runtime_log": [{"type": "task", "name": "初始化状态"}],
                        "variables_snapshot": {"trust_score": 44},
                        "generation_meta": {},
                        "created_at": "2026-05-15T00:00:01",
                        "updated_at": "2026-05-15T00:00:01",
                    },
                ]
                session["variables"] = {"trust_score": 44}
                asyncio.run(service.store.save_chat_session(document["id"], session))

                fake_client = FakeClient()
                with mock.patch.object(service, "_create_chat_client", return_value=fake_client):
                    result = asyncio.run(service.run_agent(document["id"], "请审查这张卡并给出 patch 建议"))

                self.assertIn("压缩摘要指出测试角色在关键危机中总是先评估风险", result["context"])
                self.assertIn("当前角色卡", result["context"])
                self.assertIn("最近对话", result["context"])
                self.assertIn("trust_score", result["context"])
                self.assertIn("最近命中/执行日志", result["context"])
                self.assertIn("初始化状态", result["context"])
                self.assertIn("压缩摘要指出测试角色在关键危机中总是先评估风险", fake_client.prompts[0])
                self.assertIn("唯一外部事实来源", fake_client.prompts[0])
                self.assertIn("请审查这张卡并给出 patch 建议", fake_client.prompts[0])
                self.assertIn("worldbook_add", fake_client.prompts[0])
                self.assertIn("regex_add", fake_client.prompts[0])
                self.assertIn("task_add", fake_client.prompts[0])
                self.assertIn("不要使用 name、regex、replacement、condition", fake_client.prompts[0])

class CharacterStudioChatServiceTests(unittest.TestCase):
    def test_get_chat_state_bootstraps_active_session_with_opening_message(self) -> None:
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

                state = asyncio.run(service.get_chat_state(document["id"]))

                self.assertEqual(state["doc_id"], document["id"])
                self.assertIsNotNone(state["active_session"])
                self.assertEqual(state["archived_sessions"], [])
                self.assertEqual(
                    state["active_session"]["messages"][0]["content"],
                    document["coreMessages"]["first_message"],
                )

    def test_send_chat_message_uses_vlm_and_records_runtime_state(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeVlmClient:
            def __init__(self) -> None:
                self.requests: list[list[dict]] = []

            async def generate_messages(self, messages, *, on_stream_chunk=None):
                self.requests.append(messages)
                if on_stream_chunk:
                    on_stream_chunk("先")
                    on_stream_chunk("别急，")
                    on_stream_chunk("按计划推进。")
                return "先别急，按计划推进。"

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                document["regexScripts"] = [
                    {
                        "id": "regex_1",
                        "scriptName": "问候归一",
                        "findRegex": "hello",
                        "replaceString": "hi",
                        "placement": [1],
                        "markdownOnly": False,
                        "promptOnly": False,
                        "runOnEdit": True,
                        "disabled": False,
                    }
                ]
                document["stateTasks"].append(
                    {
                        "id": "task_2",
                        "name": "收到消息打标",
                        "triggerTiming": "message_received",
                        "interval": 1,
                        "commands": "<<taskjs>>\nawait STscript('/setvar key=received yes');\n<</taskjs>>",
                        "disabled": False,
                    }
                )
                asyncio.run(service.store.save_document(document))

                state = asyncio.run(service.get_chat_state(document["id"]))
                session_id = state["active_session"]["session_id"]
                fake_client = FakeVlmClient()
                events: list[dict] = []

                with mock.patch.object(service, "_create_vlm_client", return_value=fake_client):
                    updated = asyncio.run(
                        service.send_chat_message(
                            document["id"],
                            session_id=session_id,
                            content="hello 测试角色",
                            on_event=events.append,
                        )
                    )

                assistant_message = updated["session"]["messages"][-1]
                self.assertEqual(assistant_message["role"], "assistant")
                self.assertEqual(assistant_message["content"], "先别急，按计划推进。")
                self.assertEqual(assistant_message["variables_snapshot"]["trust_score"], "20")
                self.assertEqual(assistant_message["variables_snapshot"]["received"], "yes")
                runtime_types = {item["type"] for item in assistant_message["runtime_log"]}
                self.assertIn("regex", runtime_types)
                self.assertIn("lorebook", runtime_types)
                self.assertIn("task", runtime_types)
                event_types = {item["type"] for item in events}
                self.assertIn("assistant_delta", event_types)
                self.assertIn("assistant_done", event_types)
                self.assertIsInstance(fake_client.requests[-1][-1]["content"], str)

    def test_send_chat_message_with_image_builds_multimodal_user_payload(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeVlmClient:
            def __init__(self) -> None:
                self.requests: list[list[dict]] = []

            async def generate_messages(self, messages, *, on_stream_chunk=None):
                self.requests.append(messages)
                return "我看到了这张图片。"

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                asyncio.run(service.store.save_document(document))

                state = asyncio.run(service.get_chat_state(document["id"]))
                session_id = state["active_session"]["session_id"]
                fake_client = FakeVlmClient()

                with mock.patch.object(service, "_create_vlm_client", return_value=fake_client):
                    updated = asyncio.run(
                        service.send_chat_message(
                            document["id"],
                            session_id=session_id,
                            content="看看这张图里有什么",
                            attachments=[
                                {
                                    "filename": "scene.png",
                                    "mime_type": "image/png",
                                    "bytes": _tiny_png_bytes(),
                                }
                            ],
                        )
                    )

                user_message = updated["session"]["messages"][-2]
                self.assertEqual(user_message["attachments"][0]["mime_type"], "image/png")
                self.assertTrue(user_message["attachments"][0]["asset_path"].endswith(".png"))
                last_user_payload = fake_client.requests[-1][-1]["content"]
                self.assertIsInstance(last_user_payload, list)
                self.assertEqual(last_user_payload[0]["type"], "image_url")
                self.assertEqual(last_user_payload[-1]["type"], "text")

    def test_edit_delete_and_regenerate_chat_messages_rewind_following_history(self) -> None:
        from src.core.manga_insight.character_studio.service import CharacterStudioService

        class FakeVlmClient:
            def __init__(self) -> None:
                self.responses = [
                    "第一轮回复",
                    "第二轮回复",
                    "重生成后的第一轮回复",
                    "第二轮重新开始后的回复",
                ]

            async def generate_messages(self, messages, *, on_stream_chunk=None):
                return self.responses.pop(0)

            async def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            insight_dir = os.path.join(temp_dir, "insight")

            with mock.patch(
                "src.core.manga_insight.storage.get_insight_storage_path",
                return_value=insight_dir,
            ):
                service = CharacterStudioService("book-demo")
                document = _demo_document()
                asyncio.run(service.store.save_document(document))

                state = asyncio.run(service.get_chat_state(document["id"]))
                session_id = state["active_session"]["session_id"]
                fake_client = FakeVlmClient()

                with mock.patch.object(service, "_create_vlm_client", return_value=fake_client):
                    first_round = asyncio.run(
                        service.send_chat_message(
                            document["id"],
                            session_id=session_id,
                            content="第一轮用户消息",
                        )
                    )
                    second_round = asyncio.run(
                        service.send_chat_message(
                            document["id"],
                            session_id=session_id,
                            content="第二轮用户消息",
                        )
                    )

                    first_user_id = first_round["session"]["messages"][1]["message_id"]
                    first_assistant_id = first_round["session"]["messages"][2]["message_id"]
                    second_user_id = second_round["session"]["messages"][3]["message_id"]

                    edited = asyncio.run(
                        service.edit_chat_message(
                            document["id"],
                            session_id=session_id,
                            message_id=first_user_id,
                            new_content="已编辑的第一轮用户消息",
                        )
                    )
                    self.assertEqual(
                        [item["role"] for item in edited["session"]["messages"]],
                        ["assistant", "user"],
                    )

                    regenerated = asyncio.run(
                        service.regenerate_chat_message(
                            document["id"],
                            session_id=session_id,
                            anchor_message_id=first_user_id,
                        )
                    )
                    self.assertEqual(
                        [item["content"] for item in regenerated["session"]["messages"]],
                        [
                            document["coreMessages"]["first_message"],
                            "已编辑的第一轮用户消息",
                            "重生成后的第一轮回复",
                        ],
                    )

                    replayed = asyncio.run(
                        service.send_chat_message(
                            document["id"],
                            session_id=session_id,
                            content="第二轮重新开始",
                        )
                    )
                    replayed_second_user_id = replayed["session"]["messages"][-2]["message_id"]
                    deleted = asyncio.run(
                        service.delete_chat_message(
                            document["id"],
                            session_id=session_id,
                            message_id=replayed_second_user_id,
                        )
                    )

                self.assertEqual(
                    [item["content"] for item in replayed["session"]["messages"]][-2:],
                    ["第二轮重新开始", "第二轮重新开始后的回复"],
                )
                self.assertEqual(
                    [item["role"] for item in deleted["session"]["messages"]],
                    ["assistant", "user", "assistant"],
                )


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

    def test_character_studio_document_route_returns_document_only(self) -> None:
        with mock.patch("isolated_character_studio_pkg.character_studio_routes.CharacterStudioService") as service_cls:
            service = service_cls.return_value
            service.store.load_document = mock.AsyncMock(return_value=_demo_document())
            response = self.client.get("/api/manga-insight/book-demo/character-studio/documents/doc_alpha")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["document"]["id"], "doc_alpha")

    def test_character_studio_chat_state_route_exists(self) -> None:
        with mock.patch("isolated_character_studio_pkg.character_studio_routes.CharacterStudioService") as service_cls:
            service_cls.return_value.get_chat_state = mock.AsyncMock(return_value={
                "doc_id": "doc_alpha",
                "active_session": {
                    "session_id": "chat_alpha",
                    "messages": [{"role": "assistant", "content": "我是测试角色，我们先谈正事。"}],
                },
                "archived_sessions": [],
                "available_greetings": [],
            })
            response = self.client.get("/api/manga-insight/book-demo/character-studio/documents/doc_alpha/chat")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["doc_id"], "doc_alpha")
        self.assertEqual(payload["active_session"]["session_id"], "chat_alpha")

    def test_character_studio_create_chat_session_route_exists(self) -> None:
        with mock.patch("isolated_character_studio_pkg.character_studio_routes.CharacterStudioService") as service_cls:
            service_cls.return_value.create_new_chat_session = mock.AsyncMock(return_value={
                "doc_id": "doc_alpha",
                "active_session": {
                    "session_id": "chat_beta",
                    "messages": [{"role": "assistant", "content": "新的开场白"}],
                },
                "archived_sessions": [{"session_id": "chat_alpha"}],
                "available_greetings": [],
            })
            response = self.client.post(
                "/api/manga-insight/book-demo/character-studio/documents/doc_alpha/chat/sessions",
                json={"greeting_id": "alternate_1"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["active_session"]["session_id"], "chat_beta")


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
