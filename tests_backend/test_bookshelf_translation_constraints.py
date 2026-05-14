import copy
import importlib.util
import json
import os
import sys
import types
import shutil
import tempfile
import unittest
from unittest import mock

from flask import Flask
from src.core.bookshelf_manager import DEFAULT_AUTO_GLOSSARY_PROMPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BookshelfTranslationConstraintsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="saber-bookshelf-constraints-")
        self._original_modules = {}
        self._stubbed_module_names = [
            "yaml",
            "openai",
            "cv2",
            "torch",
            "isolated_bookshelf_pkg",
            "isolated_bookshelf_pkg.bookshelf_api",
        ]
        for module_name in self._stubbed_module_names:
            self._original_modules[module_name] = sys.modules.get(module_name)

        if "yaml" not in sys.modules:
            yaml_stub = types.ModuleType("yaml")
            yaml_stub.safe_load = lambda *_args, **_kwargs: {}
            yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
            yaml_stub.dump = lambda *_args, **_kwargs: ""
            sys.modules["yaml"] = yaml_stub

        if "openai" not in sys.modules:
            openai_stub = types.ModuleType("openai")
            openai_stub.OpenAI = type("OpenAI", (), {"__init__": lambda self, *args, **kwargs: None})
            sys.modules["openai"] = openai_stub

        if "cv2" not in sys.modules:
            sys.modules["cv2"] = types.ModuleType("cv2")

        if "torch" not in sys.modules:
            torch_stub = types.ModuleType("torch")
            torch_stub.cuda = types.SimpleNamespace(
                is_available=lambda: False,
                empty_cache=lambda: None,
                synchronize=lambda: None,
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
                get_device_name=lambda _index=0: "stub-gpu",
                get_device_properties=lambda _index=0: types.SimpleNamespace(total_memory=0),
            )
            torch_stub.hub = types.SimpleNamespace(set_dir=lambda *_args, **_kwargs: None)
            sys.modules["torch"] = torch_stub

        self.resource_path_patcher = mock.patch(
            "src.shared.path_helpers.resource_path",
            side_effect=self._resource_path,
        )
        self.bookshelf_resource_path_patcher = mock.patch(
            "src.core.bookshelf_manager.resource_path",
            side_effect=self._resource_path,
        )
        self.resource_path_patcher.start()
        self.bookshelf_resource_path_patcher.start()

        from src.core import bookshelf_manager

        self.bookshelf_manager = bookshelf_manager
        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api")
        module_path = os.path.join(package_dir, "bookshelf_api.py")
        spec = importlib.util.spec_from_file_location(
            "isolated_bookshelf_pkg.bookshelf_api",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["isolated_bookshelf_pkg.bookshelf_api"] = module
        spec.loader.exec_module(module)

        self.app = Flask(__name__)
        self.app.register_blueprint(module.bookshelf_bp)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.resource_path_patcher.stop()
        self.bookshelf_resource_path_patcher.stop()
        for module_name in self._stubbed_module_names:
            original = self._original_modules.get(module_name)
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _resource_path(self, relative_path: str) -> str:
        normalized = relative_path.replace("/", os.sep).replace("\\", os.sep)
        return os.path.join(self._tmpdir, normalized)

    def _read_book_meta(self, book_id: str) -> dict:
        book_meta_path = os.path.join(
            self._tmpdir,
            "data",
            "bookshelf",
            book_id,
            "book_meta.json",
        )
        with open(book_meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_create_book_initializes_empty_translation_constraints(self) -> None:
        book = self.bookshelf_manager.create_book("测试书籍", None, ["tag-a"])

        self.assertIsNotNone(book)
        assert book is not None
        self.assertIn("translation_constraints", book)
        self.assertEqual(
            book["translation_constraints"],
            {
                "glossary": {"enabled": False, "autoExtractEnabled": False, "autoExtractPrompt": DEFAULT_AUTO_GLOSSARY_PROMPT, "entries": []},
                "non_translate": {"enabled": False, "entries": []},
            },
        )

        persisted = self._read_book_meta(book["id"])
        self.assertEqual(
            persisted["translation_constraints"],
            {
                "glossary": {"enabled": False, "autoExtractEnabled": False, "autoExtractPrompt": DEFAULT_AUTO_GLOSSARY_PROMPT, "entries": []},
                "non_translate": {"enabled": False, "entries": []},
            },
        )

    def test_get_book_backfills_translation_constraints_when_missing(self) -> None:
        book = self.bookshelf_manager.create_book("测试书籍", None, [])
        assert book is not None

        book_meta_path = os.path.join(
            self._tmpdir,
            "data",
            "bookshelf",
            book["id"],
            "book_meta.json",
        )
        with open(book_meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.pop("translation_constraints", None)
        with open(book_meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        loaded = self.bookshelf_manager.get_book(book["id"])

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(
            loaded["translation_constraints"],
            {
                "glossary": {"enabled": False, "autoExtractEnabled": False, "autoExtractPrompt": DEFAULT_AUTO_GLOSSARY_PROMPT, "entries": []},
                "non_translate": {"enabled": False, "entries": []},
            },
        )
        persisted = self._read_book_meta(book["id"])
        self.assertIn("translation_constraints", persisted)

    def test_update_book_persists_translation_constraints_without_other_updates(self) -> None:
        book = self.bookshelf_manager.create_book("测试书籍", None, ["tag-a"])
        assert book is not None

        constraints = {
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": True,
                "autoExtractPrompt": "提取这本漫画的人名和专有名词",
                "entries": [
                    {
                        "source": "Alice",
                        "target": "爱丽丝",
                        "note": "主角",
                        "matchMode": "text",
                    }
                ],
            },
            "non_translate": {
                "enabled": True,
                "entries": [
                    {
                        "pattern": "<keep>",
                        "note": "占位符",
                        "matchMode": "text",
                    }
                ],
            },
        }

        updated = self.bookshelf_manager.update_book(
            book["id"],
            translation_constraints=copy.deepcopy(constraints),
        )

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated["title"], "测试书籍")
        self.assertEqual(updated["tags"], ["tag-a"])
        self.assertEqual(updated["translation_constraints"], constraints)

        persisted = self._read_book_meta(book["id"])
        self.assertEqual(persisted["translation_constraints"], constraints)

    def test_update_book_filters_blank_constraint_rows(self) -> None:
        book = self.bookshelf_manager.create_book("测试书籍", None, [])
        assert book is not None

        updated = self.bookshelf_manager.update_book(
            book["id"],
            translation_constraints={
                "glossary": {
                    "enabled": True,
                    "autoExtractEnabled": True,
                    "autoExtractPrompt": "提取这本漫画的人名和专有名词",
                    "entries": [
                        {"source": "Alice", "target": "爱丽丝", "note": "", "matchMode": "text"},
                        {"source": " ", "target": " ", "note": "", "matchMode": "regex"},
                    ],
                },
                "non_translate": {
                    "enabled": True,
                    "entries": [
                        {"pattern": "<keep>", "note": "", "matchMode": "text"},
                        {"pattern": "   ", "note": "", "matchMode": "regex"},
                    ],
                },
            },
        )

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(
            updated["translation_constraints"],
            {
                "glossary": {
                    "enabled": True,
                    "autoExtractEnabled": True,
                    "autoExtractPrompt": "提取这本漫画的人名和专有名词",
                    "entries": [
                        {"source": "Alice", "target": "爱丽丝", "note": "", "matchMode": "text"},
                    ],
                },
                "non_translate": {
                    "enabled": True,
                    "entries": [
                        {"pattern": "<keep>", "note": "", "matchMode": "text"},
                    ],
                },
            },
        )

    def test_bookshelf_update_api_rejects_invalid_regex_constraints(self) -> None:
        book = self.bookshelf_manager.create_book("测试书籍", None, ["tag-a"])
        assert book is not None

        response = self.client.put(
            f"/api/bookshelf/books/{book['id']}",
            json={
                "translation_constraints": {
                    "glossary": {
                        "enabled": True,
                        "entries": [
                            {"source": "(", "target": "爱丽丝", "note": "", "matchMode": "regex"},
                        ],
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("正则无效", payload["error"])


if __name__ == "__main__":
    unittest.main()
