import io
import json
import os
import sys
import tempfile
import hashlib
import types
import unittest
import zipfile
import importlib.util
import importlib.machinery
from unittest import mock

from flask import Flask


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _install_dependency_stubs() -> None:
    if "cv2" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
        sys.modules["cv2"] = cv2_stub

    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")

        class _OpenAI:
            def __init__(self, *args, **kwargs):
                pass

        openai_stub.OpenAI = _OpenAI
        openai_stub.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
        sys.modules["openai"] = openai_stub

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        sys.modules["torch"] = torch_stub


def tearDownModule():
    for module_name in ("cv2", "openai", "torch"):
        sys.modules.pop(module_name, None)


def _write_plugin(
    plugin_dir: str,
    plugin_id: str,
    *,
    display_name: str | None = None,
    description: str = "test plugin",
) -> None:
    os.makedirs(plugin_dir, exist_ok=True)
    class_name = "".join(part.capitalize() for part in plugin_id.split("_")) or "TempPlugin"
    with open(os.path.join(plugin_dir, "__init__.py"), "w", encoding="utf-8") as handle:
        handle.write(f"from .plugin import {class_name}\n")
    with open(os.path.join(plugin_dir, "plugin.py"), "w", encoding="utf-8") as handle:
        handle.write(
            "from src.plugins.base import PluginBase\n\n"
            f"class {class_name}(PluginBase):\n"
            f"    plugin_id = {plugin_id!r}\n"
            f"    display_name = {(display_name or plugin_id)!r}\n"
            "    plugin_version = '1.0.0'\n"
            "    plugin_author = 'Tests'\n"
            f"    plugin_description = {description!r}\n"
            "    default_enabled = False\n"
            "    supported_steps = ('ocr',)\n"
            "    supported_modes = ('standard',)\n"
            "    def after_ocr(self, context, result):\n"
            "        return result\n"
        )
    os.makedirs(os.path.join(plugin_dir, "__pycache__"), exist_ok=True)
    with open(os.path.join(plugin_dir, "__pycache__", "plugin.cpython-312.pyc"), "wb") as handle:
        handle.write(b"compiled")


class PluginPackageIoTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_dependency_stubs()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.app_root = self.temp_dir.name
        self.plugins_root = os.path.join(self.app_root, "plugins")
        os.makedirs(self.plugins_root, exist_ok=True)

        self._manager_patcher = mock.patch("src.plugins.manager.get_app_root", return_value=self.app_root)
        self._manager_patcher.start()
        self.addCleanup(self._manager_patcher.stop)

        from src.plugins.manager import PluginManager

        self.PluginManager = PluginManager

    def test_export_plugin_bundle_contains_manifest_and_source_without_cache_files(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "sample_plugin")
        _write_plugin(plugin_dir, "sample_plugin", display_name="Sample Plugin")

        manager = self.PluginManager()
        manager.discover_and_load_plugins()

        bundle_bytes, filename = manager.export_plugin_bundle("sample_plugin")

        self.assertEqual(filename, "sample_plugin.zip")
        with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as archive:
            names = sorted(archive.namelist())
            self.assertIn("manifest.json", names)
            self.assertIn("sample_plugin/__init__.py", names)
            self.assertIn("sample_plugin/plugin.py", names)
            self.assertNotIn("sample_plugin/__pycache__/plugin.cpython-312.pyc", names)

            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest["plugin_id"], "sample_plugin")
            self.assertEqual(manifest["display_name"], "Sample Plugin")
            self.assertEqual(manifest["package_version"], 1)
            self.assertIn("files", manifest)
            file_names = [entry["path"] for entry in manifest["files"]]
            self.assertEqual(sorted(file_names), ["sample_plugin/__init__.py", "sample_plugin/plugin.py"])

    def test_import_plugin_bundle_installs_new_plugin_disabled_by_default(self) -> None:
        source_dir = os.path.join(self.plugins_root, "source_plugin")
        _write_plugin(source_dir, "source_plugin", display_name="Source Plugin")

        manager = self.PluginManager()
        manager.discover_and_load_plugins()
        bundle_bytes, _ = manager.export_plugin_bundle("source_plugin")

        manager.remove_plugin("source_plugin")
        shutil_path = os.path.join(self.plugins_root, "source_plugin")
        if os.path.exists(shutil_path):
            import shutil
            shutil.rmtree(shutil_path)

        result = manager.import_plugin_bundle(bundle_bytes, replace=False)

        self.assertTrue(result["success"])
        self.assertEqual(result["plugin"]["id"], "source_plugin")
        imported_plugin = manager.get_plugin("source_plugin")
        self.assertIsNotNone(imported_plugin)
        self.assertFalse(imported_plugin.is_enabled())
        self.assertFalse(manager.plugin_default_states["source_plugin"])

    def test_import_plugin_bundle_rejects_existing_plugin_without_replace(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "conflict_plugin")
        _write_plugin(plugin_dir, "conflict_plugin", display_name="Conflict Plugin")

        manager = self.PluginManager()
        manager.discover_and_load_plugins()
        bundle_bytes, _ = manager.export_plugin_bundle("conflict_plugin")

        from src.shared.exceptions import PluginException

        with self.assertRaises(PluginException) as ctx:
            manager.import_plugin_bundle(bundle_bytes, replace=False)

        self.assertIn("已存在", str(ctx.exception))
        self.assertEqual(ctx.exception.details.get("plugin_id"), "conflict_plugin")

    def test_import_plugin_bundle_replaces_existing_plugin_and_preserves_default_state(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "replace_plugin")
        _write_plugin(plugin_dir, "replace_plugin", display_name="Replace Plugin", description="old")

        manager = self.PluginManager()
        manager.discover_and_load_plugins()
        manager.set_plugin_default_state("replace_plugin", True)
        manager.enable_plugin("replace_plugin")

        with open(os.path.join(plugin_dir, "plugin.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "from src.plugins.base import PluginBase\n\n"
                "class ReplacePlugin(PluginBase):\n"
                "    plugin_id = 'replace_plugin'\n"
                "    display_name = 'Replace Plugin Reloaded'\n"
                "    plugin_version = '2.0.0'\n"
                "    plugin_author = 'Tests'\n"
                "    plugin_description = 'new'\n"
                "    default_enabled = False\n"
                "    supported_steps = ('ocr',)\n"
                "    supported_modes = ('standard',)\n"
                "    def after_ocr(self, context, result):\n"
                "        return result\n"
            )

        bundle_bytes, _ = manager.export_plugin_bundle("replace_plugin")

        with open(os.path.join(plugin_dir, "plugin.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "from src.plugins.base import PluginBase\n\n"
                "class ReplacePlugin(PluginBase):\n"
                "    plugin_id = 'replace_plugin'\n"
                "    display_name = 'Broken Local Plugin'\n"
                "    plugin_version = '0.1.0'\n"
                "    plugin_author = 'Tests'\n"
                "    plugin_description = 'broken local'\n"
                "    default_enabled = False\n"
                "    supported_steps = ('ocr',)\n"
                "    supported_modes = ('standard',)\n"
                "    def after_ocr(self, context, result):\n"
                "        return result\n"
            )
        manager.refresh_plugins()

        result = manager.import_plugin_bundle(bundle_bytes, replace=True)

        self.assertTrue(result["success"])
        self.assertEqual(result["plugin"]["display_name"], "Replace Plugin Reloaded")
        self.assertTrue(manager.plugin_default_states["replace_plugin"])
        self.assertTrue(manager.get_plugin("replace_plugin").is_enabled())

    def test_import_plugin_bundle_rejects_zip_slip_entries(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps({
                "package_version": 1,
                "plugin_id": "evil_plugin",
                "display_name": "Evil Plugin",
                "source_directory": "evil_plugin",
                "files": [{"path": "../evil.py", "sha256": "ignored"}],
            }))
            archive.writestr("../evil.py", "print('bad')")

        manager = self.PluginManager()

        from src.shared.exceptions import PluginException

        with self.assertRaises(PluginException) as ctx:
            manager.import_plugin_bundle(buffer.getvalue(), replace=False)

        self.assertIn("非法", str(ctx.exception))

    def test_import_plugin_bundle_rejects_missing_manifest(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sample_plugin/__init__.py", "from .plugin import SamplePlugin\n")
            archive.writestr("sample_plugin/plugin.py", "print('missing manifest')\n")

        manager = self.PluginManager()
        from src.shared.exceptions import PluginException

        with self.assertRaises(PluginException) as ctx:
            manager.import_plugin_bundle(buffer.getvalue(), replace=False)

        self.assertIn("manifest", str(ctx.exception).lower())

    def test_import_plugin_bundle_rejects_unexpected_file_not_declared_in_manifest(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            plugin_source = (
                "from src.plugins.base import PluginBase\n\n"
                "class SamplePlugin(PluginBase):\n"
                "    plugin_id = 'sample_plugin'\n"
                "    display_name = 'Sample Plugin'\n"
                "    plugin_version = '1.0.0'\n"
                "    plugin_author = 'Tests'\n"
                "    plugin_description = 'sample'\n"
                "    default_enabled = False\n"
                "    supported_steps = ('ocr',)\n"
                "    supported_modes = ('standard',)\n"
            )
            init_content = "from .plugin import SamplePlugin\n"
            archive.writestr(
                "manifest.json",
                json.dumps({
                    "package_version": 1,
                    "plugin_id": "sample_plugin",
                    "display_name": "Sample Plugin",
                    "source_directory": "sample_plugin",
                    "files": [
                        {
                            "path": "sample_plugin/__init__.py",
                            "sha256": hashlib.sha256(init_content.encode("utf-8")).hexdigest(),
                        },
                        {
                            "path": "sample_plugin/plugin.py",
                            "sha256": hashlib.sha256(plugin_source.encode("utf-8")).hexdigest(),
                        },
                    ],
                }),
            )
            archive.writestr("sample_plugin/__init__.py", init_content)
            archive.writestr("sample_plugin/plugin.py", plugin_source)
            archive.writestr("sample_plugin/extra.py", "print('unexpected')")

        manager = self.PluginManager()
        from src.shared.exceptions import PluginException

        with self.assertRaises(PluginException) as ctx:
            manager.import_plugin_bundle(buffer.getvalue(), replace=False)

        self.assertIn("文件清单", str(ctx.exception))

    def test_import_plugin_bundle_rolls_back_when_replacement_validation_fails(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "rollback_plugin")
        _write_plugin(plugin_dir, "rollback_plugin", display_name="Rollback Plugin", description="stable")

        manager = self.PluginManager()
        manager.discover_and_load_plugins()
        manager.set_plugin_default_state("rollback_plugin", True)
        manager.enable_plugin("rollback_plugin")

        original_display_name = manager.get_plugin("rollback_plugin").display_name

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            bad_plugin_source = (
                "from src.plugins.base import PluginBase\n\n"
                "class RollbackPlugin(PluginBase):\n"
                "    plugin_id = 'rollback_plugin'\n"
                "    display_name = 'Broken Replacement'\n"
                "    supported_steps = ('not_a_step',)\n"
                "    supported_modes = ('standard',)\n"
            )
            archive.writestr(
                "manifest.json",
                json.dumps({
                    "package_version": 1,
                    "plugin_id": "rollback_plugin",
                    "display_name": "Broken Replacement",
                    "files": [],
                }),
            )
            archive.writestr("rollback_plugin/__init__.py", "from .plugin import RollbackPlugin\n")
            archive.writestr("rollback_plugin/plugin.py", bad_plugin_source)

        from src.shared.exceptions import PluginException

        with self.assertRaises(PluginException):
            manager.import_plugin_bundle(buffer.getvalue(), replace=True)

        restored = manager.get_plugin("rollback_plugin")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.display_name, original_display_name)
        self.assertTrue(restored.is_enabled())
        self.assertTrue(manager.plugin_default_states["rollback_plugin"])


class PluginImportExportApiTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_dependency_stubs()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.app_root = self.temp_dir.name
        self.plugins_root = os.path.join(self.app_root, "plugins")
        os.makedirs(self.plugins_root, exist_ok=True)

        self._manager_patcher = mock.patch("src.plugins.manager.get_app_root", return_value=self.app_root)
        self._manager_patcher.start()
        self.addCleanup(self._manager_patcher.stop)

        package_dir = os.path.join(PROJECT_ROOT, "src", "app", "api", "system")
        package_name = "isolated_plugin_system"
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [package_dir]
        from flask import Blueprint
        package_module.system_bp = Blueprint("isolated_system_api", __name__, url_prefix="/api")
        sys.modules[package_name] = package_module
        self.addCleanup(lambda: sys.modules.pop(package_name, None))

        module_path = os.path.join(package_dir, "plugins.py")
        spec = importlib.util.spec_from_file_location(f"{package_name}.plugins", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.plugins_api_module = module
        self.addCleanup(lambda: sys.modules.pop(spec.name, None))

        self.app = Flask(__name__)
        self.app.register_blueprint(package_module.system_bp)
        self.client = self.app.test_client()

        from src.plugins.manager import PluginManager

        self.PluginManager = PluginManager
        self.manager = self.PluginManager()
        self._singleton_patcher = mock.patch.object(self.plugins_api_module, "get_plugin_manager", return_value=self.manager)
        self._singleton_patcher.start()
        self.addCleanup(self._singleton_patcher.stop)

    def test_export_plugin_route_returns_zip_attachment(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "api_export_plugin")
        _write_plugin(plugin_dir, "api_export_plugin", display_name="API Export Plugin")
        self.manager.discover_and_load_plugins()

        response = self.client.get("/api/plugins/api_export_plugin/export")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        disposition = response.headers.get("Content-Disposition", "")
        self.assertIn("attachment", disposition)
        self.assertIn("api_export_plugin.zip", disposition)
        with zipfile.ZipFile(io.BytesIO(response.data), "r") as archive:
            self.assertIn("manifest.json", archive.namelist())

    def test_import_plugin_route_returns_conflict_409_without_replace(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "api_conflict_plugin")
        _write_plugin(plugin_dir, "api_conflict_plugin", display_name="API Conflict Plugin")
        self.manager.discover_and_load_plugins()
        bundle_bytes, _ = self.manager.export_plugin_bundle("api_conflict_plugin")

        response = self.client.post(
            "/api/plugins/import",
            data={"file": (io.BytesIO(bundle_bytes), "api_conflict_plugin.zip")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["details"]["plugin_id"], "api_conflict_plugin")

    def test_import_plugin_route_allows_replace_when_flag_is_true(self) -> None:
        plugin_dir = os.path.join(self.plugins_root, "api_replace_plugin")
        _write_plugin(plugin_dir, "api_replace_plugin", display_name="Old Plugin", description="old")
        self.manager.discover_and_load_plugins()

        with open(os.path.join(plugin_dir, "plugin.py"), "w", encoding="utf-8") as handle:
            handle.write(
                "from src.plugins.base import PluginBase\n\n"
                "class ApiReplacePlugin(PluginBase):\n"
                "    plugin_id = 'api_replace_plugin'\n"
                "    display_name = 'New Plugin'\n"
                "    plugin_version = '2.0.0'\n"
                "    plugin_author = 'Tests'\n"
                "    plugin_description = 'new'\n"
                "    default_enabled = False\n"
                "    supported_steps = ('ocr',)\n"
                "    supported_modes = ('standard',)\n"
                "    def after_ocr(self, context, result):\n"
                "        return result\n"
            )
        bundle_bytes, _ = self.manager.export_plugin_bundle("api_replace_plugin")

        response = self.client.post(
            "/api/plugins/import",
            data={
                "file": (io.BytesIO(bundle_bytes), "api_replace_plugin.zip"),
                "replace": "true",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["plugin"]["display_name"], "New Plugin")
