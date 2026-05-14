import os
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class PathHelpersRuntimePathTests(unittest.TestCase):
    def test_get_app_root_uses_repo_root_in_development(self) -> None:
        from src.shared import path_helpers

        expected_root = os.path.abspath(os.path.join(os.path.dirname(path_helpers.__file__), "..", ".."))

        with mock.patch.object(sys, "frozen", False, create=True):
            self.assertEqual(path_helpers.get_app_root(), expected_root)

    def test_get_app_root_uses_executable_directory_in_packaged_mode(self) -> None:
        from src.shared import path_helpers

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_executable = os.path.join(temp_dir, "Saber-Translator.exe")
            with mock.patch.object(sys, "frozen", True, create=True):
                with mock.patch.object(sys, "executable", fake_executable):
                    with mock.patch.object(sys, "_MEIPASS", os.path.join(temp_dir, "_internal"), create=True):
                        self.assertEqual(path_helpers.get_app_root(), temp_dir)


class ConfigLoaderRuntimePathTests(unittest.TestCase):
    def test_get_config_path_uses_app_root_config_in_packaged_mode(self) -> None:
        from src.shared import config_loader

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_executable = os.path.join(temp_dir, "Saber-Translator.exe")
            with mock.patch.object(sys, "frozen", True, create=True):
                with mock.patch.object(sys, "executable", fake_executable):
                    with mock.patch.object(sys, "_MEIPASS", os.path.join(temp_dir, "_internal"), create=True):
                        with mock.patch.object(config_loader, "CONFIG_DIR", None):
                            self.assertEqual(
                                config_loader.get_config_path("user_settings.json"),
                                os.path.join(temp_dir, "config", "user_settings.json"),
                            )


class PluginManagerRuntimePathTests(unittest.TestCase):
    def test_plugin_manager_defaults_to_single_app_root_plugins_directory(self) -> None:
        from src.plugins.manager import PluginManager

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("src.plugins.manager.get_app_root", return_value=temp_dir):
                manager = PluginManager()

        self.assertEqual(manager.plugin_dirs, [os.path.join(temp_dir, "plugins")])

    def test_plugin_manager_stores_plugin_persistence_under_app_root_config(self) -> None:
        from src.plugins.manager import PluginManager

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("src.plugins.manager.get_app_root", return_value=temp_dir):
                manager = PluginManager(plugin_dirs=[])

        self.assertEqual(
            manager.plugin_config_dir,
            os.path.join(temp_dir, "config", "plugin_configs"),
        )
        self.assertEqual(
            manager.plugin_default_states_path,
            os.path.join(temp_dir, "config", "plugin_default_states.json"),
        )

    def test_plugin_manager_treats_empty_default_states_file_as_empty_mapping(self) -> None:
        from src.plugins.manager import PluginManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            default_states_path = os.path.join(config_dir, "plugin_default_states.json")
            with open(default_states_path, "w", encoding="utf-8") as handle:
                handle.write("")

            with mock.patch("src.plugins.manager.get_app_root", return_value=temp_dir):
                with mock.patch("src.plugins.manager.logger.error") as logger_error:
                    manager = PluginManager(plugin_dirs=[])

        self.assertEqual(manager.plugin_default_states, {})
        logger_error.assert_not_called()


class PyInstallerSpecPluginLayoutTests(unittest.TestCase):
    def test_spec_keeps_plugins_as_top_level_collect_content(self) -> None:
        spec_path = os.path.join(PROJECT_ROOT, "app.spec")
        with open(spec_path, "r", encoding="utf-8") as handle:
            spec_text = handle.read()

        self.assertNotIn("contents_directory='.'", spec_text)
        self.assertNotIn("datas.append((src_plugins_path", spec_text)
        self.assertNotIn("datas.append((plugins_path, 'plugins'))", spec_text)
        self.assertNotIn("os.path.join('src', 'plugins')", spec_text)
        self.assertIn("shutil.copytree(plugins_path, bundle_plugins_path)", spec_text)

    def test_spec_packages_ai_provider_manifest_next_to_src_shared_module(self) -> None:
        spec_path = os.path.join(PROJECT_ROOT, "app.spec")
        with open(spec_path, "r", encoding="utf-8") as handle:
            spec_text = handle.read()

        self.assertIn("ai_provider_manifest.json", spec_text)
        self.assertIn("os.path.join('src', 'shared')", spec_text)

    def test_spec_packages_plugin_agent_skill_markdown(self) -> None:
        spec_path = os.path.join(PROJECT_ROOT, "app.spec")
        with open(spec_path, "r", encoding="utf-8") as handle:
            spec_text = handle.read()

        self.assertIn("plugin_builder_skill.md", spec_text)
        self.assertIn("os.path.join('src', 'core', 'plugin_agent')", spec_text)


if __name__ == "__main__":
    unittest.main()
