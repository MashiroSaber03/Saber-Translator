import importlib
import importlib.util
import inspect
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
from typing import Any, Dict, List, Optional, Tuple

from src.shared.config_loader import get_config_path, load_json_config, save_json_config
from src.shared.exceptions import PluginException
from src.shared.path_helpers import resource_path

from .base import PluginBase
from .hooks import (
    HOOK_METHOD_TO_STEP_PHASE,
    STEP_HOOK_METHODS,
    PluginContext,
    normalize_plugin_mode,
    normalize_plugin_step,
)

logger = logging.getLogger("PluginManager")

PLUGIN_DEFAULT_STATES_FILE = "plugin_default_states.json"


class PluginManager:
    """
    负责发现、加载、管理并执行 v2 插件。
    """

    def __init__(self, app=None, plugin_dirs=None):
        self.app = app
        self.plugins: Dict[str, PluginBase] = {}
        self.hooks: Dict[str, List[Tuple[int, str, Any]]] = {
            method_name: [] for method_name in HOOK_METHOD_TO_STEP_PHASE
        }
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self.plugin_sources: Dict[str, str] = {}
        self.plugin_module_names: Dict[str, List[str]] = {}
        self.plugin_dirs = plugin_dirs or [
            resource_path("src/plugins"),
            resource_path("plugins"),
        ]
        self.plugin_config_dir = get_config_path("plugin_configs")
        os.makedirs(self.plugin_config_dir, exist_ok=True)
        os.makedirs(resource_path("plugins"), exist_ok=True)
        self.plugin_default_states = self._load_plugin_default_states()
        logger.debug("插件管理器初始化，扫描目录: %s", self.plugin_dirs)

    def reset_for_testing(self, *, clear_defaults: bool = True) -> None:
        for plugin in self.plugins.values():
            try:
                plugin.disable()
            except Exception:
                logger.debug("测试重置时禁用插件失败: %s", plugin.plugin_id, exc_info=True)
        self.plugins.clear()
        self.plugin_metadata.clear()
        self.plugin_sources.clear()
        self.plugin_module_names.clear()
        self.hooks = {method_name: [] for method_name in HOOK_METHOD_TO_STEP_PHASE}
        if clear_defaults:
            self.plugin_default_states = {}

    def _load_plugin_default_states(self) -> Dict[str, bool]:
        loaded_states = load_json_config(PLUGIN_DEFAULT_STATES_FILE, default_value={})
        if not isinstance(loaded_states, dict):
            return {}
        normalized_states: Dict[str, bool] = {}
        for plugin_id, enabled in loaded_states.items():
            if plugin_id:
                normalized_states[str(plugin_id)] = bool(enabled)
        logger.debug("已加载插件默认启用状态: %s", normalized_states)
        return normalized_states

    def save_plugin_default_states(self) -> bool:
        success = save_json_config(PLUGIN_DEFAULT_STATES_FILE, self.plugin_default_states)
        if success:
            logger.info("插件默认启用状态已保存。")
        else:
            logger.error("保存插件默认启用状态失败。")
        return success

    def set_plugin_default_state(self, plugin_id: str, enabled: bool) -> bool:
        if plugin_id not in self.plugins:
            logger.warning("尝试设置未加载插件 '%s' 的默认状态。", plugin_id)
            return False
        self.plugin_default_states[plugin_id] = bool(enabled)
        self._refresh_metadata(plugin_id)
        return self.save_plugin_default_states()

    def discover_and_load_plugins(self) -> None:
        logger.debug("开始发现和加载插件...")
        needs_saving_defaults = False
        loaded_count = 0

        for plugin_name, package_path in self._find_potential_plugins():
            try:
                plugin_module = self._load_plugin_module(package_path, plugin_name)
                for attr_name in dir(plugin_module):
                    attr = getattr(plugin_module, attr_name)
                    if inspect.isclass(attr) and issubclass(attr, PluginBase) and attr is not PluginBase:
                        plugin_loaded, default_added = self._load_plugin_class(attr, source_path=package_path)
                        if plugin_loaded:
                            loaded_count += 1
                        if default_added:
                            needs_saving_defaults = True
                        break
            except Exception as exc:
                logger.error("加载插件目录 '%s' 失败: %s", plugin_name, exc, exc_info=True)

        logger.info("插件加载完成，共加载 %s 个插件。", loaded_count)
        if needs_saving_defaults:
            self.save_plugin_default_states()

    def _find_potential_plugins(self) -> List[Tuple[str, str]]:
        potential_plugins: List[Tuple[str, str]] = []
        for plugin_dir in self.plugin_dirs:
            if not os.path.isdir(plugin_dir):
                logger.warning("插件目录不存在或不是目录: %s", plugin_dir)
                continue
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "__init__.py")) and os.path.exists(os.path.join(item_path, "plugin.py")):
                    potential_plugins.append((item, item_path))
        return potential_plugins

    def _get_module_import_path(self, package_path: str, plugin_name: str) -> Optional[str]:
        normalized_path = package_path.replace("\\", "/")
        if "src/plugins" in normalized_path:
            return f"src.plugins.{plugin_name}.plugin"
        logger.warning("无法确定插件 '%s' 的导入路径: %s", plugin_name, package_path)
        return None

    def _load_plugin_module(self, package_path: str, plugin_name: str):
        normalized_path = package_path.replace("\\", "/")
        if "src/plugins" not in normalized_path:
            return self._load_external_plugin_module(package_path, plugin_name)
        module_import_path = self._get_module_import_path(package_path, plugin_name)
        if module_import_path:
            return importlib.import_module(module_import_path)
        return self._load_external_plugin_module(package_path, plugin_name)

    def _load_external_plugin_module(self, package_path: str, plugin_name: str):
        init_path = os.path.join(package_path, "__init__.py")
        plugin_path = os.path.join(package_path, "plugin.py")
        module_suffix = self._build_external_module_suffix(package_path, plugin_name)
        self._ensure_external_plugins_namespace()
        package_module_name = f"_saber_user_plugins.{module_suffix}"
        plugin_module_name = f"{package_module_name}.plugin"

        package_module = sys.modules.get(package_module_name)
        if package_module is None:
            package_spec = importlib.util.spec_from_file_location(
                package_module_name,
                init_path,
                submodule_search_locations=[package_path],
            )
            if package_spec is None or package_spec.loader is None:
                raise PluginException(f"无法加载插件包: {package_path}")
            package_module = importlib.util.module_from_spec(package_spec)
            sys.modules[package_module_name] = package_module
            package_spec.loader.exec_module(package_module)

        if plugin_module_name in sys.modules:
            return sys.modules[plugin_module_name]

        plugin_spec = importlib.util.spec_from_file_location(
            plugin_module_name,
            plugin_path,
        )
        if plugin_spec is None or plugin_spec.loader is None:
            raise PluginException(f"无法加载插件模块: {plugin_path}")
        plugin_module = importlib.util.module_from_spec(plugin_spec)
        sys.modules[plugin_module_name] = plugin_module
        plugin_spec.loader.exec_module(plugin_module)
        return plugin_module

    def _ensure_external_plugins_namespace(self) -> None:
        namespace_name = "_saber_user_plugins"
        if namespace_name in sys.modules:
            return
        namespace_spec = importlib.util.spec_from_loader(namespace_name, loader=None)
        namespace_module = importlib.util.module_from_spec(namespace_spec)
        namespace_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[namespace_name] = namespace_module

    def _build_external_module_suffix(self, package_path: str, plugin_name: str) -> str:
        normalized_name = "".join(
            character if character.isalnum() else "_"
            for character in plugin_name
        ).strip("_") or "plugin"
        package_digest = hashlib.sha1(
            os.path.abspath(package_path).encode("utf-8")
        ).hexdigest()[:12]
        return f"{normalized_name}_{package_digest}"

    def _load_plugin_class(self, plugin_class, *, source_path: str) -> Tuple[bool, bool]:
        try:
            plugin_instance = plugin_class(plugin_manager=self, app=self.app)
            default_added = plugin_instance.plugin_id not in self.plugin_default_states
            self.register_plugin_instance(plugin_instance, source_path=source_path)
            return True, default_added
        except Exception as exc:
            logger.error(
                "实例化或设置插件类 '%s' 失败: %s",
                plugin_class.__name__,
                exc,
                exc_info=True,
            )
            return False, False

    def register_plugin_instance(
        self,
        plugin_instance: PluginBase,
        *,
        source_path: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> PluginBase:
        plugin_instance.validate_metadata()
        plugin_id = plugin_instance.plugin_id

        if plugin_id in self.plugins:
            raise PluginException(f"插件 ID 冲突: '{plugin_id}' 已存在。")

        if plugin_instance.setup() is False:
            raise PluginException(f"插件 '{plugin_id}' 的 setup 方法返回 False。")

        self.plugins[plugin_id] = plugin_instance
        if source_path:
            self.plugin_sources[plugin_id] = source_path
        self.plugin_module_names[plugin_id] = self._collect_plugin_module_names(plugin_instance)

        config_data = self._load_plugin_config_file(plugin_id)
        plugin_instance.load_config(config_data)
        self._register_hooks(plugin_instance)

        if plugin_id not in self.plugin_default_states:
            self.plugin_default_states[plugin_id] = bool(plugin_instance.default_enabled)

        should_enable = (
            self.plugin_default_states.get(plugin_id, bool(plugin_instance.default_enabled))
            if enabled is None
            else bool(enabled)
        )

        if should_enable:
            plugin_instance.enable()
        else:
            plugin_instance.disable()

        self._refresh_metadata(plugin_id)
        logger.debug("插件 '%s' 注册成功。", plugin_id)
        return plugin_instance

    def _register_hooks(self, plugin_instance: PluginBase) -> None:
        for method_name in HOOK_METHOD_TO_STEP_PHASE:
            method = getattr(plugin_instance, method_name, None)
            base_method = getattr(PluginBase, method_name, None)
            if not callable(method) or base_method is None:
                continue
            if getattr(method, "__code__", None) is getattr(base_method, "__code__", None):
                continue
            self.hooks[method_name].append(
                (int(plugin_instance.priority), plugin_instance.plugin_id, method)
            )
            self.hooks[method_name].sort(key=lambda item: (item[0], item[1]))

    def unregister_plugin_hooks(self, plugin_id_to_remove: str) -> None:
        for method_name in self.hooks:
            self.hooks[method_name] = [
                entry for entry in self.hooks[method_name]
                if entry[1] != plugin_id_to_remove
            ]

    def _refresh_metadata(self, plugin_id: str) -> None:
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            self.plugin_metadata.pop(plugin_id, None)
            return
        metadata = plugin.get_metadata()
        metadata["enabled"] = plugin.is_enabled()
        self.plugin_metadata[plugin_id] = metadata

    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        return self.plugins.get(plugin_id)

    def get_plugin_records(self) -> List[Dict[str, Any]]:
        for plugin_id in list(self.plugins):
            self._refresh_metadata(plugin_id)
        return sorted(
            self.plugin_metadata.values(),
            key=lambda record: (str(record.get("display_name", "")), str(record.get("id", ""))),
        )

    def get_plugin_source_path(self, plugin_id: str) -> Optional[str]:
        return self.plugin_sources.get(plugin_id)

    def _collect_plugin_module_names(self, plugin_instance: PluginBase) -> List[str]:
        module_name = plugin_instance.__class__.__module__
        module_names = {module_name}
        package_name = module_name.rpartition(".")[0]
        while package_name:
            module_names.add(package_name)
            package_name = package_name.rpartition(".")[0]
        return sorted(module_names, key=len, reverse=True)

    def unload_plugin_modules(self, plugin_id: str) -> None:
        for module_name in self.plugin_module_names.get(plugin_id, []):
            sys.modules.pop(module_name, None)
        importlib.invalidate_caches()

    def _handle_remove_readonly(self, func, path, exc_info) -> None:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                try:
                    os.chmod(parent_dir, stat.S_IWRITE | stat.S_IREAD)
                except OSError:
                    pass
        func(path)

    def delete_plugin_directory(self, plugin_id: str) -> bool:
        plugin_path = self.get_plugin_source_path(plugin_id)
        if not plugin_path or not os.path.exists(plugin_path):
            return False

        plugin = self.get_plugin(plugin_id)
        if plugin:
            try:
                plugin.disable()
            except Exception:
                logger.debug("删除插件 '%s' 前禁用失败。", plugin_id, exc_info=True)

        self.unload_plugin_modules(plugin_id)
        shutil.rmtree(plugin_path, onerror=self._handle_remove_readonly)
        return True

    def enable_plugin(self, plugin_id: str) -> bool:
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False
        plugin.enable()
        self._refresh_metadata(plugin_id)
        return True

    def disable_plugin(self, plugin_id: str) -> bool:
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False
        plugin.disable()
        self._refresh_metadata(plugin_id)
        return True

    def remove_plugin(self, plugin_id: str) -> None:
        plugin = self.plugins.pop(plugin_id, None)
        if plugin:
            try:
                plugin.disable()
            except Exception:
                logger.debug("移除插件 '%s' 时禁用失败。", plugin_id, exc_info=True)
        self.unregister_plugin_hooks(plugin_id)
        self.plugin_metadata.pop(plugin_id, None)
        self.plugin_sources.pop(plugin_id, None)
        self.plugin_module_names.pop(plugin_id, None)
        self.plugin_default_states.pop(plugin_id, None)
        self._delete_plugin_config_file(plugin_id)

    def run_before_step(
        self,
        step: str,
        payload: Any,
        *,
        mode: str = "standard",
        route: str,
        scope: str = "image",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._run_step_phase(
            step,
            "before",
            payload,
            mode=mode,
            route=route,
            scope=scope,
            metadata=metadata,
        )

    def run_after_step(
        self,
        step: str,
        result: Any,
        *,
        mode: str = "standard",
        route: str,
        scope: str = "image",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._run_step_phase(
            step,
            "after",
            result,
            mode=mode,
            route=route,
            scope=scope,
            metadata=metadata,
        )

    def _run_step_phase(
        self,
        step: str,
        phase: str,
        data: Any,
        *,
        mode: str,
        route: str,
        scope: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        normalized_step = normalize_plugin_step(step)
        normalized_mode = normalize_plugin_mode(mode)

        if normalized_step not in STEP_HOOK_METHODS:
            logger.warning("忽略未知插件步骤: %s", normalized_step)
            return data

        method_name = STEP_HOOK_METHODS[normalized_step][phase]
        hook_entries = self.hooks.get(method_name, [])
        if not hook_entries:
            return data

        context = PluginContext(
            step=normalized_step,
            mode=normalized_mode,
            route=route,
            scope=scope,
            metadata=metadata or {},
        )
        current_data = data

        for _, plugin_id, hook_method in hook_entries:
            plugin = self.plugins.get(plugin_id)
            if not plugin or not plugin.is_enabled():
                continue
            if normalized_step not in plugin.supported_steps:
                continue
            if normalized_mode not in plugin.supported_modes:
                continue

            try:
                result = hook_method(context, current_data)
                if result is not None:
                    current_data = result
            except Exception as exc:
                logger.error(
                    "执行插件 '%s' 的钩子 '%s' 时出错: %s",
                    plugin_id,
                    method_name,
                    exc,
                    exc_info=True,
                )
                if plugin.failure_policy == "fail":
                    raise PluginException(
                        f"插件 '{plugin_id}' 在步骤 '{normalized_step}' 执行失败",
                        details={
                            "hook": method_name,
                            "route": route,
                            "mode": normalized_mode,
                        },
                    ) from exc

        return current_data

    def _get_plugin_config_filepath(self, plugin_id: str) -> str:
        safe_filename = "".join(
            c for c in plugin_id
            if c.isalnum() or c in ("_", "-")
        ).rstrip()
        if not safe_filename:
            plugin_digest = hashlib.sha1(
                plugin_id.encode("utf-8")
            ).hexdigest()[:12]
            safe_filename = f"plugin_{plugin_digest}"
        return os.path.join(self.plugin_config_dir, f"{safe_filename}.json")

    def _load_plugin_config_file(self, plugin_id: str) -> Dict[str, Any]:
        config_path = self._get_plugin_config_filepath(plugin_id)
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                    if isinstance(loaded, dict):
                        return loaded
            except (json.JSONDecodeError, IOError) as exc:
                logger.error(
                    "加载插件 '%s' 配置文件 '%s' 失败: %s",
                    plugin_id,
                    config_path,
                    exc,
                )
        return {}

    def save_plugin_config(self, plugin_id: str, config_data: Dict[str, Any]) -> bool:
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            logger.error("尝试保存未加载插件 '%s' 的配置。", plugin_id)
            return False

        config_path = self._get_plugin_config_filepath(plugin_id)
        try:
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config_data, file, indent=2, ensure_ascii=False)
            plugin.load_config(config_data)
            self._refresh_metadata(plugin_id)
            logger.info("插件 '%s' 的配置已保存到 '%s'。", plugin_id, config_path)
            return True
        except IOError as exc:
            logger.error(
                "保存插件 '%s' 配置文件 '%s' 失败: %s",
                plugin_id,
                config_path,
                exc,
            )
            return False

    def _delete_plugin_config_file(self, plugin_id: str) -> None:
        config_path = self._get_plugin_config_filepath(plugin_id)
        if not os.path.exists(config_path):
            return
        try:
            os.remove(config_path)
        except OSError as exc:
            logger.warning(
                "删除插件 '%s' 的配置文件 '%s' 失败: %s",
                plugin_id,
                config_path,
                exc,
            )


plugin_manager_instance: Optional[PluginManager] = None


def get_plugin_manager(app=None) -> PluginManager:
    global plugin_manager_instance
    if plugin_manager_instance is None:
        logger.info("创建插件管理器实例...")
        plugin_manager_instance = PluginManager(app=app)
        plugin_manager_instance.discover_and_load_plugins()
    elif app and plugin_manager_instance.app is not app:
        plugin_manager_instance.app = app
        for plugin in plugin_manager_instance.plugins.values():
            plugin.app = app
    return plugin_manager_instance


def apply_before_step_hooks(
    step: str,
    payload: Any,
    *,
    mode: str = "standard",
    route: str,
    scope: str = "image",
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    manager = get_plugin_manager()
    return manager.run_before_step(
        step,
        payload,
        mode=mode,
        route=route,
        scope=scope,
        metadata=metadata,
    )


def apply_after_step_hooks(
    step: str,
    result: Any,
    *,
    mode: str = "standard",
    route: str,
    scope: str = "image",
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    manager = get_plugin_manager()
    return manager.run_after_step(
        step,
        result,
        mode=mode,
        route=route,
        scope=scope,
        metadata=metadata,
    )
