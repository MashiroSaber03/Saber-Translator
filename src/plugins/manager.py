import os
import importlib
import inspect
import logging
import json # 需要 json 来读写配置文件
from .base import PluginBase # 导入插件基类
from .hooks import ALL_HOOKS, BEFORE_PROCESSING, AFTER_OCR # 导入所有钩子名称常量
from src.shared.path_helpers import resource_path # 需要路径助手
from src.shared.config_loader import get_config_path # 需要获取 config 目录

logger = logging.getLogger("PluginManager")

class PluginManager:
    """
    负责发现、加载、管理和执行插件。
    """
    def __init__(self, app=None, plugin_dirs=None):
        """
        初始化插件管理器。

        Args:
            app: Flask 应用实例 (可选)。
            plugin_dirs (list, optional): 包含插件的目录列表。
                                         默认为 ['src/plugins', 'plugins']。
        """
        self.app = app
        self.plugins = {} # 存储加载的插件实例 {plugin_name: instance}
        self.hooks = {hook_name: [] for hook_name in ALL_HOOKS} # 存储每个钩子点注册的插件方法
        self.plugin_metadata = {} # 存储插件元数据
        self.plugin_dirs = plugin_dirs or [
            resource_path('src/plugins'), # 内置插件目录
            resource_path('plugins')      # 用户自定义插件目录 (根目录下)
        ]
        self.plugin_config_dir = get_config_path('plugin_configs') # config/plugin_configs/
        os.makedirs(self.plugin_config_dir, exist_ok=True)
        # 确保用户插件目录存在
        os.makedirs(resource_path('plugins'), exist_ok=True)

        logger.info(f"插件管理器初始化，扫描目录: {self.plugin_dirs}")

    def discover_and_load_plugins(self):
        """
        扫描指定目录，发现并加载所有有效的插件。
        """
        logger.info("开始发现和加载插件...")
        loaded_count = 0
        potential_plugins = self._find_potential_plugins()

        for name, module_path, package_path in potential_plugins:
            try:
                logger.debug(f"尝试加载插件模块: {name} 从 {module_path}")
                # 动态导入插件模块
                # 需要将包路径添加到 sys.path 吗？通常 importlib 能处理
                # spec = importlib.util.spec_from_file_location(name, module_path)
                # plugin_module = importlib.util.module_from_spec(spec)
                # spec.loader.exec_module(plugin_module)
                # 使用更简单的方式导入，假设目录结构允许
                module_import_path = self._get_module_import_path(package_path, name)
                if not module_import_path: continue # 无法确定导入路径

                plugin_module = importlib.import_module(module_import_path)

                # 在模块中查找继承自 PluginBase 的类
                for attr_name in dir(plugin_module):
                    attr = getattr(plugin_module, attr_name)
                    if inspect.isclass(attr) and issubclass(attr, PluginBase) and attr is not PluginBase:
                        logger.info(f"发现插件类: {attr.__name__} in {name}")
                        self._load_plugin_class(attr)
                        loaded_count += 1
                        break # 每个模块只加载第一个找到的插件类

            except ImportError as e:
                 logger.error(f"导入插件模块 '{module_import_path}' 失败: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"加载插件 '{name}' 时发生未知错误: {e}", exc_info=True)

        logger.info(f"插件加载完成，共加载 {loaded_count} 个插件。")
        self._enable_default_plugins() # 加载后启用默认插件

    def _find_potential_plugins(self):
        """在插件目录中查找可能的插件模块。"""
        potential_plugins = []
        for plugin_dir in self.plugin_dirs:
            if not os.path.isdir(plugin_dir):
                logger.warning(f"插件目录不存在或不是目录: {plugin_dir}")
                continue
            logger.info(f"扫描插件目录: {plugin_dir}")
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                # 插件通常是一个包含 __init__.py 和 plugin.py 的目录
                if os.path.isdir(item_path) and \
                   os.path.exists(os.path.join(item_path, '__init__.py')) and \
                   os.path.exists(os.path.join(item_path, 'plugin.py')):
                    plugin_name = item
                    module_file_path = os.path.join(item_path, 'plugin.py')
                    # 记录插件包路径和模块文件名
                    potential_plugins.append((plugin_name, module_file_path, item_path))
                    logger.debug(f"  发现潜在插件目录: {plugin_name}")
        return potential_plugins

    def _get_module_import_path(self, package_path, plugin_name):
        """根据包路径和插件名生成 Python 导入路径。"""
        # 简化的路径处理
        normalized_path = package_path.replace('\\', '/')
        if 'src/plugins' in normalized_path:
            return f"src.plugins.{plugin_name}.plugin"
        elif 'plugins' in normalized_path and not normalized_path.startswith('src'):
            return f"plugins.{plugin_name}.plugin"
        else:
            logger.warning(f"无法确定插件 '{plugin_name}' 的导入路径: {package_path}")
            return None


    def _load_plugin_class(self, plugin_class):
        """实例化插件类并注册钩子。"""
        try:
            plugin_instance = plugin_class(plugin_manager=self, app=self.app)
            plugin_name = plugin_instance.plugin_name

            if plugin_name in self.plugins:
                logger.warning(f"插件名称冲突: '{plugin_name}' 已存在，跳过 {plugin_class.__name__}。")
                return

            # 执行插件的 setup 方法
            if plugin_instance.setup():
                self.plugins[plugin_name] = plugin_instance
                self.plugin_metadata[plugin_name] = plugin_instance.get_metadata()
                # --- 加载插件配置 ---
                config_data = self._load_plugin_config_file(plugin_name)
                plugin_instance.load_config(config_data)
                # --------------------
                self._register_hooks(plugin_instance)
                logger.info(f"插件 '{plugin_name}' 加载、配置并设置成功。")
            else:
                logger.error(f"插件 '{plugin_name}' 的 setup 方法返回 False，加载失败。")

        except Exception as e:
            logger.error(f"实例化或设置插件类 '{plugin_class.__name__}' 失败: {e}", exc_info=True)

    def _register_hooks(self, plugin_instance):
        """检查插件实例并注册其实现的钩子方法。"""
        plugin_name = plugin_instance.plugin_name
        for hook_name in ALL_HOOKS:
            if hasattr(plugin_instance, hook_name) and callable(getattr(plugin_instance, hook_name)):
                # 确保方法不是基类中的默认 pass 方法
                method = getattr(plugin_instance, hook_name)
                base_method = getattr(PluginBase, hook_name, None)
                if method.__code__ is not base_method.__code__: # 比较字节码判断是否被覆盖
                    self.hooks[hook_name].append(method)
                    logger.debug(f"插件 '{plugin_name}' 注册了钩子: {hook_name}")

    def _enable_default_plugins(self):
        """启用标记为默认启用的插件。"""
        logger.info("跳过启用默认插件，所有插件将默认保持关闭状态...")
        # 以下是原代码，现在被禁用
        # for name, instance in self.plugins.items():
        #     if instance.plugin_enabled_by_default:
        #         instance.enable()

    def get_plugin(self, name):
        """获取指定名称的插件实例。"""
        return self.plugins.get(name)

    def get_all_plugins(self):
        """获取所有已加载的插件实例。"""
        return list(self.plugins.values())

    def get_all_metadata(self):
        """获取所有插件的元数据。"""
        return list(self.plugin_metadata.values())

    def enable_plugin(self, name):
        """启用指定名称的插件。"""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enable()
        else:
            logger.warning(f"尝试启用未找到的插件: {name}")

    def disable_plugin(self, name):
        """禁用指定名称的插件。"""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.disable()
        else:
            logger.warning(f"尝试禁用未找到的插件: {name}")

    def unregister_plugin_hooks(self, plugin_name_to_remove):
        """注销指定插件的所有钩子"""
        removed_count = 0
        for hook_name in self.hooks:
            original_len = len(self.hooks[hook_name])
            self.hooks[hook_name] = [
                method for method in self.hooks[hook_name]
                if not (hasattr(method, '__self__') and hasattr(method.__self__, 'plugin_name') and method.__self__.plugin_name == plugin_name_to_remove)
            ]
            removed_count += original_len - len(self.hooks[hook_name])
        if removed_count > 0:
            logger.info(f"已注销插件 '{plugin_name_to_remove}' 的 {removed_count} 个钩子。")

    def trigger_hook(self, hook_name, *args, **kwargs):
        """
        触发指定的钩子点，并按顺序执行所有注册的方法。
        对于需要修改数据的钩子，它会传递修改后的结果。

        Args:
            hook_name (str): 要触发的钩子名称 (使用 hooks.py 中的常量)。
            *args: 传递给钩子函数的参数。
            **kwargs: 传递给钩子函数的关键字参数。

        Returns:
            Any: 对于需要修改数据的钩子，返回最后一个插件修改后的结果；
                 对于通知类钩子，返回 None。
        """
        if hook_name not in self.hooks:
            logger.warning(f"尝试触发未知的钩子: {hook_name}")
            return args if args else None # 返回原始参数或 None

        if not self.hooks[hook_name]:
            # logger.debug(f"钩子 '{hook_name}' 没有注册的处理函数。")
            return args if args else None

        logger.debug(f"触发钩子: {hook_name} (有 {len(self.hooks[hook_name])} 个处理函数)")

        # 确定钩子是否需要修改数据 (基于钩子名称约定或显式定义)
        # 假设需要修改数据的钩子返回修改后的参数元组，否则返回 None
        modifies_data = hook_name not in ["on_enable", "on_disable"] # 简单示例

        current_args = args
        current_kwargs = kwargs # 允许插件修改 kwargs 吗？暂时不允许

        for hook_method in self.hooks[hook_name]:
            try:
                # 只调用启用的插件的钩子
                plugin_instance = hook_method.__self__ # 获取方法所属的实例
                if isinstance(plugin_instance, PluginBase) and plugin_instance.is_enabled():
                    logger.debug(f"  执行插件 '{plugin_instance.plugin_name}' 的钩子: {hook_name}")
                    result = hook_method(*current_args, **current_kwargs)

                    # 如果钩子设计为修改数据，则更新 current_args
                    if modifies_data and result is not None:
                        if isinstance(result, tuple):
                            # 使用插件返回的结果作为下一个插件的输入
                            # 如果元素过多，只使用前 n 个元素；如果元素过少，用原始输入补充
                            if len(result) >= len(current_args):
                                # 仅使用需要的元素数量
                                current_args = result[:len(current_args)]
                                logger.debug(f"    钩子返回了修改后的数据。")
                            elif len(result) > 0:
                                # 如果返回的元素数量不足，但还有一些元素，那么我们将这些元素替换到当前参数中
                                new_args = list(current_args)
                                for i, val in enumerate(result):
                                    new_args[i] = val
                                current_args = tuple(new_args)
                                logger.debug(f"    钩子返回了部分修改后的数据。")
                            else:
                                logger.warning(f"    插件 '{plugin_instance.plugin_name}' 的钩子 '{hook_name}' 返回了空元组，已忽略。")
                        else:
                             logger.warning(f"    插件 '{plugin_instance.plugin_name}' 的钩子 '{hook_name}' 返回了非元组数据，已忽略。")

                # else: # 插件未启用
                #     logger.debug(f"  跳过禁用的插件 '{plugin_instance.plugin_name}' 的钩子: {hook_name}")

            except Exception as e:
                logger.error(f"执行插件 '{hook_method.__self__.plugin_name}' 的钩子 '{hook_name}' 时出错: {e}", exc_info=True)
                # 选择继续执行其他插件还是中断？这里选择继续
                continue

        # 如果钩子修改数据，返回最终结果；否则返回 None
        return current_args if modifies_data else None

    def _get_plugin_config_filepath(self, plugin_name):
        """获取插件配置文件的路径"""
        # 使用插件名称作为文件名，确保文件名安全
        safe_filename = "".join(c for c in plugin_name if c.isalnum() or c in ('_', '-')).rstrip()
        if not safe_filename:
            safe_filename = f"plugin_{hash(plugin_name)}" # 如果名称无效，使用哈希值
        return os.path.join(self.plugin_config_dir, f"{safe_filename}.json")
    
    def _load_plugin_config_file(self, plugin_name):
        """从文件加载指定插件的配置"""
        config_path = self._get_plugin_config_filepath(plugin_name)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"加载插件 '{plugin_name}' 配置文件 '{config_path}' 失败: {e}")
        return {} # 文件不存在或加载失败，返回空字典
    
    def save_plugin_config(self, plugin_name, config_data):
        """保存指定插件的配置到文件"""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            logger.error(f"尝试保存未加载插件 '{plugin_name}' 的配置。")
            return False
    
        config_path = self._get_plugin_config_filepath(plugin_name)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"插件 '{plugin_name}' 的配置已保存到 '{config_path}'。")
            # 保存后，让插件实例重新加载配置
            plugin.load_config(config_data)
            return True
        except IOError as e:
            logger.error(f"保存插件 '{plugin_name}' 配置文件 '{config_path}' 失败: {e}")
            return False

# --- 全局插件管理器实例 ---
# 可以在应用启动时创建
plugin_manager_instance = None

def get_plugin_manager(app=None):
    """获取插件管理器的单例实例。"""
    global plugin_manager_instance
    if plugin_manager_instance is None:
        logger.info("创建插件管理器实例...")
        plugin_manager_instance = PluginManager(app=app)
        plugin_manager_instance.discover_and_load_plugins() # 创建时自动加载插件
    elif app and plugin_manager_instance.app is None:
         # 如果之前没有 app 实例，现在传入了，则更新
         plugin_manager_instance.app = app
    return plugin_manager_instance

# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试插件管理器 ---")
    # 需要先创建示例插件 src/plugins/example_plugin/plugin.py
    
    # 确保在 example_plugin/__init__.py 中可以导入 ExamplePlugin 类
    # 例如: from .plugin import ExamplePlugin

    # 初始化日志以便查看输出
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("初始化插件管理器...")
    manager = get_plugin_manager()

    print("\n已加载插件:")
    for plugin in manager.get_all_plugins():
        meta = plugin.get_metadata()
        print(f"  - {meta['name']} v{meta['version']} by {meta['author']} (Enabled: {plugin.is_enabled()})")
        print(f"    {meta['description']}")

    print("\n测试触发钩子:")
    # 模拟触发钩子，需要伪造参数
    try:
        from PIL import Image
        dummy_image = Image.new('RGB', (10, 10))
        dummy_params = {'target_language': 'zh'}
        dummy_coords = [(0,0,5,5)]
        dummy_texts = ["原始文本"]

        print(f"\n触发 {BEFORE_PROCESSING}...")
        manager.trigger_hook(BEFORE_PROCESSING, dummy_image, dummy_params)

        print(f"\n触发 {AFTER_OCR}...")
        manager.trigger_hook(AFTER_OCR, dummy_image, dummy_texts, dummy_coords, dummy_params)

        print("\n测试禁用插件:")
        manager.disable_plugin("示例插件")
        plugin = manager.get_plugin("示例插件")
        if plugin: print(f"示例插件启用状态: {plugin.is_enabled()}")

        print(f"\n再次触发 {AFTER_OCR} (插件应被跳过)...")
        manager.trigger_hook(AFTER_OCR, dummy_image, dummy_texts, dummy_coords, dummy_params)

        print("\n测试启用插件:")
        manager.enable_plugin("示例插件")
        if plugin: print(f"示例插件启用状态: {plugin.is_enabled()}")

        print(f"\n再次触发 {AFTER_OCR} (插件应执行)...")
        manager.trigger_hook(AFTER_OCR, dummy_image, dummy_texts, dummy_coords, dummy_params)
    except ImportError as e:
        print(f"测试需要PIL库: {e}")
    except Exception as e:
        print(f"测试过程中出错: {e}")