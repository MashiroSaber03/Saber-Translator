# src/plugins/base.py
import logging

class PluginBase:
    """
    所有插件的基类。
    插件应继承此类并实现所需的方法（特别是钩子方法）。
    """
    # --- 插件元数据 ---
    # 插件开发者应在子类中覆盖这些属性
    plugin_name = "未命名插件"
    plugin_version = "0.1.0"
    plugin_author = "未知作者"
    plugin_description = "这是一个基础插件描述。"
    # 插件是否默认启用
    plugin_enabled_by_default = False

    def __init__(self, plugin_manager, app=None):
        """
        初始化插件实例。

        Args:
            plugin_manager: 插件管理器实例，用于插件间通信或访问管理器功能。
            app: Flask 应用实例 (可选)，用于需要访问 Flask 上下文的插件。
        """
        self.plugin_manager = plugin_manager
        self.app = app
        self.logger = logging.getLogger(f"Plugin.{self.plugin_name}")
        self._enabled = False # 插件内部的启用状态
        self.config = {} # 用于存储加载后的配置值

    def setup(self):
        """
        插件设置和初始化方法。
        在插件被加载后、应用启动前调用。
        可以在这里进行资源加载、API 客户端初始化等。
        如果返回 False，则插件加载失败。
        """
        self.logger.info(f"插件 '{self.plugin_name}' v{self.plugin_version} 正在设置...")
        # 默认实现，返回 True 表示设置成功
        return True

    def enable(self):
        """启用插件。"""
        if not self._enabled:
            self._enabled = True
            self.logger.info(f"插件 '{self.plugin_name}' 已启用。")
            self.on_enable() # 调用启用时的钩子

    def disable(self):
        """禁用插件。"""
        if self._enabled:
            self._enabled = False
            self.logger.info(f"插件 '{self.plugin_name}' 已禁用。")
            self.on_disable() # 调用禁用时的钩子

    def is_enabled(self):
        """检查插件当前是否启用。"""
        return self._enabled

    def get_metadata(self):
        """返回插件的元数据字典。"""
        return {
            "name": self.plugin_name,
            "version": self.plugin_version,
            "author": self.plugin_author,
            "description": self.plugin_description,
            "enabled_by_default": self.plugin_enabled_by_default
        }

    # --- 钩子方法 (插件可以覆盖这些方法来响应事件) ---

    def on_enable(self):
        """插件启用时调用的钩子。"""
        pass

    def on_disable(self):
        """插件禁用时调用的钩子。"""
        pass

    def before_processing(self, image_pil, params):
        """
        在核心图像处理流程开始之前调用。
        可以修改 params 字典来影响处理流程。

        Args:
            image_pil (PIL.Image.Image): 原始输入图像。
            params (dict): 包含所有处理参数的字典。

        Returns:
            tuple(PIL.Image.Image, dict) or None: 修改后的图像和参数，或 None 表示不修改。
        """
        # self.logger.debug("钩子: before_processing")
        return None

    def after_detection(self, image_pil, bubble_coords, params):
        """
        在气泡检测完成之后调用。
        可以修改气泡坐标列表。

        Args:
            image_pil (PIL.Image.Image): 原始图像。
            bubble_coords (list): 检测到的气泡坐标列表。
            params (dict): 处理参数。

        Returns:
            list or None: 修改后的气泡坐标列表，或 None 表示不修改。
        """
        # self.logger.debug(f"钩子: after_detection - 找到 {len(bubble_coords)} 个气泡")
        return None

    def before_ocr(self, image_pil, bubble_coords, params):
        """在 OCR 开始之前调用。"""
        # self.logger.debug("钩子: before_ocr")
        return None

    def after_ocr(self, image_pil, original_texts, bubble_coords, params):
        """
        在 OCR 完成之后调用。
        可以修改识别出的原始文本列表。

        Args:
            image_pil (PIL.Image.Image): 原始图像。
            original_texts (list): OCR 识别出的文本列表。
            bubble_coords (list): 气泡坐标列表。
            params (dict): 处理参数。

        Returns:
            list or None: 修改后的原始文本列表，或 None 表示不修改。
        """
        # self.logger.debug(f"钩子: after_ocr - 识别出 {len(original_texts)} 段文本")
        return None

    def before_translation(self, original_texts, params):
        """
        在文本翻译之前调用。
        可以修改待翻译的文本列表或翻译参数。

        Args:
            original_texts (list): 待翻译的文本列表。
            params (dict): 处理参数 (包含翻译相关设置)。

        Returns:
            tuple(list, dict) or None: 修改后的文本列表和参数，或 None 表示不修改。
        """
        # self.logger.debug("钩子: before_translation")
        return None

    def after_translation(self, translated_bubble_texts, translated_textbox_texts, original_texts, params):
        """
        在文本翻译之后调用。
        可以修改翻译结果。

        Args:
            translated_bubble_texts (list): 气泡翻译结果列表。
            translated_textbox_texts (list): 文本框翻译结果列表。
            original_texts (list): 原始文本列表。
            params (dict): 处理参数。

        Returns:
            tuple(list, list) or None: 修改后的气泡译文和文本框译文列表，或 None 表示不修改。
        """
        # self.logger.debug("钩子: after_translation")
        return None

    def before_inpainting(self, image_pil, bubble_coords, params):
        """在背景修复/填充之前调用。"""
        # self.logger.debug("钩子: before_inpainting")
        return None

    def after_inpainting(self, inpainted_image, clean_background, bubble_coords, params):
        """
        在背景修复/填充之后调用。
        可以修改修复后的图像或干净背景。

        Args:
            inpainted_image (PIL.Image.Image): 修复/填充后的图像。
            clean_background (PIL.Image.Image or None): 生成的干净背景。
            bubble_coords (list): 气泡坐标。
            params (dict): 处理参数。

        Returns:
            tuple(PIL.Image.Image, PIL.Image.Image or None) or None: 修改后的图像和干净背景，或 None。
        """
        # self.logger.debug("钩子: after_inpainting")
        return None

    def before_rendering(self, image_to_render_on, translated_texts, bubble_coords, bubble_styles, params):
        """
        在文本渲染之前调用。
        可以修改用于渲染的图像、文本、坐标或样式。

        Args:
            image_to_render_on (PIL.Image.Image): 将在其上渲染文本的基础图像。
            translated_texts (list): 要渲染的文本列表。
            bubble_coords (list): 气泡坐标。
            bubble_styles (dict): 气泡样式字典。
            params (dict): 处理参数。

        Returns:
            tuple(PIL.Image.Image, list, list, dict) or None: 修改后的参数，或 None。
        """
        # self.logger.debug("钩子: before_rendering")
        return None

    def after_processing(self, final_image, results, params):
        """
        在整个图像处理流程完成之后，返回最终结果之前调用。
        可以修改最终的图像或结果字典。

        Args:
            final_image (PIL.Image.Image): 最终处理完成的图像。
            results (dict): 包含所有中间结果的字典 (例如 'original_texts', 'bubble_texts', ...)。
            params (dict): 处理参数。

        Returns:
             tuple(PIL.Image.Image, dict) or None: 修改后的最终图像和结果字典，或 None。
        """
        # self.logger.debug("钩子: after_processing")
        return None

    # 可以根据需要添加更多钩子点，例如：
    # - 应用启动/关闭时
    # - API 请求处理前后
    # - UI 元素加载后

    def get_config_spec(self):
        """
        插件声明其配置项。
        子类应覆盖此方法。

        Returns:
            list[dict] or None: 配置项列表，每个字典包含:
                - name: str, 配置项内部名称 (必需)
                - label: str, 显示给用户的标签 (必需)
                - type: str, 输入类型 ('text', 'number', 'boolean', 'select') (必需)
                - default: any, 默认值 (必需)
                - description: str, 配置项描述 (可选)
                - options: list[str], type='select' 时的选项列表 (可选)
            如果插件没有配置项，返回 None 或空列表。
        """
        return None # 默认没有配置项

    def load_config(self, config_data):
        """
        加载用户配置到插件实例。
        由插件管理器在加载插件或配置更新后调用。

        Args:
            config_data (dict): 从存储中加载的该插件的配置字典。
        """
        spec = self.get_config_spec() or []
        loaded_config = {}
        # 基于 spec 加载配置，确保类型正确并使用默认值
        for item in spec:
            key = item.get('name')
            default_value = item.get('default')
            value_type = item.get('type')
            if key:
                # 从传入的 config_data 获取值，否则使用默认值
                value = config_data.get(key, default_value)
                # (可选) 进行类型转换或验证
                try:
                    if value_type == 'number':
                        value = float(value) if '.' in str(value) else int(value)
                    elif value_type == 'boolean':
                        value = str(value).lower() in ['true', '1', 'yes', 'on']
                except (ValueError, TypeError):
                    self.logger.warning(f"配置项 '{key}' 的值 '{value}' 类型无效，使用默认值 '{default_value}'")
                    value = default_value
                loaded_config[key] = value
        self.config = loaded_config
        self.logger.info(f"插件 '{self.plugin_name}' 配置已加载: {self.config}")