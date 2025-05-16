# plugins/text_affixer/plugin.py
import logging
from src.plugins.base import PluginBase
from src.plugins.hooks import AFTER_TRANSLATION # 我们要在翻译后修改文本

class TextAffixerPlugin(PluginBase):
    """
    一个在译文前后添加自定义文本的插件。
    """
    # --- 插件元数据 ---
    plugin_name = "文本装饰器"
    plugin_version = "1.0"
    plugin_author = "MashiroSaber"
    plugin_description = "在每个气泡的译文之前或之后添加自定义的前缀或后缀文本。"
    plugin_enabled_by_default = False # 默认不启用，让用户手动开启

    def get_config_spec(self):
        """定义插件配置项"""
        return [
            {
                "name": "text_prefix",               # 配置项内部名称
                "label": "文本前缀",                  # 显示给用户的标签
                "type": "text",                     # 输入框类型
                "default": "",                      # 默认值（空字符串）
                "description": "添加到每个气泡译文之前的内容。例如：'(内心) ' 或 '「'"
            },
            {
                "name": "text_suffix",               # 配置项内部名称
                "label": "文本后缀",                  # 显示给用户的标签
                "type": "text",                     # 输入框类型
                "default": "",                      # 默认值
                "description": "添加到每个气泡译文之后的内容。例如：'！' 或 '」'"
            },
            {
                "name": "apply_to_textbox",          # 配置项内部名称
                "label": "应用于文本框内容",          # 显示给用户的标签
                "type": "boolean",                  # 复选框类型
                "default": False,                   # 默认不应用于文本框
                "description": "如果勾选，前缀和后缀也将应用于右侧文本框的翻译内容。"
            }
        ]

    def setup(self):
        """插件设置"""
        self.logger.info(f"'{self.plugin_name}' 插件已加载并设置。")
        # 可以在这里预加载或检查资源，如果需要的话
        return True

    def after_translation(self, translated_bubble_texts, translated_textbox_texts, original_texts, params):
        """
        翻译完成后执行的钩子，用于添加前缀和后缀。
        """
        if not self.is_enabled(): # 检查插件是否已启用
            return None # 未启用则不作任何修改

        # 从已加载的配置中获取用户设置的前缀和后缀
        # self.config 是在管理器加载配置后，通过 load_config 方法设置的
        prefix = self.config.get('text_prefix', '')
        suffix = self.config.get('text_suffix', '')
        apply_to_textbox = self.config.get('apply_to_textbox', False)

        # 如果前缀和后缀都为空，则无需处理
        if not prefix and not suffix:
            return None

        self.logger.info(f"钩子 {AFTER_TRANSLATION}: 应用前缀 '{prefix}' 和后缀 '{suffix}'")

        # --- 修改气泡文本 ---
        modified_bubble_texts = []
        for text in translated_bubble_texts:
            # 只对非空文本添加前后缀
            if text and text.strip():
                modified_bubble_texts.append(f"{prefix}{text}{suffix}")
            else:
                modified_bubble_texts.append(text) # 保留空字符串或纯空格

        # --- 选择性修改文本框文本 ---
        modified_textbox_texts = translated_textbox_texts # 默认不修改
        if apply_to_textbox:
            modified_textbox_texts = []
            for text in translated_textbox_texts:
                if text and text.strip():
                    modified_textbox_texts.append(f"{prefix}{text}{suffix}")
                else:
                    modified_textbox_texts.append(text)
        else:
             # 如果不应用到文本框，则保持原始文本框翻译结果
             modified_textbox_texts = translated_textbox_texts

        # 返回修改后的文本列表元组 (顺序必须与钩子参数对应)
        # 我们只修改了前两个列表，所以返回修改后的这两个
        return modified_bubble_texts, modified_textbox_texts