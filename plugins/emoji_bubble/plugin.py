# plugins/emoji_bubble/plugin.py
import logging
import random
from src.plugins.base import PluginBase
from src.plugins.hooks import BEFORE_RENDERING # 我们在渲染前修改文本

class EmojiBubblePlugin(PluginBase):
    """
    用随机表情符号替换气泡中的翻译文本。
    """
    # --- 插件元数据 ---
    plugin_name = "气泡文本表情化"
    plugin_version = "1.0"
    plugin_author = "AI 助手"
    plugin_description = "将翻译后的文本随机替换为指定的表情符号。可通过设置自定义表情列表。"
    plugin_enabled_by_default = False # 默认不启用，因为会替换文本

    def get_config_spec(self):
        """定义插件配置项"""
        return [
            {
                "name": "enable_replacement", # 内部名称
                "label": "启用表情替换",     # 显示给用户的标签
                "type": "boolean",         # 输入类型
                "default": True,           # 默认值
                "description": "勾选后才会执行表情替换功能。" # 描述
            },
            {
                "name": "emoji_chars",
                "label": "表情符号列表",
                "type": "text",            # 文本输入框
                "default": "😀😂🤣😊😇😍🤔🥳👍🎉🍕🍔🍟🥨🍎", # 提供一些默认表情
                "description": "输入你想用来随机替换文本的表情符号，直接连续输入即可。"
            }
        ]

    def setup(self):
        """插件设置"""
        self.logger.info(f"'{self.plugin_name}' 插件已加载。")
        # 这里不需要特别的设置
        return True

    def before_rendering(self, image_to_render_on, translated_texts, bubble_coords, bubble_styles, params):
        """
        渲染前执行，将 translated_texts 替换为随机表情。
        """
        # 检查插件和配置是否都启用了替换功能
        if not self.is_enabled() or not self.config.get('enable_replacement', False):
            return None # 不做任何修改

        self.logger.info(f"钩子 {BEFORE_RENDERING}: '{self.plugin_name}' 正在执行...")

        # 从配置中获取表情列表，如果为空则使用默认的✅
        emoji_list = self.config.get('emoji_chars', "✅")
        if not emoji_list:
            emoji_list = "✅"
            self.logger.warning("配置中的表情列表为空，将使用默认表情 '✅'")

        # 将字符串转换为表情符号列表
        valid_emojis = [char for char in emoji_list] # Python 3 字符串可以直接迭代字符
        if not valid_emojis:
             valid_emojis = ["✅"] # 最终回退

        self.logger.debug(f"使用的表情列表: {' '.join(valid_emojis)}")

        # 创建一个新的列表来存储修改后的文本
        modified_texts = []
        for original_text in translated_texts:
            if original_text and original_text.strip(): # 只替换非空文本
                # 生成与原文等长的随机表情字符串
                emoji_string = "".join(random.choice(valid_emojis) for _ in original_text)
                modified_texts.append(emoji_string)
            else:
                modified_texts.append(original_text) # 保留空文本或空格

        self.logger.info(f"已将 {len(modified_texts)} 段文本替换为随机表情。")

        # 返回修改后的文本列表和其他未修改的参数
        # 顺序必须是: image, texts, coords, styles, params
        return image_to_render_on, modified_texts, bubble_coords, bubble_styles, params