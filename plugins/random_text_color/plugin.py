# plugins/random_text_color/plugin.py
import logging
import random
from src.plugins.base import PluginBase
from src.plugins.hooks import BEFORE_RENDERING

class RandomTextColorPlugin(PluginBase):
    """
    一个简单的示例插件，在渲染文本前为每个气泡设置随机颜色。
    """
    # --- 插件元数据 ---
    plugin_name = "随机文本颜色"
    plugin_version = "1.0"
    plugin_author = "你的AI助手"
    plugin_description = "使每个气泡框内的文本显示为随机颜色。可通过设置调整亮度下限。"
    plugin_enabled_by_default = True # 默认启用这个趣味插件

    def get_config_spec(self):
        """定义插件配置项"""
        return [
            {
                "name": "min_brightness",
                "label": "最低亮度",
                "type": "number",
                "default": 80,
                "description": "随机颜色的最低亮度值 (0-255)，值越高颜色越亮。"
            },
            {
                "name": "avoid_white",
                "label": "避免接近白色",
                "type": "boolean",
                "default": True,
                "description": "勾选后将避免生成非常接近白色的颜色。"
            }
        ]

    def setup(self):
        """插件设置"""
        self.logger.info(f"'{self.plugin_name}' 插件已加载。")
        # 这个插件很简单，不需要复杂的设置
        return True

    def get_random_bright_hex_color(self):
        """生成一个随机且符合配置亮度的十六进制颜色代码"""
        # 从加载的配置中获取值，如果配置未加载则使用默认值
        min_brightness = self.config.get('min_brightness', 80)
        avoid_white = self.config.get('avoid_white', True)

        while True:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            # 避免颜色太暗 (亮度简单估算)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            # 避免颜色太接近白色 (可选)
            too_white = avoid_white and (r > 240 and g > 240 and b > 240)
            if brightness >= min_brightness and not too_white:
                return f"#{r:02x}{g:02x}{b:02x}"

    def before_rendering(self, image_to_render_on, translated_texts, bubble_coords, bubble_styles, params):
        """
        在渲染文本之前执行，修改气泡样式中的文本颜色。
        """
        if not self.is_enabled():
            return None

        self.logger.info(f"钩子 {BEFORE_RENDERING}: 正在为 {len(bubble_styles)} 个气泡设置随机颜色...")

        # 创建新的样式字典
        modified_styles = bubble_styles.copy()
        for i_str in modified_styles.keys():
            random_color = self.get_random_bright_hex_color() # 使用新的颜色生成函数
            current_style = modified_styles.get(i_str, {})
            current_style['text_color'] = random_color
            modified_styles[i_str] = current_style
            
        # 返回完整的5元素元组，与钩子函数接收的参数数量相同
        return image_to_render_on, translated_texts, bubble_coords, modified_styles, params