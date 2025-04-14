# src/plugins/example_plugin/plugin.py
from ..base import PluginBase
from ..hooks import * # 导入所有钩子常量

class ExamplePlugin(PluginBase):
    plugin_name = "示例插件"
    plugin_version = "1.0"
    plugin_author = "测试者"
    plugin_description = "一个简单的示例插件，用于演示钩子功能。"
    plugin_enabled_by_default = False # 禁用示例插件

    def setup(self):
        self.logger.info("示例插件正在设置...")
        # 可以在这里加载资源或进行初始化检查
        return True # 返回 True 表示设置成功

    def on_enable(self):
         self.logger.info("示例插件已启用！")

    def on_disable(self):
         self.logger.info("示例插件已禁用。")

    def before_processing(self, image_pil, params):
        self.logger.info(f"钩子 {BEFORE_PROCESSING}: 收到图像，参数: {params}")
        # 示例：修改参数
        # params['target_language'] = 'en'
        # self.logger.info(f"  修改目标语言为: {params['target_language']}")
        # return image_pil, params # 返回修改后的数据
        return None # 不修改

    def after_ocr(self, image_pil, original_texts, bubble_coords, params):
         self.logger.info(f"钩子 {AFTER_OCR}: 识别到 {len(original_texts)} 段文本: {original_texts}")
         # 示例：在文本前后添加标记
         # modified_texts = [f"[Plugin]{t}[/Plugin]" for t in original_texts]
         # return modified_texts # 返回修改后的文本列表
         return None