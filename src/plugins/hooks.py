# src/plugins/hooks.py

"""
定义插件系统可用的钩子点名称常量。
插件管理器和核心代码将使用这些常量来触发和注册钩子。
"""

# --- 处理流程钩子 ---
BEFORE_PROCESSING = "before_processing"
AFTER_DETECTION = "after_detection"
BEFORE_OCR = "before_ocr"
AFTER_OCR = "after_ocr"
BEFORE_TRANSLATION = "before_translation"
AFTER_TRANSLATION = "after_translation"
BEFORE_INPAINTING = "before_inpainting"
AFTER_INPAINTING = "after_inpainting"
BEFORE_RENDERING = "before_rendering"
AFTER_PROCESSING = "after_processing"

# --- 应用生命周期钩子 (示例) ---
# ON_APP_STARTUP = "on_app_startup"
# ON_APP_SHUTDOWN = "on_app_shutdown"

# --- 其他可能的钩子 ---
# MODIFY_TRANSLATION_PARAMS = "modify_translation_params"
# MODIFY_RENDERING_STYLE = "modify_rendering_style"

# 获取所有已定义的钩子名称列表 (可选)
ALL_HOOKS = [
    BEFORE_PROCESSING, AFTER_DETECTION, BEFORE_OCR, AFTER_OCR,
    BEFORE_TRANSLATION, AFTER_TRANSLATION, BEFORE_INPAINTING, AFTER_INPAINTING,
    BEFORE_RENDERING, AFTER_PROCESSING,
    # ON_APP_STARTUP, ON_APP_SHUTDOWN, # 如果添加了生命周期钩子
]