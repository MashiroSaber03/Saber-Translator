// src/app/static/js/constants.js

// 这个常量在 main.js 的 loadPromptContent 和 loadTextboxPromptContent 中用到了
export const DEFAULT_PROMPT_NAME = "默认提示词";

// 你可以根据需要将其他在前端 JS 中使用的常量添加到这里
// 例如，默认字号、字体等，如果它们在多个 JS 文件中需要共享的话
// export const DEFAULT_FRONTEND_FONT_SIZE = 25;

// 如果有其他在 script.js 中定义的、需要在多个新 JS 模块中使用的常量，也移到这里并导出

// 默认气泡填充颜色（白色）
export const DEFAULT_FILL_COLOR = '#FFFFFF';
<<<<<<< HEAD

// --- 新增：自动存档常量 ---
export const AUTO_SAVE_SLOT_NAME = "__autosave__"; // 内部使用的固定名称
export const AUTO_SAVE_DISPLAY_NAME = "自动存档"; // UI上显示的名称
// ------------------------

// 新增 AI 视觉 OCR 默认提示词
export const DEFAULT_AI_VISION_OCR_PROMPT = `你是一个ocr助手，你需要将我发送给你的图片中的文字提取出来并返回给我，要求：
1、完整识别：我发送给你的图片中的文字都是需要识别的内容
2、非贪婪输出：不要返回任何其他解释和说明。`;

// --- 新增 JSON 格式默认提示词 ---
export const DEFAULT_TRANSLATE_JSON_PROMPT = `你是一个专业的翻译引擎。请将用户提供的文本翻译成简体中文。\n当文本中包含特殊字符（如大括号{}、引号""、反斜杠\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。\n请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:\n{\n  "translated_text": "[翻译后的文本放在这里]"\n}`;
export const DEFAULT_AI_VISION_OCR_JSON_PROMPT = `你是一个OCR助手。请将我发送给你的图片中的所有文字提取出来。\n当文本中包含特殊字符（如大括号{}、引号""、反斜杠\等）时，请在输出中保留它们但不要将它们视为JSON语法的一部分。如果需要，你可以使用转义字符\\来表示这些特殊字符。\n请严格按照以下 JSON 格式返回结果，不要添加任何额外的解释或对话:\n{\n  "extracted_text": "[这里放入所有识别到的文字，可以包含换行符以大致保留原始分段，但不要包含任何其他非文本内容]"\n}`;
// ----------------------------

// --- 新增 RPD 默认值 ---
export const DEFAULT_RPD_TRANSLATION = 0; // 0 表示无限制
export const DEFAULT_RPD_AI_VISION_OCR = 0;
// ------------------------
=======
>>>>>>> c92c015a833d6ba188c79cc00af9af36ed518915
