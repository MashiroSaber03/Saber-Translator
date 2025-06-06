// src/app/static/js/constants.js

// 这个常量在 main.js 的 loadPromptContent 和 loadTextboxPromptContent 中用到了
export const DEFAULT_PROMPT_NAME = "默认提示词";

// 你可以根据需要将其他在前端 JS 中使用的常量添加到这里
// 例如，默认字号、字体等，如果它们在多个 JS 文件中需要共享的话
// export const DEFAULT_FRONTEND_FONT_SIZE = 25;

// 如果有其他在 script.js 中定义的、需要在多个新 JS 模块中使用的常量，也移到这里并导出

// 默认气泡填充颜色（白色）
export const DEFAULT_FILL_COLOR = '#FFFFFF';

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

// --- 新增：高质量翻译模式默认提示词 ---
export const DEFAULT_HQ_TRANSLATE_PROMPT = `你是一个漫画翻译助手，我会提供原始漫画图片和导出的只含原文不含译文JSON翻译文件，帮我将原文翻译成中文
翻译要求：
1.仅修改JSON中的"translated"译文内容，保持其他所有结构和字段不变
2.json中的"imagelndex"序号是每张图片的页码，在翻译前先根据json文件中每页图片的原文内容和所有漫画图片对我给你的所有图片进行排序，在进行上下文对比时要严格按照"imagelndex"序号进行对比，你在翻译时要按照"imagelndex"的顺序进行顺序翻译，从而使得每句翻译足够连贯，上下文不会突兀
3.通过json中每页图片的"original"原文内容和所有的漫画图片明确每句话在那页图的哪个位置，并结合漫画图像和上下文语境，让翻译更加连贯自然，符合角色语气和场景
4.json中的"bubblelndex"标号可能不正确，你需要根据图片自行判断每句话的正确顺序，从而输出符合上下文连贯的翻译，但在输出翻译时不要修改原本的"bubblelndex"标号
5.保留原文的表达意图和情感，但使用更地道、流畅的表达方式
6.注意专有名词和术语的一致性翻译
7.如遇幽默、双关语或文化特定内容，请尽量找到目标语言中恰当的表达
8.不要添加原文中不存在的信息或解释
9.保持简洁明了，符合气泡空间限制
10.译文字数尽量不要超过原文
11.你需要直接返回修改后的完整JSON文件，无需解释每处修改原因。`;

// AI校对的默认提示词
export const DEFAULT_PROOFREADING_PROMPT = `你是一个专业的漫画校对助手，请帮我校对漫画翻译结果。我会给你漫画未经翻译的原图、由漫画原文和已有的翻译文本组成的JSON文件，请根据图片内容和上下文关系，检查并改进翻译质量。
校对要点：
1. json中的"imagelndex"序号是每张图片的页码，在翻译前先根据json文件中每页图片的原文内容和所有漫画图片对我给你的所有图片进行排序，在进行上下文对比时要严格按照"imagelndex"序号进行对比，你在翻译时要按照"imagelndex"的顺序进行顺序翻译，从而使得每句翻译足够连贯，上下文不会突兀
2. 通过json中每页图片的"original"原文内容和所有的漫画图片明确每句话在那页图的哪个位置，并结合漫画图像和上下文语境，让翻译更加连贯自然，符合角色语气和场景
3. json中的"bubblelndex"标号可能不正确，你需要根据图片自行判断每句话的正确顺序，从而输出符合上下文连贯的翻译，但在输出翻译时不要修改原本的"bubblelndex"标号
4. 针对特殊术语或专有名词的翻译进行统一
5. 重点关注易错点的人称和语气词
6. 修正任何语法或表达错误
请直接返回修改后的JSON数据，保持原有格式，只需更新"translated"字段的内容。`;
// -----------------------------------------

// --- 新增 rpm 默认值 ---
export const DEFAULT_rpm_TRANSLATION = 0; // 0 表示无限制
export const DEFAULT_rpm_AI_VISION_OCR = 0;
// ------------------------

// --- 新增：自定义 AI 视觉 OCR 服务商 ID (前端使用) ---
export const CUSTOM_AI_VISION_PROVIDER_ID_FRONTEND = 'custom_openai_vision';
// ----------------------------------------------------
