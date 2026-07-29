export type OcrEngine =
  | 'manga_ocr'
  | 'paddle_ocr'
  | 'paddleocr_vl'
  | 'baidu_ocr'
  | 'ai_vision'
  | '48px_ocr'

export type HybridOcrEngine = Extract<OcrEngine, 'manga_ocr' | '48px_ocr'>

export type TextDetector = 'ctd' | 'yolo' | 'default'

export type TranslationProvider =
  | 'siliconflow'
  | 'deepseek'
  | 'volcano'
  | 'caiyun'
  | 'baidu_translate'
  | 'youdao_translate'
  | 'gemini'
  | 'ollama'
  | 'sakura'
  | 'custom'

export type HqTranslationProvider =
  | 'siliconflow'
  | 'deepseek'
  | 'volcano'
  | 'gemini'
  | 'ollama'
  | 'custom'

export type PluginAgentProvider = HqTranslationProvider
