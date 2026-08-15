import promptDefaults from '../../../src/shared/prompt_defaults_factory.json'

export const DEFAULT_AI_VISION_OCR_PROMPT = promptDefaults.aiVisionOcrNormal
export const DEFAULT_SINGLE_BUBBLE_PROMPT = promptDefaults.singleNormal
export const DEFAULT_SINGLE_BUBBLE_JSON_PROMPT = promptDefaults.singleJson
export const DEFAULT_TRANSLATE_PROMPT = promptDefaults.batchNormal
export const DEFAULT_TRANSLATE_JSON_PROMPT = promptDefaults.batchJson
export const DEFAULT_AI_VISION_OCR_JSON_PROMPT = promptDefaults.aiVisionOcrJson

export const getPaddleOcrVlPrompt = (langName: string = '日语') => `对图中的${langName}进行OCR:`

export const PADDLEOCR_VL_LANG_MAP: Record<string, string> = {
  japanese: '日语',
  chinese: '简体中文',
  chinese_cht: '繁体中文',
  korean: '韩语',
  english: '英语',
  french: '法语',
  german: '德语',
  spanish: '西班牙语',
  italian: '意大利语',
  portuguese: '葡萄牙语',
  dutch: '荷兰语',
  polish: '波兰语',
  thai: '泰语',
  vietnamese: '越南语',
  indonesian: '印尼语',
  malay: '马来语',
  russian: '俄语',
  arabic: '阿拉伯语',
  hindi: '印地语',
  turkish: '土耳其语',
  greek: '希腊语',
  hebrew: '希伯来语',
}

export function inferPaddleOcrVlPromptLanguage(
  prompt: string,
  fallback = 'japanese',
): string {
  const normalizedPrompt = prompt.trim()
  const matched = Object.entries(PADDLEOCR_VL_LANG_MAP).find(
    ([, languageName]) => normalizedPrompt === getPaddleOcrVlPrompt(languageName),
  )
  return matched?.[0] ?? fallback
}

export const DEFAULT_HQ_TRANSLATE_PROMPT = promptDefaults.hqTranslation
export const DEFAULT_PROOFREADING_PROMPT = promptDefaults.proofreading
export const DEFAULT_AUTO_GLOSSARY_PROMPT = promptDefaults.autoGlossary
export const DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT = promptDefaults.webImportExtraction
