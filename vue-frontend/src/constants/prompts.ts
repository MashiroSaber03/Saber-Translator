import promptDefaults from '../../../src/shared/prompt_defaults_factory.json'
import type { PaddleOcrVlLanguage } from '@/types/settings'

export const DEFAULT_AI_VISION_OCR_PROMPT = promptDefaults.aiVisionOcrNormal
export const DEFAULT_SINGLE_BUBBLE_PROMPT = promptDefaults.singleNormal
export const DEFAULT_SINGLE_BUBBLE_JSON_PROMPT = promptDefaults.singleJson
export const DEFAULT_TRANSLATE_PROMPT = promptDefaults.batchNormal
export const DEFAULT_TRANSLATE_JSON_PROMPT = promptDefaults.batchJson
export const DEFAULT_AI_VISION_OCR_JSON_PROMPT = promptDefaults.aiVisionOcrJson

export const PADDLEOCR_VL_DEFAULT_LANGUAGE: PaddleOcrVlLanguage = 'japanese'

export const PADDLEOCR_VL_LANG_MAP = {
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
} as const satisfies Record<PaddleOcrVlLanguage, string>

export function isPaddleOcrVlLanguage(value: unknown): value is PaddleOcrVlLanguage {
  return typeof value === 'string' && Object.hasOwn(PADDLEOCR_VL_LANG_MAP, value)
}

export function getPaddleOcrVlPrompt(
  sourceLanguage: PaddleOcrVlLanguage = PADDLEOCR_VL_DEFAULT_LANGUAGE,
): string {
  return `对图中的${PADDLEOCR_VL_LANG_MAP[sourceLanguage]}进行OCR:`
}

export function inferPaddleOcrVlPromptLanguage(
  prompt: string,
  fallback: PaddleOcrVlLanguage = PADDLEOCR_VL_DEFAULT_LANGUAGE,
): PaddleOcrVlLanguage {
  const normalizedPrompt = prompt.trim()
  const matched = (Object.keys(PADDLEOCR_VL_LANG_MAP) as PaddleOcrVlLanguage[]).find(
    language => normalizedPrompt === getPaddleOcrVlPrompt(language),
  )
  return matched ?? fallback
}

export const DEFAULT_HQ_TRANSLATE_PROMPT = promptDefaults.hqTranslation
export const DEFAULT_PROOFREADING_PROMPT = promptDefaults.proofreading
export const DEFAULT_AUTO_GLOSSARY_PROMPT = promptDefaults.autoGlossary
export const DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT = promptDefaults.webImportExtraction
