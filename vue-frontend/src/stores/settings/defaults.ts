/**
 * Settings Store 默认值定义
 * 包含所有设置的默认值
 */

import type {
  TextStyleSettings,
  BaiduOcrSettings,
  PaddleOcrVlSettings,
  AiVisionOcrSettings,
  TranslationServiceSettings,
  HqTranslationSettings,
  ProofreadingSettings,
  BoxExpandSettings,
  PreciseMaskSettings,
  TranslationSettings,
  ParallelSettings
} from '@/types/settings'
import {
  DEFAULT_FONT_FAMILY,
  DEFAULT_FILL_COLOR,
  DEFAULT_STROKE_ENABLED,
  DEFAULT_STROKE_COLOR,
  DEFAULT_STROKE_WIDTH,
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_SINGLE_BUBBLE_PROMPT,
  DEFAULT_SINGLE_BUBBLE_JSON_PROMPT,
  DEFAULT_HQ_TRANSLATE_PROMPT,
  DEFAULT_RPM_TRANSLATION,
  DEFAULT_RPM_AI_VISION_OCR,
  DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE,
  DEFAULT_TRANSLATION_MAX_RETRIES,
  DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  DEFAULT_PROOFREADING_MAX_RETRIES
} from '@/constants'

// ============================================================
// 默认值定义
// ============================================================

/** 默认文字样式设置 */
export const DEFAULT_TEXT_STYLE: TextStyleSettings = {
  fontSize: 25,
  autoFontSize: false,
  fontFamily: DEFAULT_FONT_FAMILY,
  layoutDirection: 'auto',
  textColor: '#000000',
  fillColor: DEFAULT_FILL_COLOR,
  strokeEnabled: DEFAULT_STROKE_ENABLED,
  strokeColor: DEFAULT_STROKE_COLOR,
  strokeWidth: DEFAULT_STROKE_WIDTH,
  inpaintMethod: 'solid',
  // 智能颜色识别默认关闭
  useAutoTextColor: false
}

/** 默认百度OCR设置 */
export const DEFAULT_BAIDU_OCR: BaiduOcrSettings = {
  apiKey: '',
  secretKey: '',
  version: 'standard',
  sourceLanguage: 'JAP'
}

/** 默认PaddleOCR-VL设置 */
export const DEFAULT_PADDLEOCR_VL: PaddleOcrVlSettings = {
  sourceLanguage: 'japanese'
}

/** 默认AI视觉OCR设置 */
export const DEFAULT_AI_VISION_OCR: AiVisionOcrSettings = {
  provider: 'gemini',
  apiKey: '',
  modelName: '',
  prompt: DEFAULT_AI_VISION_OCR_PROMPT,
  rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
  customBaseUrl: '',
  isJsonMode: false,
  minImageSize: DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE
}

/** 默认翻译服务设置 */
export const DEFAULT_TRANSLATION_SERVICE: TranslationServiceSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  rpmLimit: DEFAULT_RPM_TRANSLATION,
  maxRetries: DEFAULT_TRANSLATION_MAX_RETRIES,
  isJsonMode: false,
  translationMode: 'batch',  // 默认使用整页批量翻译
  // 4个独立的提示词存储
  batchNormalPrompt: DEFAULT_TRANSLATE_PROMPT,
  batchJsonPrompt: DEFAULT_TRANSLATE_JSON_PROMPT,
  singleNormalPrompt: DEFAULT_SINGLE_BUBBLE_PROMPT,
  singleJsonPrompt: DEFAULT_SINGLE_BUBBLE_JSON_PROMPT
}

/** 默认高质量翻译设置 */
export const DEFAULT_HQ_TRANSLATION: HqTranslationSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  batchSize: 3,
  sessionReset: 3,
  rpmLimit: 7,
  maxRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  lowReasoning: false,
  noThinkingMethod: 'gemini',
  forceJsonOutput: false,
  useStream: false,
  prompt: DEFAULT_HQ_TRANSLATE_PROMPT
}

/** 默认AI校对设置 */
export const DEFAULT_PROOFREADING: ProofreadingSettings = {
  enabled: false,
  rounds: [],
  maxRetries: DEFAULT_PROOFREADING_MAX_RETRIES
}

/** 默认文本框扩展参数 */
export const DEFAULT_BOX_EXPAND: BoxExpandSettings = {
  ratio: 0,
  top: 0,
  bottom: 0,
  left: 0,
  right: 0
}

/** 默认精确文字掩膜设置 */
export const DEFAULT_PRECISE_MASK: PreciseMaskSettings = {
  enabled: true,
  dilateSize: 10,
  boxExpandRatio: 20
}

/** 默认并行翻译设置 */
export const DEFAULT_PARALLEL: ParallelSettings = {
  enabled: false,
  deepLearningLockSize: 1
}

/** 创建默认翻译设置 */
export function createDefaultSettings(): TranslationSettings {
  return {
    textStyle: { ...DEFAULT_TEXT_STYLE },
    ocrEngine: 'manga_ocr',
    sourceLanguage: 'japanese',
    textDetector: 'default',
    baiduOcr: { ...DEFAULT_BAIDU_OCR },
    paddleOcrVl: { ...DEFAULT_PADDLEOCR_VL },
    aiVisionOcr: { ...DEFAULT_AI_VISION_OCR },
    translation: { ...DEFAULT_TRANSLATION_SERVICE },
    targetLanguage: 'zh',
    translatePrompt: DEFAULT_TRANSLATE_PROMPT,
    useTextboxPrompt: false,
    textboxPrompt: '',
    hqTranslation: { ...DEFAULT_HQ_TRANSLATION },
    proofreading: { ...DEFAULT_PROOFREADING },
    boxExpand: { ...DEFAULT_BOX_EXPAND },
    preciseMask: { ...DEFAULT_PRECISE_MASK },
    pdfProcessingMethod: 'backend',
    showDetectionDebug: false,
    parallel: { ...DEFAULT_PARALLEL },
    autoSaveInBookshelfMode: false,
    removeTextWithOcr: false,
    enableVerboseLogs: false  // 默认关闭详细日志
  }
}
