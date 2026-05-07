/**
 * Settings Store 默认值定义
 * 包含所有设置的默认值
 */

import type {
  TextStyleSettings,
  BaiduOcrSettings,
  PaddleOcrVlSettings,
  AiVisionOcrSettings,
  HybridOcrSettings,
  TranslationServiceSettings,
  HqTranslationSettings,
  PluginAgentSettings,
  ProofreadingSettings,
  BoxExpandSettings,
  PreciseMaskSettings,
  TranslationSettings,
  ParallelSettings
} from '@/types/settings'
import type {
  GlossarySettings,
  NonTranslateSettings,
} from '@/types/translationConstraints'
import { getTextStyleDefaults } from '@/defaults/textStyleDefaults'
import {
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
import { createDefaultOpenAiOptions } from '@/utils/openaiOptions'

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

// ============================================================
// 默认值定义
// ============================================================

/** 默认文字样式设置 */
export function createDefaultTextStyle(): TextStyleSettings {
  const defaults = getTextStyleDefaults()
  return {
    fontSize: defaults.fontSize,
    autoFontSize: defaults.autoFontSize,
    fontFamily: defaults.fontFamily,
    layoutDirection: defaults.layoutDirection,
    textColor: defaults.textColor,
    fillColor: defaults.fillColor,
    strokeEnabled: defaults.strokeEnabled,
    strokeColor: defaults.strokeColor,
    strokeWidth: defaults.strokeWidth,
    inpaintMethod: defaults.inpaintMethod,
    // 智能颜色识别默认关闭
    useAutoTextColor: defaults.useAutoTextColor,
    // 排版设置
    lineSpacing: defaults.lineSpacing,
    textAlign: defaults.textAlign
  }
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
  promptMode: 'normal',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: false,
      rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
      transportRetries: 1,
      businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
    }
  }),
  minImageSize: DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE
}

/** 默认混合OCR设置 */
export const DEFAULT_HYBRID_OCR: HybridOcrSettings = {
  enabled: false,
  secondaryEngine: 'manga_ocr',
  confidenceThreshold: 0.2
}

/** 默认翻译服务设置 */
export const DEFAULT_TRANSLATION_SERVICE: TranslationServiceSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: false,
      rpmLimit: DEFAULT_RPM_TRANSLATION,
      transportRetries: 1,
      businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
    }
  }),
  translationMode: 'batch',  // 默认使用整页批量翻译
  // 4个独立的提示词存储
  batchNormalPrompt: DEFAULT_TRANSLATE_PROMPT,
  batchJsonPrompt: DEFAULT_TRANSLATE_JSON_PROMPT,
  singleNormalPrompt: DEFAULT_SINGLE_BUBBLE_PROMPT,
  singleJsonPrompt: DEFAULT_SINGLE_BUBBLE_JSON_PROMPT
}

/** 默认术语表设置 */
export const DEFAULT_GLOSSARY: GlossarySettings = {
  enabled: false,
  entries: []
}

/** 默认禁翻表设置 */
export const DEFAULT_NON_TRANSLATE: NonTranslateSettings = {
  enabled: false,
  entries: []
}

/** 默认高质量翻译设置 */
export const DEFAULT_HQ_TRANSLATION: HqTranslationSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: true,
      rpmLimit: 7,
      transportRetries: 1,
      businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
    }
  }),
  batchSize: 3,
  prompt: DEFAULT_HQ_TRANSLATE_PROMPT
}

/** 默认插件 Agent 设置 */
export const DEFAULT_PLUGIN_AGENT: PluginAgentSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: true,
      rpmLimit: 7,
      transportRetries: 1,
      businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
    }
  })
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
    settingsSchemaVersion: 3,
    textStyle: createDefaultTextStyle(),
    ocrEngine: 'manga_ocr',
    sourceLanguage: 'japanese',
    textDetector: 'default',
    enableAuxYoloDetection: false,
    auxYoloConfThreshold: 0.4,
    auxYoloOverlapThreshold: 0.1,
    enableSaberYoloRefine: true,
    saberYoloRefineOverlapThreshold: 50,
    baiduOcr: cloneJson(DEFAULT_BAIDU_OCR),
    paddleOcrVl: cloneJson(DEFAULT_PADDLEOCR_VL),
    aiVisionOcr: cloneJson(DEFAULT_AI_VISION_OCR),
    hybridOcr: cloneJson(DEFAULT_HYBRID_OCR),
    translation: cloneJson(DEFAULT_TRANSLATION_SERVICE),
    glossary: cloneJson(DEFAULT_GLOSSARY),
    nonTranslate: cloneJson(DEFAULT_NON_TRANSLATE),
    targetLanguage: 'zh',
    translatePrompt: DEFAULT_TRANSLATE_PROMPT,
    useTextboxPrompt: false,
    textboxPrompt: '',
    hqTranslation: cloneJson(DEFAULT_HQ_TRANSLATION),
    pluginAgent: cloneJson(DEFAULT_PLUGIN_AGENT),
    proofreading: cloneJson(DEFAULT_PROOFREADING),
    boxExpand: cloneJson(DEFAULT_BOX_EXPAND),
    preciseMask: cloneJson(DEFAULT_PRECISE_MASK),
    pdfProcessingMethod: 'backend',
    showDetectionDebug: false,
    parallel: cloneJson(DEFAULT_PARALLEL),
    autoSaveInBookshelfMode: false,
    removeTextWithOcr: false,
    enableVerboseLogs: false,  // 默认关闭详细日志
    lamaDisableResize: false  // 默认允许LAMA自动缩放（提高速度，减少显存占用）
  }
}
