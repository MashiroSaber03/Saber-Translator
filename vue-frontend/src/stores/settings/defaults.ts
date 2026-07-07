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
import { deepClone } from '@/utils/deepClone'

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
    useAutoTextColor: defaults.useAutoTextColor,
    lineSpacing: defaults.lineSpacing,
    textAlign: defaults.textAlign
  }
}

export const DEFAULT_BAIDU_OCR: BaiduOcrSettings = {
  apiKey: '',
  secretKey: '',
  version: 'standard',
  sourceLanguage: 'JAP'
}

export const DEFAULT_PADDLEOCR_VL: PaddleOcrVlSettings = {
  sourceLanguage: 'japanese'
}

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

export const DEFAULT_HYBRID_OCR: HybridOcrSettings = {
  enabled: false,
  secondaryEngine: '48px_ocr',
  confidenceThreshold: 0.2
}

export const DEFAULT_TRANSLATION_SERVICE: TranslationServiceSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: true,
      rpmLimit: DEFAULT_RPM_TRANSLATION,
      transportRetries: 1,
      businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
    }
  }),
  translationMode: 'batch',
  // Prompt variants stay separate so mode toggles do not overwrite user edits.
  batchNormalPrompt: DEFAULT_TRANSLATE_PROMPT,
  batchJsonPrompt: DEFAULT_TRANSLATE_JSON_PROMPT,
  singleNormalPrompt: DEFAULT_SINGLE_BUBBLE_PROMPT,
  singleJsonPrompt: DEFAULT_SINGLE_BUBBLE_JSON_PROMPT
}

export const DEFAULT_HQ_TRANSLATION: HqTranslationSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: true,
      rpmLimit: 7,
      transportRetries: 3,
      businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
    }
  }),
  batchSize: 3,
  prompt: DEFAULT_HQ_TRANSLATE_PROMPT
}

export const DEFAULT_PLUGIN_AGENT: PluginAgentSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  openaiOptions: createDefaultOpenAiOptions({
    execution: {
      useStream: true,
      rpmLimit: 0,
      transportRetries: 10,
      businessRetries: 10
    }
  })
}

export const DEFAULT_PROOFREADING: ProofreadingSettings = {
  enabled: false,
  rounds: [],
  maxRetries: DEFAULT_PROOFREADING_MAX_RETRIES
}

export const DEFAULT_BOX_EXPAND: BoxExpandSettings = {
  ratio: 0,
  top: 0,
  bottom: 0,
  left: 0,
  right: 0
}

export const DEFAULT_PRECISE_MASK: PreciseMaskSettings = {
  dilateSize: 10,
  boxExpandRatio: 20
}

export const DEFAULT_PARALLEL: ParallelSettings = {
  enabled: false,
  deepLearningLockSize: 1
}

export function createDefaultSettings(): TranslationSettings {
  return {
    settingsSchemaVersion: 3,
    textStyle: createDefaultTextStyle(),
    ocrEngine: 'manga_ocr',
    sourceLanguage: 'japanese',
    textDetector: 'default',
    minTextBlockAreaPercent: 0.05,
    enableAuxYoloDetection: false,
    auxYoloConfThreshold: 0.4,
    auxYoloOverlapThreshold: 0.1,
    enableSaberYoloRefine: true,
    saberYoloRefineOverlapThreshold: 50,
    baiduOcr: deepClone(DEFAULT_BAIDU_OCR),
    paddleOcrVl: deepClone(DEFAULT_PADDLEOCR_VL),
    aiVisionOcr: deepClone(DEFAULT_AI_VISION_OCR),
    hybridOcr: deepClone(DEFAULT_HYBRID_OCR),
    translation: deepClone(DEFAULT_TRANSLATION_SERVICE),
    targetLanguage: 'zh',
    translatePrompt: DEFAULT_TRANSLATE_PROMPT,
    useTextboxPrompt: false,
    textboxPrompt: '',
    hqTranslation: deepClone(DEFAULT_HQ_TRANSLATION),
    pluginAgent: deepClone(DEFAULT_PLUGIN_AGENT),
    proofreading: deepClone(DEFAULT_PROOFREADING),
    boxExpand: deepClone(DEFAULT_BOX_EXPAND),
    preciseMask: deepClone(DEFAULT_PRECISE_MASK),
    pdfProcessingMethod: 'backend',
    showDetectionDebug: false,
    parallel: deepClone(DEFAULT_PARALLEL),
    autoSaveInBookshelfMode: true,
    removeTextWithOcr: false,
    enableVerboseLogs: false,
    lamaDisableResize: false
  }
}
