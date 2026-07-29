import type {
  HqTranslationProvider,
  OcrEngine,
  PluginAgentProvider,
  TextDetector,
  TranslationProvider,
} from './settingsProviders'
import type { OpenAICompatibleOptions } from './openaiSettings'
import type {
  AiVisionOcrSettings,
  BaiduOcrSettings,
  HybridOcrSettings,
  PaddleOcrVlSettings,
} from './ocrSettings'
import type { TextStyleSettings } from './textStyleSettings'

export type TranslationMode = 'batch' | 'single'

export interface TranslationServiceSettings {
  provider: TranslationProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  openaiOptions: OpenAICompatibleOptions
  translationMode: TranslationMode
  batchNormalPrompt: string
  batchJsonPrompt: string
  singleNormalPrompt: string
  singleJsonPrompt: string
}

export interface HqTranslationSettings {
  provider: HqTranslationProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  openaiOptions: OpenAICompatibleOptions
  batchSize: number
  prompt: string
}

export interface PluginAgentSettings {
  provider: PluginAgentProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  openaiOptions: OpenAICompatibleOptions
}

export interface ProofreadingRound {
  name: string
  provider: HqTranslationProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  openaiOptions: OpenAICompatibleOptions
  batchSize: number
  prompt: string
}

export interface ProofreadingSettings {
  enabled: boolean
  rounds: ProofreadingRound[]
  maxRetries: number
}

export interface BoxExpandSettings {
  ratio: number
  top: number
  bottom: number
  left: number
  right: number
}

export interface PreciseMaskSettings {
  dilateSize: number
  boxExpandRatio: number
}

export interface ParallelSettings {
  enabled: boolean
  deepLearningLockSize: number
}

export interface TranslationSettings {
  settingsSchemaVersion: number
  textStyle: TextStyleSettings
  ocrEngine: OcrEngine
  sourceLanguage: string
  textDetector: TextDetector
  minTextBlockAreaPercent: number
  enableAuxYoloDetection: boolean
  auxYoloConfThreshold: number
  auxYoloOverlapThreshold: number
  enableSaberYoloRefine: boolean
  saberYoloRefineOverlapThreshold: number
  baiduOcr: BaiduOcrSettings
  paddleOcrVl: PaddleOcrVlSettings
  aiVisionOcr: AiVisionOcrSettings
  hybridOcr: HybridOcrSettings
  translation: TranslationServiceSettings
  targetLanguage: string
  translatePrompt: string
  useTextboxPrompt: boolean
  textboxPrompt: string
  hqTranslation: HqTranslationSettings
  pluginAgent: PluginAgentSettings
  proofreading: ProofreadingSettings
  boxExpand: BoxExpandSettings
  preciseMask: PreciseMaskSettings
  showDetectionDebug: boolean
  parallel: ParallelSettings
  removeTextWithOcr: boolean
  enableVerboseLogs: boolean
  lamaDisableResize: boolean
}

export type TranslationSettingsUpdates = Partial<TranslationSettings>
