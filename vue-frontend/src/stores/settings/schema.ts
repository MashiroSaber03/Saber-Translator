import {
  getProviderManifest,
  normalizeProviderId,
  providerSupportsCapability,
  type ProviderCapability,
} from '@/config/aiProviders'
import type {
  HqTranslationProvider,
  OpenAICompatibleOptions,
  TranslationMode,
  TranslationSettings,
} from '@/types/settings'
import { deepClone } from '@/utils/deepClone'
import type { WorkflowMode } from '@/types/workflow'

import {
  TRANSLATION_SETTINGS_SCHEMA_VERSION,
  createDefaultSettings,
} from './defaults'
import { isProofreadingRoundId } from './proofreadingIdentity'

type PlainRecord = Record<string, unknown>
type AiVisionPromptMode = TranslationSettings['aiVisionOcr']['promptMode']

export interface CurrentWorkflowPreferences {
  rememberWorkflowModeEnabled: boolean
  lastWorkflowMode: WorkflowMode
}

const CURRENT_WORKFLOW_MODES: ReadonlySet<string> = new Set([
  'translate-current',
  'translate-batch',
  'hq-batch',
  'proofread-batch',
  'remove-current',
  'remove-batch',
  'retry-failed',
  'delete-current',
  'clear-all',
])

function isPlainRecord(value: unknown): value is PlainRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function hasExactKeys(value: PlainRecord, keys: readonly string[]): boolean {
  const actual = Object.keys(value)
  return actual.length === keys.length && keys.every(key => Object.hasOwn(value, key))
}

function parseNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseInteger(
  value: unknown,
  minimum: number,
  maximum: number,
): number | null {
  return Number.isInteger(value) && Number(value) >= minimum && Number(value) <= maximum
    ? Number(value)
    : null
}

function parseBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function parseString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function isTranslationMode(value: unknown): value is TranslationMode {
  return value === 'batch' || value === 'single'
}

function isAiVisionPromptMode(value: unknown): value is AiVisionPromptMode {
  return value === 'normal' || value === 'json' || value === 'paddleocr_vl'
}

function parseCurrentOpenAiOptions(
  value: unknown,
): OpenAICompatibleOptions | null {
  if (!isPlainRecord(value) || !isPlainRecord(value.request) || !isPlainRecord(value.execution)) {
    return null
  }
  if (!hasExactKeys(value, ['request', 'execution'])) return null
  const requestKeys = Object.keys(value.request)
  if (
    !Object.hasOwn(value.request, 'forceJsonOutput')
    || requestKeys.some(key => !['forceJsonOutput', 'temperature', 'extraBody'].includes(key))
    || !hasExactKeys(value.execution, [
      'useStream',
      'rpmLimit',
      'transportRetries',
      'businessRetries',
    ])
  ) {
    return null
  }

  const forceJsonOutput = parseBoolean(value.request.forceJsonOutput)
  const useStream = parseBoolean(value.execution.useStream)
  const rpmLimit = parseInteger(value.execution.rpmLimit, 0, 100_000)
  const transportRetries = parseInteger(value.execution.transportRetries, 0, 100)
  const businessRetries = parseInteger(value.execution.businessRetries, 0, 100)
  if (
    forceJsonOutput === null
    || useStream === null
    || rpmLimit === null
    || transportRetries === null
    || businessRetries === null
  ) {
    return null
  }

  const request: OpenAICompatibleOptions['request'] = { forceJsonOutput }
  if (value.request.temperature !== undefined) {
    const temperature = parseNumber(value.request.temperature)
    if (temperature === null || temperature < 0 || temperature > 2) return null
    request.temperature = temperature
  }
  if (value.request.extraBody !== undefined) {
    if (!isPlainRecord(value.request.extraBody)) return null
    request.extraBody = deepClone(value.request.extraBody)
  }

  return {
    request,
    execution: {
      useStream,
      rpmLimit,
      transportRetries,
      businessRetries,
    },
  }
}

function sanitizeByTemplate(value: unknown, template: unknown, path = ''): unknown | null {
  if (path.endsWith('.openaiOptions')) {
    return parseCurrentOpenAiOptions(value)
  }
  if (Array.isArray(template)) {
    return Array.isArray(value) ? deepClone(value) : null
  }
  if (isPlainRecord(template)) {
    if (!isPlainRecord(value)) return null
    if (!hasExactKeys(value, Object.keys(template))) return null
    const result: PlainRecord = {}
    for (const key of Object.keys(template)) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) return null
      const sanitized = sanitizeByTemplate(value[key], template[key], `${path}.${key}`)
      if (sanitized === null) return null
      result[key] = sanitized
    }
    return result
  }
  if (typeof template === 'number') return parseNumber(value)
  if (typeof template === 'boolean') return parseBoolean(value)
  if (typeof template === 'string') return parseString(value)
  return value === template ? value : null
}

function isCurrentProviderId(provider: unknown): provider is string {
  return typeof provider === 'string'
    && provider === normalizeProviderId(provider)
    && Boolean(getProviderManifest(provider))
}

function isCurrentProviderForCapability(
  provider: unknown,
  capability: ProviderCapability,
): provider is string {
  return isCurrentProviderId(provider)
    && providerSupportsCapability(provider, capability)
}

function isFiniteRange(value: number, minimum: number, maximum: number): boolean {
  return Number.isFinite(value) && value >= minimum && value <= maximum
}

function isIntegerRange(value: number, minimum: number, maximum?: number): boolean {
  return Number.isInteger(value)
    && value >= minimum
    && (maximum === undefined || value <= maximum)
}

function sanitizeProofreadingRounds(value: unknown): TranslationSettings['proofreading']['rounds'] | null {
  if (!Array.isArray(value)) return null
  const rounds: TranslationSettings['proofreading']['rounds'] = []
  const roundIds = new Set<string>()
  for (const round of value) {
    if (
      !isPlainRecord(round)
      || !isCurrentProviderForCapability(round.provider, 'hqTranslation')
    ) return null
    if (!hasExactKeys(round, [
      'id',
      'name',
      'provider',
      'apiKey',
      'modelName',
      'customBaseUrl',
      'openaiOptions',
      'batchSize',
      'prompt',
    ])) return null
    if (!isProofreadingRoundId(round.id)) return null
    if (roundIds.has(round.id)) return null
    roundIds.add(round.id)
    const name = parseString(round.name)
    const apiKey = parseString(round.apiKey)
    const modelName = parseString(round.modelName)
    const customBaseUrl = parseString(round.customBaseUrl)
    const prompt = parseString(round.prompt)
    const batchSize = parseInteger(round.batchSize, 1, 10)
    const openaiOptions = parseCurrentOpenAiOptions(
      round.openaiOptions,
    )
    if (
      name === null
      || apiKey === null
      || modelName === null
      || customBaseUrl === null
      || prompt === null
      || batchSize === null
      || openaiOptions === null
    ) {
      return null
    }
    rounds.push({
      id: round.id,
      name,
      provider: round.provider as HqTranslationProvider,
      apiKey,
      modelName,
      customBaseUrl,
      openaiOptions,
      batchSize,
      prompt,
    })
  }
  return rounds
}

export function parseCurrentSettings(value: unknown): TranslationSettings | null {
  if (
    !isPlainRecord(value)
    || value.settingsSchemaVersion !== TRANSLATION_SETTINGS_SCHEMA_VERSION
  ) return null
  const sanitized = sanitizeByTemplate(value, createDefaultSettings()) as TranslationSettings | null
  if (!sanitized) return null
  if (!isCurrentProviderForCapability(sanitized.translation.provider, 'translation')) return null
  if (!isCurrentProviderForCapability(sanitized.hqTranslation.provider, 'hqTranslation')) return null
  if (!isCurrentProviderForCapability(sanitized.pluginAgent.provider, 'pluginAgent')) return null
  if (!isCurrentProviderForCapability(sanitized.aiVisionOcr.provider, 'visionOcr')) return null
  if (!isTranslationMode(sanitized.translation.translationMode)) return null
  if (!isAiVisionPromptMode(sanitized.aiVisionOcr.promptMode)) return null
  if (![
    'manga_ocr',
    'paddle_ocr',
    'paddleocr_vl',
    'baidu_ocr',
    'ai_vision',
    '48px_ocr',
  ].includes(sanitized.ocrEngine)) return null
  if (!['ctd', 'yolo', 'default'].includes(sanitized.textDetector)) return null
  if (!['standard', 'high_precision'].includes(sanitized.baiduOcr.version)) return null
  if (![
    'auto_detect',
    'CHN_ENG',
    'ENG',
    'JAP',
    'KOR',
    'FRE',
    'GER',
    'RUS',
  ].includes(sanitized.baiduOcr.sourceLanguage)) return null
  if (!['manga_ocr', '48px_ocr'].includes(sanitized.hybridOcr.secondaryEngine)) return null
  if (
    sanitized.hybridOcr.enabled
    && (
      !['manga_ocr', '48px_ocr'].includes(sanitized.ocrEngine)
      || sanitized.hybridOcr.secondaryEngine === sanitized.ocrEngine
    )
  ) return null
  if (!isFiniteRange(sanitized.minTextBlockAreaPercent, 0, 100)) return null
  if (!isFiniteRange(sanitized.auxYoloConfThreshold, 0, 1)) return null
  if (!isFiniteRange(sanitized.auxYoloOverlapThreshold, 0, 1)) return null
  if (!isFiniteRange(sanitized.saberYoloRefineOverlapThreshold, 0, 100)) return null
  if (!isFiniteRange(sanitized.hybridOcr.confidenceThreshold, 0, 1)) return null
  if (!Object.values(sanitized.boxExpand).every(value => isFiniteRange(value, 0, 50))) {
    return null
  }
  if (!isIntegerRange(sanitized.preciseMask.dilateSize, 0)) return null
  if (!isFiniteRange(sanitized.preciseMask.boxExpandRatio, 0, 100)) return null
  if (!isIntegerRange(sanitized.aiVisionOcr.minImageSize, 0)) return null
  if (!isIntegerRange(sanitized.parallel.deepLearningLockSize, 1)) return null
  if (!isIntegerRange(sanitized.hqTranslation.batchSize, 1, 10)) return null
  const rounds = sanitizeProofreadingRounds((value.proofreading as PlainRecord).rounds)
  if (!rounds) return null
  sanitized.proofreading.rounds = rounds
  return sanitized
}

export function parseCurrentWorkflowPreferences(
  value: unknown,
): CurrentWorkflowPreferences | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'rememberWorkflowModeEnabled',
    'lastWorkflowMode',
  ])) {
    return null
  }
  if (
    typeof value.rememberWorkflowModeEnabled !== 'boolean'
    || typeof value.lastWorkflowMode !== 'string'
    || !CURRENT_WORKFLOW_MODES.has(value.lastWorkflowMode)
  ) {
    return null
  }
  return {
    rememberWorkflowModeEnabled: value.rememberWorkflowModeEnabled,
    lastWorkflowMode: value.lastWorkflowMode as WorkflowMode,
  }
}
