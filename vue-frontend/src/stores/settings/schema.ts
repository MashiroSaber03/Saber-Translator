import { getProviderManifest, normalizeProviderId } from '@/config/aiProviders'
import type {
  HqTranslationProvider,
  OpenAICompatibleOptions,
  TranslationMode,
  TranslationSettings,
} from '@/types/settings'
import { deepClone } from '@/utils/deepClone'

import { createDefaultSettings } from './defaults'

type PlainRecord = Record<string, unknown>
type AiVisionPromptMode = TranslationSettings['aiVisionOcr']['promptMode']

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
  defaults: OpenAICompatibleOptions,
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
  const rpmLimit = parseNumber(value.execution.rpmLimit)
  const transportRetries = parseNumber(value.execution.transportRetries)
  const businessRetries = parseNumber(value.execution.businessRetries)
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
    if (temperature === null) return null
    request.temperature = temperature
  } else if (defaults.request.temperature !== undefined) {
    request.temperature = defaults.request.temperature
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
    return parseCurrentOpenAiOptions(value, template as OpenAICompatibleOptions)
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

function sanitizeProofreadingRounds(value: unknown): TranslationSettings['proofreading']['rounds'] | null {
  if (!Array.isArray(value)) return null
  const rounds: TranslationSettings['proofreading']['rounds'] = []
  for (const round of value) {
    if (!isPlainRecord(round) || !isCurrentProviderId(round.provider)) return null
    if (!hasExactKeys(round, [
      'name',
      'provider',
      'apiKey',
      'modelName',
      'customBaseUrl',
      'openaiOptions',
      'batchSize',
      'prompt',
    ])) return null
    const name = parseString(round.name)
    const apiKey = parseString(round.apiKey)
    const modelName = parseString(round.modelName)
    const customBaseUrl = parseString(round.customBaseUrl)
    const prompt = parseString(round.prompt)
    const batchSize = parseNumber(round.batchSize)
    const openaiOptions = parseCurrentOpenAiOptions(
      round.openaiOptions,
      createDefaultSettings().hqTranslation.openaiOptions,
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
  if (!isPlainRecord(value) || value.settingsSchemaVersion !== 3) return null
  const sanitized = sanitizeByTemplate(value, createDefaultSettings()) as TranslationSettings | null
  if (!sanitized) return null
  if (!isCurrentProviderId(sanitized.translation.provider)) return null
  if (!isCurrentProviderId(sanitized.hqTranslation.provider)) return null
  if (!isCurrentProviderId(sanitized.pluginAgent.provider)) return null
  if (!isCurrentProviderId(sanitized.aiVisionOcr.provider)) return null
  if (!isTranslationMode(sanitized.translation.translationMode)) return null
  if (!isAiVisionPromptMode(sanitized.aiVisionOcr.promptMode)) return null
  const rounds = sanitizeProofreadingRounds((value.proofreading as PlainRecord).rounds)
  if (!rounds) return null
  sanitized.proofreading.rounds = rounds
  return sanitized
}
