import type {
  BatchConfig,
  StoreEmbeddingConfig,
  StoreImageGenConfig,
  StoreInsightConfig,
  StoreLlmConfig,
  StoreRerankerConfig,
  StoreVlmConfig,
} from '@/types/insight'

export const INSIGHT_CONFIG_STORAGE_KEY = 'manga_insight_config'
const INSIGHT_CONFIG_SCHEMA_VERSION = 1

type InsightConfigStoragePayload = {
  insightConfigSchemaVersion: typeof INSIGHT_CONFIG_SCHEMA_VERSION
  config: StoreInsightConfig
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasStringField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'string'
}

function hasOptionalStringField(value: Record<string, unknown>, key: string): boolean {
  return value[key] === undefined || typeof value[key] === 'string'
}

function hasNumberField(value: Record<string, unknown>, key: string): boolean {
  const field = value[key]
  return typeof field === 'number' && Number.isFinite(field)
}

function hasOptionalNumberField(value: Record<string, unknown>, key: string): boolean {
  return value[key] === undefined || hasNumberField(value, key)
}

function hasBooleanField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'boolean'
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return isRecord(value) && Object.values(value).every(entry => typeof entry === 'string')
}

function isOpenAiOptions(value: unknown): boolean {
  if (!isRecord(value) || !isRecord(value.request) || !isRecord(value.execution)) return false
  if (!hasBooleanField(value.request, 'forceJsonOutput')) return false
  if (!hasOptionalNumberField(value.request, 'temperature')) return false
  if (value.request.extraBody !== undefined && !isRecord(value.request.extraBody)) return false
  return (
    hasBooleanField(value.execution, 'useStream') &&
    hasNumberField(value.execution, 'rpmLimit') &&
    hasNumberField(value.execution, 'transportRetries') &&
    hasNumberField(value.execution, 'businessRetries')
  )
}

function isStoreVlmConfig(value: unknown): value is StoreVlmConfig {
  return (
    isRecord(value) &&
    hasStringField(value, 'provider') &&
    hasStringField(value, 'apiKey') &&
    hasStringField(value, 'model') &&
    hasOptionalStringField(value, 'baseUrl') &&
    isOpenAiOptions(value.openaiOptions) &&
    hasOptionalNumberField(value, 'imageMaxSize')
  )
}

function isStoreLlmConfig(value: unknown): value is StoreLlmConfig {
  return (
    isRecord(value) &&
    hasBooleanField(value, 'useSameAsVlm') &&
    hasStringField(value, 'provider') &&
    hasStringField(value, 'apiKey') &&
    hasStringField(value, 'model') &&
    hasStringField(value, 'baseUrl') &&
    isOpenAiOptions(value.openaiOptions)
  )
}

function isStoreEmbeddingConfig(value: unknown): value is StoreEmbeddingConfig {
  return (
    isRecord(value) &&
    hasStringField(value, 'provider') &&
    hasStringField(value, 'apiKey') &&
    hasStringField(value, 'model') &&
    hasOptionalStringField(value, 'baseUrl') &&
    hasOptionalNumberField(value, 'rpmLimit') &&
    hasOptionalNumberField(value, 'transportRetries') &&
    hasOptionalNumberField(value, 'businessRetries') &&
    hasOptionalNumberField(value, 'timeoutSeconds')
  )
}

function isStoreRerankerConfig(value: unknown): value is StoreRerankerConfig {
  return (
    isRecord(value) &&
    hasStringField(value, 'provider') &&
    hasStringField(value, 'apiKey') &&
    hasStringField(value, 'model') &&
    hasOptionalStringField(value, 'baseUrl') &&
    hasOptionalNumberField(value, 'topK') &&
    hasOptionalNumberField(value, 'transportRetries') &&
    hasOptionalNumberField(value, 'businessRetries') &&
    hasOptionalNumberField(value, 'timeoutSeconds')
  )
}

function isStoreImageGenConfig(value: unknown): value is StoreImageGenConfig {
  return (
    isRecord(value) &&
    hasStringField(value, 'provider') &&
    hasStringField(value, 'apiKey') &&
    hasStringField(value, 'model') &&
    hasOptionalStringField(value, 'baseUrl') &&
    hasOptionalNumberField(value, 'transportRetries') &&
    hasOptionalNumberField(value, 'businessRetries') &&
    hasOptionalNumberField(value, 'timeoutSeconds')
  )
}

function isBatchConfig(value: unknown): value is BatchConfig {
  return (
    isRecord(value) &&
    hasNumberField(value, 'pagesPerBatch') &&
    hasNumberField(value, 'contextBatchCount') &&
    hasStringField(value, 'architecturePreset') &&
    Array.isArray(value.customLayers) &&
    value.customLayers.every(layer => (
      isRecord(layer) &&
      hasStringField(layer, 'name') &&
      hasNumberField(layer, 'units') &&
      hasBooleanField(layer, 'align')
    ))
  )
}

export function buildInsightConfigStoragePayload(
  config: StoreInsightConfig,
): InsightConfigStoragePayload {
  return {
    insightConfigSchemaVersion: INSIGHT_CONFIG_SCHEMA_VERSION,
    config,
  }
}

export function parseInsightConfigStorage(value: unknown): StoreInsightConfig | null {
  if (!isRecord(value)) return null
  if (value.insightConfigSchemaVersion !== INSIGHT_CONFIG_SCHEMA_VERSION) return null
  if (!isRecord(value.config)) return null

  const config = value.config
  if (!isStoreVlmConfig(config.vlm)) return null
  if (!isStoreLlmConfig(config.llm)) return null
  if (!isStoreEmbeddingConfig(config.embedding)) return null
  if (!isStoreRerankerConfig(config.reranker)) return null
  if (!isStoreImageGenConfig(config.imageGen)) return null
  if (!isBatchConfig(config.batch)) return null
  if (!isStringRecord(config.prompts)) return null

  return (value as InsightConfigStoragePayload).config
}
