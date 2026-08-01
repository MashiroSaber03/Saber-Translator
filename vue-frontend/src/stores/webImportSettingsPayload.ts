import type {
  WebImportAgentProviderConfig,
  WebImportProviderConfigs,
  WebImportSettings,
  WebImportSettingsPayload,
} from '@/types/webImport'
import { deepClone } from '@/utils/deepClone'
import {
  isWebImportAgentProvider,
} from './settings/modules/webImport'

export const WEB_IMPORT_SETTINGS_SCHEMA_VERSION = 1

type PlainRecord = Record<string, unknown>

export function serializeWebImportSettingsValue(value: unknown): string {
  return JSON.stringify(value)
}

function isPlainRecord(value: unknown): value is PlainRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function hasExactKeys(value: PlainRecord, keys: readonly string[]): boolean {
  const actualKeys = Object.keys(value)
  return actualKeys.length === keys.length
    && keys.every(key => Object.prototype.hasOwnProperty.call(value, key))
}

function parseString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function parseNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function parseImageFormat(value: unknown): WebImportSettings['imagePreprocess']['formatConvert']['targetFormat'] | null {
  return value === 'jpeg' || value === 'png' || value === 'webp' || value === 'original'
    ? value
    : null
}

function parseCurrentWebImportSettings(value: unknown): WebImportSettings | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'firecrawl',
    'agent',
    'extraction',
    'download',
    'imagePreprocess',
    'advanced',
    'ui',
  ])) {
    return null
  }

  const {
    firecrawl,
    agent,
    extraction,
    download,
    imagePreprocess,
    advanced,
    ui,
  } = value

  if (
    !isPlainRecord(firecrawl) ||
    !isPlainRecord(agent) ||
    !isPlainRecord(extraction) ||
    !isPlainRecord(download) ||
    !isPlainRecord(imagePreprocess) ||
    !isPlainRecord(advanced) ||
    !isPlainRecord(ui)
  ) {
    return null
  }

  if (
    !hasExactKeys(firecrawl, ['apiKey']) ||
    !hasExactKeys(agent, ['provider', 'apiKey', 'customBaseUrl', 'modelName', 'useStream', 'forceJsonOutput', 'maxRetries', 'timeout']) ||
    !hasExactKeys(extraction, ['prompt', 'maxIterations']) ||
    !hasExactKeys(download, ['concurrency', 'timeout', 'retries', 'delay', 'useReferer']) ||
    !hasExactKeys(imagePreprocess, ['enabled', 'autoRotate', 'compression', 'formatConvert']) ||
    !hasExactKeys(advanced, ['customCookie', 'customHeaders', 'bypassProxy']) ||
    !hasExactKeys(ui, ['showAgentLogs', 'autoImport'])
  ) {
    return null
  }

  if (!isPlainRecord(imagePreprocess.compression) || !isPlainRecord(imagePreprocess.formatConvert)) {
    return null
  }
  const compression = imagePreprocess.compression
  const formatConvert = imagePreprocess.formatConvert
  if (
    !hasExactKeys(compression, ['enabled', 'quality', 'maxWidth', 'maxHeight']) ||
    !hasExactKeys(formatConvert, ['enabled', 'targetFormat'])
  ) {
    return null
  }

  const provider = parseString(agent.provider)
  const targetFormat = parseImageFormat(formatConvert.targetFormat)
  if (!isWebImportAgentProvider(provider) || !targetFormat) {
    return null
  }

  const parsed = {
    firecrawl: {
      apiKey: parseString(firecrawl.apiKey),
    },
    agent: {
      provider,
      apiKey: parseString(agent.apiKey),
      customBaseUrl: parseString(agent.customBaseUrl),
      modelName: parseString(agent.modelName),
      useStream: parseBoolean(agent.useStream),
      forceJsonOutput: parseBoolean(agent.forceJsonOutput),
      maxRetries: parseNumber(agent.maxRetries),
      timeout: parseNumber(agent.timeout),
    },
    extraction: {
      prompt: parseString(extraction.prompt),
      maxIterations: parseNumber(extraction.maxIterations),
    },
    download: {
      concurrency: parseNumber(download.concurrency),
      timeout: parseNumber(download.timeout),
      retries: parseNumber(download.retries),
      delay: parseNumber(download.delay),
      useReferer: parseBoolean(download.useReferer),
    },
    imagePreprocess: {
      enabled: parseBoolean(imagePreprocess.enabled),
      autoRotate: parseBoolean(imagePreprocess.autoRotate),
      compression: {
        enabled: parseBoolean(compression.enabled),
        quality: parseNumber(compression.quality),
        maxWidth: parseNumber(compression.maxWidth),
        maxHeight: parseNumber(compression.maxHeight),
      },
      formatConvert: {
        enabled: parseBoolean(formatConvert.enabled),
        targetFormat,
      },
    },
    advanced: {
      customCookie: parseString(advanced.customCookie),
      customHeaders: parseString(advanced.customHeaders),
      bypassProxy: parseBoolean(advanced.bypassProxy),
    },
    ui: {
      showAgentLogs: parseBoolean(ui.showAgentLogs),
      autoImport: parseBoolean(ui.autoImport),
    },
  }

  if (
    parsed.firecrawl.apiKey === null ||
    parsed.agent.apiKey === null ||
    parsed.agent.customBaseUrl === null ||
    parsed.agent.modelName === null ||
    parsed.agent.useStream === null ||
    parsed.agent.forceJsonOutput === null ||
    parsed.agent.maxRetries === null ||
    parsed.agent.timeout === null ||
    parsed.extraction.prompt === null ||
    parsed.extraction.maxIterations === null ||
    parsed.download.concurrency === null ||
    parsed.download.timeout === null ||
    parsed.download.retries === null ||
    parsed.download.delay === null ||
    parsed.download.useReferer === null ||
    parsed.imagePreprocess.enabled === null ||
    parsed.imagePreprocess.autoRotate === null ||
    parsed.imagePreprocess.compression.enabled === null ||
    parsed.imagePreprocess.compression.quality === null ||
    parsed.imagePreprocess.compression.maxWidth === null ||
    parsed.imagePreprocess.compression.maxHeight === null ||
    parsed.imagePreprocess.formatConvert.enabled === null ||
    parsed.advanced.customCookie === null ||
    parsed.advanced.customHeaders === null ||
    parsed.advanced.bypassProxy === null ||
    parsed.ui.showAgentLogs === null ||
    parsed.ui.autoImport === null
  ) {
    return null
  }

  return parsed as WebImportSettings
}

function parseCurrentAgentProviderConfig(value: unknown): WebImportAgentProviderConfig | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['apiKey', 'modelName', 'customBaseUrl'])) {
    return null
  }
  const apiKey = parseString(value.apiKey)
  const modelName = parseString(value.modelName)
  const customBaseUrl = parseString(value.customBaseUrl)
  if (apiKey === null || modelName === null || customBaseUrl === null) {
    return null
  }
  return { apiKey, modelName, customBaseUrl }
}

function parseCurrentProviderConfigs(value: unknown): WebImportProviderConfigs | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['agent']) || !isPlainRecord(value.agent)) {
    return null
  }

  const agent: WebImportProviderConfigs['agent'] = {}
  for (const [provider, config] of Object.entries(value.agent)) {
    if (!isWebImportAgentProvider(provider)) {
      continue
    }
    const parsed = parseCurrentAgentProviderConfig(config)
    if (parsed) {
      agent[provider] = parsed
    }
  }
  return { agent }
}

function buildWebImportSettingsPayload(
  settings: WebImportSettings,
  providerConfigs: WebImportProviderConfigs
): WebImportSettingsPayload {
  return {
    webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
    settings: deepClone(settings),
    providerConfigs: deepClone(providerConfigs),
  }
}

export function parseWebImportSettingsPayload(value: unknown): WebImportSettingsPayload | null {
  if (!isPlainRecord(value)) return null
  const settings = parseCurrentWebImportSettings(value.settings)
  const providerConfigs = parseCurrentProviderConfigs(value.providerConfigs)
  if (!settings || !providerConfigs) return null
  return buildWebImportSettingsPayload(settings, providerConfigs)
}
