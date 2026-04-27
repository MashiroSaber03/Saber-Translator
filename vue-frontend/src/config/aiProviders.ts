export type ProviderKind = 'openai_compatible' | 'local' | 'adapter'
export type ProviderCapability =
  | 'translation'
  | 'hqTranslation'
  | 'visionOcr'
  | 'modelFetch'
  | 'connectionTest'

export interface AiProviderManifestEntry {
  id: string
  label: string
  kind: ProviderKind
  defaultBaseUrl?: string
  capabilities: ProviderCapability[]
  requiresApiKey: boolean
  requiresModel: boolean
  requiresBaseUrl: boolean
  isLocal: boolean
  supportsStream: boolean
  supportsJsonResponse: boolean
  supportsReasoningControl: boolean
  legacyIds?: string[]
}

export const AI_PROVIDER_MANIFEST: AiProviderManifestEntry[] = [
  {
    id: 'siliconflow',
    label: 'SiliconFlow',
    kind: 'openai_compatible',
    defaultBaseUrl: 'https://api.siliconflow.cn/v1',
    capabilities: ['translation', 'hqTranslation', 'visionOcr', 'modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    kind: 'openai_compatible',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    capabilities: ['translation', 'hqTranslation', 'modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true
  },
  {
    id: 'volcano',
    label: '火山引擎',
    kind: 'openai_compatible',
    defaultBaseUrl: 'https://ark.cn-beijing.volces.com/api/v3',
    capabilities: ['translation', 'hqTranslation', 'visionOcr', 'modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true
  },
  {
    id: 'gemini',
    label: 'Google Gemini',
    kind: 'openai_compatible',
    defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    capabilities: ['translation', 'hqTranslation', 'visionOcr', 'modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true
  },
  {
    id: 'custom',
    label: '自定义 OpenAI 兼容服务',
    kind: 'openai_compatible',
    capabilities: ['translation', 'hqTranslation', 'visionOcr', 'modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: true,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true,
    legacyIds: ['custom_openai', 'custom_openai_vision']
  },
  {
    id: 'ollama',
    label: 'Ollama (本地)',
    kind: 'local',
    defaultBaseUrl: 'http://localhost:11434',
    capabilities: ['translation', 'modelFetch', 'connectionTest'],
    requiresApiKey: false,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: true,
    supportsStream: false,
    supportsJsonResponse: false,
    supportsReasoningControl: false
  },
  {
    id: 'sakura',
    label: 'Sakura (本地)',
    kind: 'local',
    defaultBaseUrl: 'http://localhost:8080/v1',
    capabilities: ['translation', 'modelFetch', 'connectionTest'],
    requiresApiKey: false,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: true,
    supportsStream: false,
    supportsJsonResponse: false,
    supportsReasoningControl: false
  },
  {
    id: 'caiyun',
    label: '彩云小译',
    kind: 'adapter',
    capabilities: ['translation', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: false,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: false,
    supportsJsonResponse: false,
    supportsReasoningControl: false
  },
  {
    id: 'baidu_translate',
    label: '百度翻译',
    kind: 'adapter',
    capabilities: ['translation', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: false,
    supportsJsonResponse: false,
    supportsReasoningControl: false
  },
  {
    id: 'youdao_translate',
    label: '有道翻译',
    kind: 'adapter',
    capabilities: ['translation', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: false,
    supportsJsonResponse: false,
    supportsReasoningControl: false
  },
  {
    id: 'openai',
    label: 'OpenAI',
    kind: 'openai_compatible',
    defaultBaseUrl: 'https://api.openai.com/v1',
    capabilities: ['modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true
  },
  {
    id: 'qwen',
    label: '通义千问',
    kind: 'openai_compatible',
    defaultBaseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    capabilities: ['modelFetch', 'connectionTest'],
    requiresApiKey: true,
    requiresModel: true,
    requiresBaseUrl: false,
    isLocal: false,
    supportsStream: true,
    supportsJsonResponse: true,
    supportsReasoningControl: true
  }
]

const LEGACY_PROVIDER_MAP = new Map(
  AI_PROVIDER_MANIFEST.flatMap(entry => (entry.legacyIds || []).map(legacyId => [legacyId, entry.id] as const))
)

const PROVIDER_MAP = new Map(AI_PROVIDER_MANIFEST.map(entry => [entry.id, entry] as const))

export function normalizeProviderId(provider?: string | null): string {
  if (!provider) return ''
  const normalized = String(provider).trim()
  return LEGACY_PROVIDER_MAP.get(normalized) || LEGACY_PROVIDER_MAP.get(normalized.toLowerCase()) || normalized
}

export function getProviderManifest(provider?: string | null): AiProviderManifestEntry | undefined {
  const normalized = normalizeProviderId(provider)
  return PROVIDER_MAP.get(normalized)
}

export function providerSupportsCapability(provider: string, capability: ProviderCapability): boolean {
  return Boolean(getProviderManifest(provider)?.capabilities.includes(capability))
}

export function providerRequiresBaseUrl(provider: string): boolean {
  return Boolean(getProviderManifest(provider)?.requiresBaseUrl)
}

export function providerRequiresApiKey(provider: string): boolean {
  return Boolean(getProviderManifest(provider)?.requiresApiKey)
}

export function isLocalProviderId(provider: string): boolean {
  return Boolean(getProviderManifest(provider)?.isLocal)
}

export function providerSupportsRpmLimit(provider: string): boolean {
  return getProviderManifest(provider)?.kind === 'openai_compatible'
}

export function getProviderDisplayName(provider: string): string {
  return getProviderManifest(provider)?.label || provider
}

export function getProviderOptionsForCapability(capability: ProviderCapability): Array<{ value: string; label: string }> {
  return AI_PROVIDER_MANIFEST
    .filter(provider => provider.capabilities.includes(capability))
    .map(provider => ({ value: provider.id, label: provider.label }))
}
