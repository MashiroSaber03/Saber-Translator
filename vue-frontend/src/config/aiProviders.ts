import providerManifestData from '../../../src/shared/ai_provider_manifest.json'

export type ProviderKind = 'openai_compatible' | 'local' | 'adapter'
export type ProviderCapability =
  | 'translation'
  | 'hqTranslation'
  | 'visionOcr'
  | 'modelFetch'
  | 'connectionTest'
  | 'webImportAgent'
  | 'vlm'
  | 'chat'
  | 'embedding'
  | 'rerank'
  | 'imageGen'

export type ProviderModelType = 'vlm' | 'chat' | 'embedding' | 'reranker' | 'imageGen'

export interface AiProviderManifestEntry {
  id: string
  label: string
  kind: ProviderKind
  defaultBaseUrl?: string
  capabilityBaseUrls?: Partial<Record<ProviderCapability, string>>
  capabilities: ProviderCapability[]
  requiresApiKey: boolean
  requiresModel: boolean
  requiresBaseUrl: boolean
  isLocal: boolean
  supportsStream: boolean
  supportsJsonResponse: boolean
  supportsReasoningControl: boolean
  legacyIds?: string[]
  defaultModels?: Partial<Record<ProviderModelType, string>>
}

export const AI_PROVIDER_MANIFEST = providerManifestData as AiProviderManifestEntry[]

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

export function getProviderBaseUrl(provider: string, capability?: ProviderCapability): string {
  const manifest = getProviderManifest(provider)
  if (!manifest) return ''
  if (capability && manifest.capabilityBaseUrls?.[capability]) {
    return manifest.capabilityBaseUrls[capability] || ''
  }
  return manifest.defaultBaseUrl || ''
}

export function getProviderDefaultModel(provider: string, modelType: ProviderModelType): string {
  return getProviderManifest(provider)?.defaultModels?.[modelType] || ''
}
