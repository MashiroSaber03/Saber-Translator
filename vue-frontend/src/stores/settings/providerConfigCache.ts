import { normalizeProviderId } from '@/config/aiProviders'

export type ProviderCredentialConfig = {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
}

export type ProviderCredentialTarget = {
  apiKey: string
  modelName: string
  customBaseUrl: string
}

type NormalizeProvider = (provider: string) => string | null

type SaveProviderCacheEntryOptions<TConfig> = {
  provider: string
  cache: Record<string, TConfig>
  buildConfig: () => TConfig
  persist?: () => void
  normalizeProvider?: NormalizeProvider
}

type RestoreProviderCacheEntryOptions<TConfig> = {
  provider: string
  cache: Record<string, TConfig>
  applyCached: (config: TConfig) => void
  applyMissing: () => void
  normalizeProvider?: NormalizeProvider
}

function defaultNormalizeProvider(provider: string): string {
  return normalizeProviderId(provider)
}

function getProviderCacheKey(provider: string, normalizeProvider: NormalizeProvider): string | null {
  if (!provider) return null
  const normalizedProvider = normalizeProvider(provider)
  return normalizedProvider || null
}

export function snapshotProviderCredentials(
  target: ProviderCredentialTarget
): Required<ProviderCredentialConfig> {
  return {
    apiKey: target.apiKey,
    modelName: target.modelName,
    customBaseUrl: target.customBaseUrl,
  }
}

export function applyProviderCredentials(
  target: ProviderCredentialTarget,
  config: ProviderCredentialConfig
): void {
  if (config.apiKey !== undefined) target.apiKey = config.apiKey
  if (config.modelName !== undefined) target.modelName = config.modelName
  if (config.customBaseUrl !== undefined) target.customBaseUrl = config.customBaseUrl
}

export function clearProviderCredentials(target: ProviderCredentialTarget): void {
  target.apiKey = ''
  target.modelName = ''
  target.customBaseUrl = ''
}

export function saveProviderCacheEntry<TConfig>({
  provider,
  cache,
  buildConfig,
  persist,
  normalizeProvider = defaultNormalizeProvider,
}: SaveProviderCacheEntryOptions<TConfig>): void {
  const providerKey = getProviderCacheKey(provider, normalizeProvider)
  if (!providerKey) return

  cache[providerKey] = buildConfig()
  persist?.()
}

export function restoreProviderCacheEntry<TConfig>({
  provider,
  cache,
  applyCached,
  applyMissing,
  normalizeProvider = defaultNormalizeProvider,
}: RestoreProviderCacheEntryOptions<TConfig>): void {
  const providerKey = getProviderCacheKey(provider, normalizeProvider)
  if (!providerKey) return

  const cached = cache[providerKey]
  if (cached) {
    applyCached(cached)
    return
  }

  applyMissing()
}
