import type { components } from '../api/generated/v2'

type V2CredentialEdit = components['schemas']['CredentialEdit']
type V2CredentialSummary = components['schemas']['CredentialSummary']
type V2ProviderSettingEntry = components['schemas']['ProviderSettingEntry']
type V2ProviderSettingMutation = components['schemas']['ProviderSettingMutation']

export function providerKeyField(domain: string): string {
  return domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key'
}

export function sameSettingValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false
  if (Array.isArray(left) !== Array.isArray(right)) return false
  const a = left as Record<string, unknown>
  const b = right as Record<string, unknown>
  const keys = (value: Record<string, unknown>) => Object.keys(value).filter(key => value[key] !== undefined)
  return keys(a).length === keys(b).length
    && keys(a).every(key => sameSettingValue(a[key], b[key]))
}

export function boundProviderCredential(
  credentials: V2CredentialSummary[],
  stored: V2ProviderSettingEntry | undefined,
  domain: string,
  provider: string,
): V2CredentialSummary | undefined {
  const version = stored?.credentialVersionId ?? `browser:${domain}:${provider}`
  return credentials.find(row => row.credentialVersionId === version
    && row.domain === domain && row.provider === provider)
}

export function mergeCredentialSummaries(
  current: V2CredentialSummary[], updates: V2CredentialSummary[],
): V2CredentialSummary[] {
  const merged = new Map(current.map(row => [row.credentialVersionId, row]))
  updates.forEach(row => merged.set(row.credentialVersionId, row))
  return [...merged.values()]
}

export function appendProviderSettingChange(
  providerSettings: V2ProviderSettingMutation[],
  credentialEdits: V2CredentialEdit[],
  options: {
    domain: string
    provider: string
    payload: Record<string, unknown>
    secret?: Record<string, unknown>
    stored?: V2ProviderSettingEntry
    credentials: V2CredentialSummary[]
  },
): boolean {
  const { domain, provider, payload, stored, credentials } = options
  const bound = boundProviderCredential(credentials, stored, domain, provider)
  const secret = options.secret === undefined ? undefined : Object.fromEntries(
    Object.entries(options.secret).map(([key, value]) => [key, typeof value === 'string' ? value.trim() : value]),
  )
  const hasSecret = secret !== undefined && Object.values(secret).some(value => value !== '' && value != null)
  const secretChanged = secret !== undefined
    && (bound ? !sameSettingValue(secret, bound.secret) : hasSecret)
  if (!secretChanged && sameSettingValue(payload, stored?.payload)) return false

  const mutation: V2ProviderSettingMutation = {
    domain, provider, payload, baseRevision: stored?.revision ?? 0,
    credentialVersionId: bound?.credentialVersionId ?? stored?.credentialVersionId ?? null,
  }
  if (secretChanged && secret !== undefined) {
    const current = credentials.filter(row => row.domain === domain && row.provider === provider)
      .sort((a, b) => b.currentVersion - a.currentVersion)[0]
    if (current && sameSettingValue(secret, current.secret)) {
      mutation.credentialVersionId = current.credentialVersionId
    } else {
      const clientRef = `credential:${domain}:${provider}`
      credentialEdits.push({
        domain, provider, secret, baseRevision: current?.revision ?? 0,
        credentialId: current?.credentialId, clientRef,
      })
      delete mutation.credentialVersionId
      mutation.credentialEditRef = clientRef
    }
  }
  providerSettings.push(mutation)
  return true
}
