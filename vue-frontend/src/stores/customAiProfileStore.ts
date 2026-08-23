import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import type {
  V2CredentialEdit,
  V2CredentialSummary,
  V2SettingsTransaction,
} from '@/api/v2/settings'
import {
  getV2Settings,
  saveV2SettingsTransaction,
} from '@/api/v2/settings'
import {
  prepareBrowserCredentialTransaction,
  restoreBrowserCredentialLeases,
} from '@/services/browserCredentials'
import {
  CUSTOM_AI_PROFILE_KINDS,
  type CustomAiProfile,
  type CustomAiProfileKind,
  type CustomAiProfilePayload,
} from '@/types/customAiProfile'

const SETTING_DOMAIN = 'custom_ai_profiles'
const CREDENTIAL_DOMAIN = 'custom_ai_profile'

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function parseProfile(value: unknown): CustomAiProfilePayload | null {
  if (!isRecord(value)) return null
  if (!['id', 'name', 'kind', 'baseUrl', 'model'].every(key => key in value)) return null
  if (Object.keys(value).length !== 5) return null
  const { id, name, kind, baseUrl, model } = value
  if (
    typeof id !== 'string'
    || typeof name !== 'string'
    || typeof kind !== 'string'
    || typeof baseUrl !== 'string'
    || typeof model !== 'string'
    || !CUSTOM_AI_PROFILE_KINDS.includes(kind as CustomAiProfileKind)
  ) {
    return null
  }
  return { id, name, kind: kind as CustomAiProfileKind, baseUrl, model }
}

function parsePayload(value: unknown): CustomAiProfilePayload[] | null {
  if (!isRecord(value) || Object.keys(value).length !== 1 || !Array.isArray(value.profiles)) {
    return null
  }
  const profiles = value.profiles.map(parseProfile)
  return profiles.every((profile): profile is CustomAiProfilePayload => profile !== null)
    ? profiles
    : null
}

function credentialIdentity(domain: string, provider: string): string {
  return `${domain}\u0000${provider}`
}

function mergeCredentials(
  current: V2CredentialSummary[],
  updates: V2CredentialSummary[],
): V2CredentialSummary[] {
  const result = new Map(
    current.map(item => [credentialIdentity(item.domain, item.provider), item]),
  )
  updates.forEach(item => result.set(credentialIdentity(item.domain, item.provider), item))
  return [...result.values()]
}

function profileApiKey(credentials: V2CredentialSummary[], profileId: string): string {
  const value = credentials.find(
    item => item.domain === CREDENTIAL_DOMAIN && item.provider === profileId,
  )?.secret.api_key
  return typeof value === 'string' ? value : ''
}

function normalizedProfile(profile: CustomAiProfile): CustomAiProfile {
  return {
    id: profile.id,
    name: profile.name.trim(),
    kind: profile.kind,
    baseUrl: profile.baseUrl.trim().replace(/\/$/, ''),
    apiKey: profile.apiKey.trim(),
    model: profile.model.trim(),
  }
}

function validateProfiles(profiles: CustomAiProfile[]): string | null {
  const ids = new Set<string>()
  const names = new Set<string>()
  for (const profile of profiles) {
    if (!profile.id || ids.has(profile.id)) return '自定义服务 ID 重复'
    ids.add(profile.id)
    if (!profile.name || profile.name.length > 80) return '配置名称应为 1 到 80 个字符'
    const nameIdentity = `${profile.kind}\u0000${profile.name.toLowerCase()}`
    if (names.has(nameIdentity)) return '同一用途下不能使用重复的配置名称'
    names.add(nameIdentity)
    try {
      const url = new URL(profile.baseUrl)
      if (!['http:', 'https:'].includes(url.protocol) || !url.host || url.search || url.hash) {
        return 'Base URL 必须是有效的 HTTP 地址，且不能包含查询参数或片段'
      }
    } catch {
      return 'Base URL 必须是有效的 HTTP 地址'
    }
    if (!profile.apiKey) return 'API Key 不能为空'
    if (!profile.model) return '模型名不能为空'
  }
  return null
}

export const useCustomAiProfileStore = defineStore('customAiProfiles', () => {
  const profiles = ref<CustomAiProfile[]>([])
  const isLoaded = ref(false)
  const isSaving = ref(false)
  const error = ref<string | null>(null)
  let settingRevision = 0
  let credentials: V2CredentialSummary[] = []
  let loadPromise: Promise<boolean> | null = null

  const profilesByKind = computed(() => {
    const result = new Map<CustomAiProfileKind, CustomAiProfile[]>()
    CUSTOM_AI_PROFILE_KINDS.forEach(kind => result.set(kind, []))
    profiles.value.forEach(profile => result.get(profile.kind)?.push(profile))
    result.forEach(items => items.sort((left, right) => left.name.localeCompare(right.name, 'zh-CN')))
    return result
  })

  async function load(): Promise<boolean> {
    if (isLoaded.value) return true
    if (loadPromise) return loadPromise
    loadPromise = (async () => {
      try {
        const [document, browserCredentials] = await Promise.all([
          getV2Settings([SETTING_DOMAIN, CREDENTIAL_DOMAIN]),
          restoreBrowserCredentialLeases(),
        ])
        const entry = document.settings.find(item => item.domain === SETTING_DOMAIN)
        if (!entry) throw new Error('后端自定义 OpenAI 配置库缺失')
        const payload = parsePayload(entry.payload)
        if (!payload) throw new Error('后端自定义 OpenAI 配置格式无效')
        settingRevision = entry.revision
        credentials = mergeCredentials(document.credentials, browserCredentials)
        profiles.value = payload.map(profile => ({
          ...profile,
          apiKey: profileApiKey(credentials, profile.id),
        }))
        error.value = null
        isLoaded.value = true
        return true
      } catch (reason) {
        error.value = reason instanceof Error ? reason.message : '自定义 OpenAI 配置加载失败'
        return false
      }
    })()
    try {
      return await loadPromise
    } finally {
      loadPromise = null
    }
  }

  async function persist(nextProfiles: CustomAiProfile[]): Promise<boolean> {
    const normalized = nextProfiles.map(normalizedProfile)
    const validationError = validateProfiles(normalized)
    if (validationError) {
      error.value = validationError
      return false
    }
    isSaving.value = true
    try {
      const credentialEdits: V2CredentialEdit[] = []
      normalized.forEach((profile) => {
        const existing = credentials.find(
          item => item.domain === CREDENTIAL_DOMAIN && item.provider === profile.id,
        )
        if (existing?.secret.api_key === profile.apiKey) return
        credentialEdits.push({
          domain: CREDENTIAL_DOMAIN,
          provider: profile.id,
          secret: { api_key: profile.apiKey },
          baseRevision: existing?.revision ?? 0,
          credentialId: existing?.credentialId,
          clientRef: `custom-ai-profile:${profile.id}`,
        })
      })
      const transaction: V2SettingsTransaction = {
        settings: [{
          domain: SETTING_DOMAIN,
          payload: {
            profiles: normalized.map(({ apiKey: _apiKey, ...profile }) => profile),
          },
          baseRevision: settingRevision,
          schemaVersion: 1,
        }],
        credentialEdits,
      }
      const prepared = await prepareBrowserCredentialTransaction(transaction)
      const result = await saveV2SettingsTransaction(prepared.transaction)
      const settingResult = result.settings.find(item => item.domain === SETTING_DOMAIN)
      if (!settingResult) throw new Error('后端未返回自定义 OpenAI 配置保存结果')
      settingRevision = settingResult.revision
      credentials = mergeCredentials(credentials, result.credentials)
      credentials = mergeCredentials(credentials, prepared.summaries)
      profiles.value = normalized
      error.value = null
      return true
    } catch (reason) {
      error.value = reason instanceof Error ? reason.message : '自定义 OpenAI 配置保存失败'
      return false
    } finally {
      isSaving.value = false
    }
  }

  function byKind(kind: CustomAiProfileKind): CustomAiProfile[] {
    return profilesByKind.value.get(kind) ?? []
  }

  async function create(profile: Omit<CustomAiProfile, 'id'>): Promise<CustomAiProfile | null> {
    const created = { ...profile, id: crypto.randomUUID() }
    return await persist([...profiles.value, created]) ? normalizedProfile(created) : null
  }

  async function update(profile: CustomAiProfile): Promise<boolean> {
    if (!profiles.value.some(item => item.id === profile.id)) {
      error.value = '要编辑的自定义服务不存在'
      return false
    }
    return persist(profiles.value.map(item => item.id === profile.id ? profile : item))
  }

  async function remove(profileId: string): Promise<boolean> {
    return persist(profiles.value.filter(item => item.id !== profileId))
  }

  return {
    profiles,
    isLoaded,
    isSaving,
    error,
    load,
    byKind,
    create,
    update,
    remove,
  }
})
