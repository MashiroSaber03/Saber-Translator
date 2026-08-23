import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import type {
  V2CredentialEdit,
  V2CredentialSummary,
  V2SettingsTransaction,
} from '@/api/v2/settings'
import {
  deleteV2Credential,
  getV2Settings,
  saveV2SettingsTransaction,
} from '@/api/v2/settings'
import {
  deleteBrowserCredential,
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
    baseUrl: profile.baseUrl.trim().replace(/\/+$/, ''),
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
    if (!profile.model) return '模型名不能为空'
  }
  return null
}

interface ProfileAuthority {
  profiles: CustomAiProfile[]
  revision: number
  credentials: V2CredentialSummary[]
}

export const useCustomAiProfileStore = defineStore('customAiProfiles', () => {
  const profiles = ref<CustomAiProfile[]>([])
  const isLoaded = ref(false)
  const isSaving = ref(false)
  const error = ref<string | null>(null)
  let contextGeneration = 0
  let loadPromise: Promise<boolean> | null = null

  const profilesByKind = computed(() => {
    const result = new Map<CustomAiProfileKind, CustomAiProfile[]>()
    CUSTOM_AI_PROFILE_KINDS.forEach(kind => result.set(kind, []))
    profiles.value.forEach(profile => result.get(profile.kind)?.push(profile))
    result.forEach(items => items.sort((left, right) => left.name.localeCompare(right.name, 'zh-CN')))
    return result
  })

  async function readAuthority(): Promise<ProfileAuthority> {
    const [document, browserCredentials] = await Promise.all([
      getV2Settings([SETTING_DOMAIN, CREDENTIAL_DOMAIN]),
      restoreBrowserCredentialLeases(),
    ])
    const entry = document.settings.find(item => item.domain === SETTING_DOMAIN)
    if (!entry) throw new Error('后端自定义 OpenAI 配置库缺失')
    const payload = parsePayload(entry.payload)
    if (!payload) throw new Error('后端自定义 OpenAI 配置格式无效')
    const credentials = mergeCredentials(document.credentials, browserCredentials)
    return {
      profiles: payload.map(profile => ({
        ...profile,
        apiKey: profileApiKey(credentials, profile.id),
      })),
      revision: entry.revision,
      credentials,
    }
  }

  async function load(): Promise<boolean> {
    if (isLoaded.value) return true
    if (loadPromise) return loadPromise
    const generation = contextGeneration
    const pending = (async () => {
      try {
        const authority = await readAuthority()
        if (generation !== contextGeneration) return false
        profiles.value = authority.profiles
        error.value = null
        isLoaded.value = true
        return true
      } catch (reason) {
        if (generation !== contextGeneration) return false
        error.value = reason instanceof Error ? reason.message : '自定义 OpenAI 配置加载失败'
        return false
      }
    })()
    loadPromise = pending
    try {
      return await pending
    } finally {
      if (loadPromise === pending) loadPromise = null
    }
  }

  async function deleteProfileCredential(
    authority: ProfileAuthority,
    profileId: string,
  ): Promise<void> {
    if (await deleteBrowserCredential(CREDENTIAL_DOMAIN, profileId)) return
    const credential = authority.credentials.find(
      item => item.domain === CREDENTIAL_DOMAIN && item.provider === profileId,
    )
    if (credential) await deleteV2Credential(credential.credentialId)
  }

  async function persist(
    mutate: (current: CustomAiProfile[]) => CustomAiProfile[],
    removedProfileId?: string,
  ): Promise<boolean> {
    const generation = contextGeneration
    isSaving.value = true
    try {
      const authority = await readAuthority()
      if (generation !== contextGeneration) return false
      const normalized = mutate(authority.profiles).map(normalizedProfile)
      const validationError = validateProfiles(normalized)
      if (validationError) throw new Error(validationError)
      const credentialEdits: V2CredentialEdit[] = []
      normalized.forEach((profile) => {
        const existing = authority.credentials.find(
          item => item.domain === CREDENTIAL_DOMAIN && item.provider === profile.id,
        )
        if (!profile.apiKey || existing?.secret.api_key === profile.apiKey) return
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
          baseRevision: authority.revision,
          schemaVersion: 1,
        }],
        credentialEdits,
      }
      const prepared = await prepareBrowserCredentialTransaction(transaction)
      const result = await saveV2SettingsTransaction(prepared.transaction)
      if (generation !== contextGeneration) return false
      const settingResult = result.settings.find(item => item.domain === SETTING_DOMAIN)
      if (!settingResult) throw new Error('后端未返回自定义 OpenAI 配置保存结果')
      profiles.value = normalized
      isLoaded.value = true
      error.value = null
      if (removedProfileId) {
        try {
          await deleteProfileCredential(authority, removedProfileId)
        } catch (reason) {
          if (generation === contextGeneration) {
            const message = reason instanceof Error ? reason.message : '未知错误'
            error.value = `配置已删除，但关联 API Key 清理失败：${message}`
          }
        }
      }
      return generation === contextGeneration
    } catch (reason) {
      if (generation !== contextGeneration) return false
      error.value = reason instanceof Error ? reason.message : '自定义 OpenAI 配置保存失败'
      return false
    } finally {
      if (generation === contextGeneration) isSaving.value = false
    }
  }

  function byKind(kind: CustomAiProfileKind): CustomAiProfile[] {
    return profilesByKind.value.get(kind) ?? []
  }

  async function create(profile: Omit<CustomAiProfile, 'id'>): Promise<CustomAiProfile | null> {
    const created = normalizedProfile({ ...profile, id: crypto.randomUUID() })
    if (!created.apiKey) {
      error.value = 'API Key 不能为空'
      return null
    }
    return await persist(current => [...current, created]) ? created : null
  }

  async function update(profile: CustomAiProfile): Promise<boolean> {
    const updated = normalizedProfile(profile)
    if (!updated.apiKey) {
      error.value = 'API Key 不能为空'
      return false
    }
    return persist((current) => {
      if (!current.some(item => item.id === updated.id)) {
        throw new Error('要编辑的自定义服务不存在')
      }
      return current.map(item => item.id === updated.id ? updated : item)
    })
  }

  async function remove(profileId: string): Promise<boolean> {
    return persist((current) => {
      if (!current.some(item => item.id === profileId)) {
        throw new Error('要删除的自定义服务不存在')
      }
      return current.filter(item => item.id !== profileId)
    }, profileId)
  }

  function clearError(): void {
    error.value = null
  }

  function reset(): void {
    contextGeneration += 1
    loadPromise = null
    profiles.value = []
    isLoaded.value = false
    isSaving.value = false
    error.value = null
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
    clearError,
    reset,
  }
})
