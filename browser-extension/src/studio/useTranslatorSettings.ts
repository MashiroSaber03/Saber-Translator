import { computed, onBeforeUnmount, ref, watch, type Ref } from 'vue'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
import type { TranslationSettings } from '../../../vue-frontend/src/types/translationSettings'
import type { components } from '../../../vue-frontend/src/api/generated/v2'

type Schema = components['schemas']
export type ServiceKey = 'translation' | 'hqTranslation' | 'aiVisionOcr'
export const serviceDomains = {
  translation: 'translation',
  hqTranslation: 'hq',
  aiVisionOcr: 'ai_vision_ocr',
} as const
const fields: Record<string, string[]> = {
  translation: ['modelName', 'customBaseUrl', 'openaiOptions', 'translationMode'],
  hq: ['modelName', 'customBaseUrl', 'openaiOptions', 'batchSize', 'prompt'],
  ai_vision_ocr: [
    'modelName',
    'customBaseUrl',
    'openaiOptions',
    'prompt',
    'promptMode',
    'minImageSize',
  ],
  ocr: ['version', 'sourceLanguage'],
}
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value))
const identity = (domain: string, provider: string) => `${domain}:${provider}`
export function useTranslatorSettings(api: PluginSettingsApi, active: Ref<boolean>) {
  const form = ref<HTMLFormElement>()
  const settings = ref<TranslationSettings>()
  const document = ref<Schema['SettingsDocument']>()
  const error = ref('')
  const loading = ref(false)
  const saving = ref(false)
  const dirty = ref(false)
  const secrets = ref<Record<string, Record<string, string>>>({})
  const drafts: Record<
    string,
    { domain: string; provider: string; payload: Record<string, unknown> }
  > = {}
  let baseline = ''
  let generation = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  let pending: Promise<boolean> | null = null
  const globalApi: PluginSettingsApi = (path, method, body) =>
    api(`${path}${path.includes('?') ? '&' : '?'}scope=global`, method, body)
  const entry = computed(() => document.value!.settings.find(row => row.domain === 'translation')!)
  const credential = (domain: string, provider: string) =>
    document.value?.credentials.find(row => row.domain === domain && row.provider === provider)
  const secretValue = (domain: string, provider: string, field: string) =>
    secrets.value[identity(domain, provider)]?.[field] ??
    String(credential(domain, provider)?.secret?.[field] ?? '')
  const pick = (domain: string, source: object) =>
    Object.fromEntries(
      fields[domain]!.map(key => [key, clone((source as Record<string, unknown>)[key])])
    )
  function remember(key: ServiceKey | 'baiduOcr') {
    const domain = key === 'baiduOcr' ? 'ocr' : serviceDomains[key]
    const provider = key === 'baiduOcr' ? 'baidu' : settings.value![key].provider
    drafts[identity(domain, provider)] = {
      domain,
      provider,
      payload: pick(domain, settings.value![key]),
    }
  }
  function change(key?: ServiceKey | 'baiduOcr') {
    if (key) remember(key)
    dirty.value = true
    generation++
    clearTimeout(timer)
    timer = setTimeout(() => void save(), 450)
  }
  function selectProvider(key: ServiceKey, provider: string) {
    const config = settings.value![key]
    if (config.provider === provider) return
    remember(key)
    const domain = serviceDomains[key]
    const cached =
      drafts[identity(domain, provider)]?.payload ??
      document.value!.providerSettings.find(
        row => row.domain === domain && row.provider === provider
      )?.payload
    Object.assign(config, { modelName: '', customBaseUrl: '' }, clone(cached ?? {}), { provider })
    change(key)
  }
  function updateSecret(key: ServiceKey | 'baiduOcr', field: string, value: string) {
    const domain = key === 'baiduOcr' ? 'ocr' : serviceDomains[key]
    const provider = key === 'baiduOcr' ? 'baidu' : settings.value![key].provider
    const id = identity(domain, provider)
    secrets.value[id] = { ...secrets.value[id], [field]: value }
    change(key)
  }
  async function load() {
    if (loading.value || saving.value) return
    clearTimeout(timer)
    loading.value = true
    try {
      const result = await globalApi<Schema['SettingsDocument']>(
        '/settings?domains=translation,hq,ai_vision_ocr,ocr'
      )
      document.value = result
      settings.value = clone(
        result.settings.find(row => row.domain === 'translation')!.payload
      ) as unknown as TranslationSettings
      for (const key of Object.keys(serviceDomains) as ServiceKey[]) {
        const config = settings.value[key]
        const saved = result.providerSettings.find(
          row => row.domain === serviceDomains[key] && row.provider === config.provider
        )
        Object.assign(config, clone(saved?.payload ?? {}))
      }
      Object.assign(
        settings.value.baiduOcr,
        clone(
          result.providerSettings.find(row => row.domain === 'ocr' && row.provider === 'baidu')
            ?.payload ?? {}
        )
      )
      baseline = JSON.stringify(settings.value)
      for (const key of Object.keys(drafts)) delete drafts[key]
      secrets.value = {}
      dirty.value = false
      error.value = ''
    } catch (e) {
      error.value = (e as Error).message
    } finally {
      loading.value = false
    }
  }
  async function persist(): Promise<boolean> {
    saving.value = true
    try {
      while (dirty.value) {
        if (form.value && !form.value.checkValidity()) return false
        const version = generation
        const payload = clone(settings.value!) as unknown as Record<string, unknown>
        const submitted = JSON.stringify(payload)
        const transaction: Schema['SettingsTransaction'] = {
          settings: [],
          providerSettings: [],
          credentialEdits: [],
        }
        if (submitted !== baseline)
          transaction.settings!.push({
            domain: 'translation',
            payload,
            baseRevision: entry.value.revision,
            schemaVersion: entry.value.schemaVersion,
          })
        for (const [id, draft] of Object.entries(drafts)) {
          const stored = document.value!.providerSettings.find(
            row => row.domain === draft.domain && row.provider === draft.provider
          )
          const key = credential(draft.domain, draft.provider)
          const changes = Object.entries(secrets.value[id] ?? {})
            .map(([field, value]) => [field, value.trim()] as const)
            .filter(([field, value]) => value && value !== String(key?.secret?.[field] ?? ''))
          const secret = changes.length
            ? { ...key?.secret, ...Object.fromEntries(changes) }
            : {}
          if (
            JSON.stringify(stored?.payload) === JSON.stringify(draft.payload) &&
            !Object.keys(secret).length
          )
            continue
          const row: Schema['ProviderSettingMutation'] = {
            ...clone(draft),
            baseRevision: stored?.revision ?? 0,
            schemaVersion: stored?.schemaVersion ?? 1,
          }
          if (Object.keys(secret).length) {
            transaction.credentialEdits!.push({
              domain: draft.domain,
              provider: draft.provider,
              secret,
              baseRevision: key?.revision ?? 0,
              ...(key ? { credentialId: key.credentialId } : {}),
              clientRef: id,
            })
            row.credentialEditRef = id
          } else if (key) row.credentialVersionId = key.credentialVersionId
          transaction.providerSettings!.push(row)
        }
        if (transaction.settings!.length || transaction.providerSettings!.length) {
          const result = await globalApi<Schema['SettingsTransactionResult']>(
            '/settings/transactions',
            'PUT',
            transaction
          )
          for (const row of result.settings)
            if (row.domain === 'translation') entry.value.revision = row.revision
          for (const row of result.credentials) {
            const index = document.value!.credentials.findIndex(
              key => key.domain === row.domain && key.provider === row.provider
            )
            if (index < 0) document.value!.credentials.push(row)
            else document.value!.credentials[index] = row
            const id = identity(row.domain, row.provider)
            const sent = transaction.credentialEdits!.find(key => key.clientRef === id)!
            for (const [field, value] of Object.entries(sent.secret))
              if (secrets.value[id]?.[field]?.trim() === value) delete secrets.value[id]![field]
          }
          for (const row of transaction.providerSettings!) {
            const revision = result.providerSettings.find(
              item => item.domain === row.domain && item.provider === row.provider
            )!.revision
            const saved = {
              domain: row.domain,
              provider: row.provider,
              payload: row.payload,
              revision,
              schemaVersion: row.schemaVersion,
              credentialVersionId:
                credential(row.domain, row.provider)?.credentialVersionId ?? null,
            }
            const index = document.value!.providerSettings.findIndex(
              item => item.domain === row.domain && item.provider === row.provider
            )
            if (index < 0) document.value!.providerSettings.push(saved)
            else document.value!.providerSettings[index] = saved
          }
        }
        baseline = submitted
        dirty.value = version !== generation
        error.value = ''
      }
      for (const key of Object.keys(drafts)) delete drafts[key]
      return true
    } catch (e) {
      clearTimeout(timer)
      error.value = `自动保存失败，修改尚未保存。${(e as Error).message}`
      return false
    } finally {
      saving.value = false
    }
  }
  function save(): Promise<boolean> {
    clearTimeout(timer)
    if (pending) return pending
    if (form.value && !form.value.checkValidity()) return Promise.resolve(false)
    if (!dirty.value || !settings.value) return Promise.resolve(true)
    pending = persist().finally(() => {
      pending = null
    })
    return pending
  }
  watch(
    active,
    value => {
      if (value && !dirty.value) void load()
    },
    { immediate: true }
  )
  onBeforeUnmount(() => {
    void save()
  })
  return {
    form,
    settings,
    error,
    loading,
    saving,
    dirty,
    secretValue,
    globalApi,
    change,
    selectProvider,
    updateSecret,
    load,
    save,
  }
}
