import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import providers from '../../../src/shared/ai_provider_manifest.json'
import type { components } from '../api/generated/v2'
import type { PluginSettingsApi } from '../types/browserExtensionSettings'
import type { TextStyleSettings } from '../types/textStyleSettings'
import { parseCompleteTextStyleSettings } from '../defaults/textStyleDefaults'
export function useBrowserExtensionSettings(api: PluginSettingsApi) {
  type Schema = components['schemas']
  type AgentDraft = {
    modelName: string
    customBaseUrl: string
    openaiOptions: Record<string, unknown>
  }

  const document = ref<Schema['SettingsDocument'] | null>(null)
  const fonts = ref<Schema['FontList']['items']>([])
  const form = ref<HTMLFormElement>()
  const notice = ref('正在读取插件配置…')
  const hasError = ref(false)
  const busy = ref(false)
  const saving = ref(false)
  const saveError = ref('')
  let savePromise: Promise<boolean> | null = null
  let autoSaveTimer: ReturnType<typeof setTimeout> | undefined
  const diagnosing = ref(false)
  const drafts = ref<Record<string, AgentDraft>>({})
  const secretDrafts = ref<Record<string, string>>({})
  const models = ref<Schema['ModelCatalogResponse']['models']>([])
  const dirtyProviders = new Set<string>()
  let original: Schema['SettingsDocument'] | null = null
  const style = computed(() => {
    const entry = document.value?.settings.find(row => row.domain === 'text_style_defaults')
    return entry ? parseCompleteTextStyleSettings(entry.payload) : null
  })
  const agent = computed(() =>
    document.value!.settings.find(row => row.domain === 'browser_dom_agent')!
  )
  const provider = computed(() => String(agent.value.payload.provider))
  const draft = computed(() => drafts.value[provider.value]!)
  const credential = computed(() =>
    document.value!.credentials.find(
      row => row.domain === 'browser_dom_agent' && row.provider === provider.value
    )
  )
  const providerMetadata = computed(() => providers.find(row => row.id === provider.value))
  const providerOptions = providers
    .filter(row => row.capabilities.includes('pluginAgent'))
    .map(row => ({ value: row.id, label: row.label }))
  const fontOptions = computed(() =>
    fonts.value.map(font => ({ value: font.id, label: font.displayName }))
  )

  function message(text: string, error = false) {
    notice.value = text
    hasError.value = error
  }
  function updateStyle(value: TextStyleSettings) {
    document.value!.settings.find(row => row.domain === 'text_style_defaults')!.payload = {
      ...value,
    }
    scheduleAutoSave()
  }
  function changeStyle(patch: Partial<TextStyleSettings>) {
    if (style.value) updateStyle({ ...style.value, ...patch })
  }
  function updateFont(value: string | number) {
    if (typeof value === 'string' && style.value) updateStyle({ ...style.value, fontFamily: value })
  }
  function ensureDraft(value: string) {
    if (drafts.value[value]) return
    const stored = original!.providerSettings.find(
      row => row.domain === 'browser_dom_agent' && row.provider === value
    )
    const initial = original!.settings.find(row => row.domain === 'browser_dom_agent')!.payload
    drafts.value[value] = {
      modelName: initial.provider === value ? String(initial.modelName) : '',
      customBaseUrl: initial.provider === value ? String(initial.customBaseUrl) : '',
      openaiOptions: structuredClone(initial.openaiOptions) as Record<string, unknown>,
      ...structuredClone(stored?.payload ?? {}),
    }
  }
  function changeProvider(value: string | number) {
    if (typeof value !== 'string') return
    agent.value.payload.provider = value
    ensureDraft(value)
    markProvider()
    models.value = []
  }
  function markProvider() {
    dirtyProviders.add(provider.value)
    scheduleAutoSave()
  }
  function updateBaseUrl(value: string) {
    draft.value.customBaseUrl = value
    markProvider()
  }
  function updateKey(value: string) {
    secretDrafts.value[provider.value] = value
    markProvider()
  }
  async function load() {
    clearTimeout(autoSaveTimer)
    busy.value = true
    try {
      const [settings, catalog] = await Promise.all([
        api<Schema['SettingsDocument']>('/settings?domains=text_style_defaults,browser_dom_agent'),
        api<Schema['FontList']>('/fonts'),
      ])
      original = structuredClone(settings)
      document.value = settings
      fonts.value = catalog.items
      drafts.value = {}
      secretDrafts.value = {}
      dirtyProviders.clear()
      saveError.value = ''
      models.value = []
      ensureDraft(provider.value)
      message('')
      return true
    } catch (error) {
      message(error instanceof Error ? error.message : '读取失败', true)
      return false
    } finally {
      busy.value = false
    }
  }
  async function diagnose(kind: 'models' | 'connection') {
    diagnosing.value = true
    const selected = provider.value
    const body = {
      domain: 'browser_dom_agent',
      provider: selected,
      baseUrl: draft.value.customBaseUrl,
      ...(secretDrafts.value[selected]?.trim()
        ? { secret: { api_key: secretDrafts.value[selected] } }
        : {}),
    }
    try {
      if (kind === 'models') {
        const result = await api<Schema['ModelCatalogResponse']>('/model-catalog', 'POST', body)
        if (provider.value === selected) models.value = result.models
        message(`已获取 ${result.models.length} 个模型`)
      } else {
        const result = await api<Schema['ConnectionTestResponse']>(
          '/connection-tests/llm',
          'POST',
          { ...body, model: draft.value.modelName }
        )
        message(result.message ?? (result.success ? '连接成功' : '连接失败'), !result.success)
      }
    } catch (error) {
      message(error instanceof Error ? error.message : '请求失败', true)
    } finally {
      diagnosing.value = false
    }
  }
  const fetchModels = () => diagnose('models')
  const testConnection = () => diagnose('connection')
  function scheduleAutoSave() {
    clearTimeout(autoSaveTimer)
    autoSaveTimer = setTimeout(() => void save(false), 450)
  }

  function buildTransaction(): Schema['SettingsTransaction'] {
    const current = document.value!
    const transaction: Schema['SettingsTransaction'] = {
      settings: [],
      providerSettings: [],
      credentialEdits: [],
    }
    for (const value of dirtyProviders) {
      const stored = current.providerSettings.find(
        row => row.domain === 'browser_dom_agent' && row.provider === value
      )
      const key = current.credentials.find(
        row => row.domain === 'browser_dom_agent' && row.provider === value
      )
      const secret = secretDrafts.value[value]?.trim()
      const secretChanged = Boolean(secret) && secret !== String(key?.secret?.api_key ?? '')
      if (!secretChanged && JSON.stringify(stored?.payload) === JSON.stringify(drafts.value[value]))
        continue
      if (secretChanged)
        transaction.credentialEdits!.push({
          domain: 'browser_dom_agent',
          provider: value,
          secret: { api_key: secret! },
          clientRef: value,
          baseRevision: key?.revision ?? 0,
          ...(key ? { credentialId: key.credentialId } : {}),
        })
      const credentialVersionId = stored?.credentialVersionId ?? key?.credentialVersionId
      transaction.providerSettings!.push({
        domain: 'browser_dom_agent',
        provider: value,
        payload: { ...drafts.value[value]! },
        schemaVersion: 1,
        baseRevision: stored?.revision ?? 0,
        ...(secretChanged
          ? { credentialEditRef: value }
          : credentialVersionId
            ? { credentialVersionId }
            : {}),
      })
    }
    for (const entry of current.settings) {
      if (
        JSON.stringify(entry.payload) !==
        JSON.stringify(original!.settings.find(row => row.domain === entry.domain)?.payload)
      )
        transaction.settings!.push({
          domain: entry.domain,
          payload: entry.payload,
          baseRevision: entry.revision,
          schemaVersion: entry.schemaVersion,
        })
    }
    // Freeze the submitted values so edits during the request stay in the form.
    return JSON.parse(JSON.stringify(transaction)) as Schema['SettingsTransaction']
  }

  function applySaved(
    transaction: Schema['SettingsTransaction'],
    result: Schema['SettingsTransactionResult']
  ) {
    for (const change of transaction.settings!) {
      const revision = result.settings.find(row => row.domain === change.domain)!.revision
      const baseline = original!.settings.find(row => row.domain === change.domain)!
      baseline.payload = change.payload
      baseline.revision = revision
      document.value!.settings.find(row => row.domain === change.domain)!.revision = revision
    }
    for (const key of result.credentials) {
      const index = document.value!.credentials.findIndex(
        row => row.domain === key.domain && row.provider === key.provider
      )
      if (index < 0) document.value!.credentials.push(key)
      else document.value!.credentials[index] = key
      const submitted = transaction.credentialEdits!.find(row => row.provider === key.provider)!
      if (secretDrafts.value[key.provider]?.trim() === submitted.secret.api_key) {
        delete secretDrafts.value[key.provider]
      }
    }
    for (const change of transaction.providerSettings!) {
      const revision = result.providerSettings.find(
        row => row.domain === change.domain && row.provider === change.provider
      )!.revision
      const key = result.credentials.find(
        row => row.domain === change.domain && row.provider === change.provider
      )
      const entry: Schema['ProviderSettingEntry'] = {
        domain: change.domain,
        provider: change.provider,
        payload: change.payload,
        schemaVersion: change.schemaVersion,
        revision,
        credentialVersionId: key?.credentialVersionId ?? change.credentialVersionId ?? null,
      }
      const index = document.value!.providerSettings.findIndex(
        row => row.domain === change.domain && row.provider === change.provider
      )
      if (index < 0) document.value!.providerSettings.push(entry)
      else document.value!.providerSettings[index] = entry
      if (
        JSON.stringify(drafts.value[change.provider]) === JSON.stringify(change.payload) &&
        !secretDrafts.value[change.provider]?.trim()
      ) {
        dirtyProviders.delete(change.provider)
      }
    }
  }

  async function persist(reportInvalid: boolean): Promise<boolean> {
    try {
      while (true) {
        clearTimeout(autoSaveTimer)
        if (
          form.value &&
          !(reportInvalid ? form.value.reportValidity() : form.value.checkValidity())
        )
          return false
        const transaction = buildTransaction()
        if (!transaction.settings!.length && !transaction.providerSettings!.length) {
          saveError.value = ''
          return true
        }
        saving.value = true
        const result = await api<Schema['SettingsTransactionResult']>(
          '/settings/transactions',
          'PUT',
          transaction
        )
        applySaved(transaction, result)
        saveError.value = ''
      }
    } catch (error) {
      clearTimeout(autoSaveTimer)
      saveError.value = `自动保存失败，修改尚未保存。${error instanceof Error ? error.message : ''}`
      return false
    } finally {
      saving.value = false
    }
  }
  function save(reportInvalid = true): Promise<boolean> {
    clearTimeout(autoSaveTimer)
    if (savePromise) return savePromise
    if (busy.value) return Promise.resolve(false)
    if (!document.value) return Promise.resolve(true)
    savePromise = persist(reportInvalid).finally(() => {
      savePromise = null
    })
    return savePromise
  }
  async function refresh() {
    if (busy.value || saving.value || saveError.value || (form.value && !form.value.checkValidity())) return
    if (document.value) {
      const transaction = buildTransaction()
      if (transaction.settings!.length || transaction.providerSettings!.length) return
    }
    await load()
  }
  onMounted(load)
  onBeforeUnmount(() => {
    void save(false)
  })
  return {
    document,
    style,
    fontOptions,
    form,
    notice,
    hasError,
    busy,
    saving,
    saveError,
    diagnosing,
    draft,
    provider,
    providerOptions,
    providerMetadata,
    credential,
    secretDrafts,
    models,
    changeStyle,
    updateFont,
    changeProvider,
    markProvider,
    updateBaseUrl,
    updateKey,
    load,
    refresh,
    save,
    fetchModels,
    testConnection,
  }
}
