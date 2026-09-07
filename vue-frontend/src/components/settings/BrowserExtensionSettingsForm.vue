<template>
  <div class="browser-extension-settings">
    <ProductStatusBanner v-if="saveError || notice" :tone="saveError || hasError ? 'danger' : 'info'" role="status">
      {{ saveError || notice }}
    </ProductStatusBanner>
    <ProductActionRow v-if="saveError" justify="start">
      <UiButton variant="secondary" :disabled="saving" @click="save()">重试保存</UiButton>
      <UiButton variant="secondary" :disabled="saving" @click="load">放弃修改并重新读取</UiButton>
    </ProductActionRow>
    <form v-if="document && style" ref="form" @submit.prevent="save()">
      <fieldset :disabled="busy">
        <p class="browser-extension-settings__intro">
          插件文本样式独立保存，翻译服务沿用翻译器配置。
        </p>
        <TextStyleForm
          :model-value="style"
          :id-prefix="idPrefix"
          :font-select-options="fontOptions"
          @change="changeStyle"
          @font-change="updateFont"
        />
        <details class="browser-extension-settings__agent">
          <summary>网页识别助手（可选）</summary>
          <ProductFormSection>
            <p class="browser-extension-settings__intro">
              用于识别网页中的漫画图片，不参与翻译。
            </p>
            <UiField variant="settings" label="识别助手服务商" :control-id="idPrefix + 'Provider'">
              <UiSelect
                :id="idPrefix + 'Provider'"
                :model-value="provider"
                :options="providerOptions"
                @change="changeProvider"
              />
            </UiField>
            <AiProviderCredentialFields
              :api-key="secretDrafts[provider] ?? ''"
              :api-key-input-id="idPrefix + 'Key'"
              :api-key-label="credential ? 'API Key · 已配置，留空保持' : 'API Key'"
              :base-url="draft.customBaseUrl"
              :base-url-input-id="idPrefix + 'BaseUrl'"
              :include-api-key="Boolean(providerMetadata?.requiresApiKey)"
              show-base-url
              @update:api-key="updateKey"
              @update:base-url="updateBaseUrl"
            />
            <UiField variant="settings" label="模型名称" :control-id="idPrefix + 'Model'">
              <UiInput
                :id="idPrefix + 'Model'"
                v-model="draft.modelName"
                :list="idPrefix + 'Models'"
                @update:model-value="markProvider"
              />
              <datalist :id="idPrefix + 'Models'">
                <option v-for="model in models" :key="model.id" :value="model.id">
                  {{ model.name }}
                </option>
              </datalist>
            </UiField>
            <ProductActionRow justify="start">
              <UiButton
                v-if="providerMetadata?.capabilities.includes('modelFetch')"
                type="button"
                variant="secondary"
                :disabled="diagnosing"
                @click="fetchModels"
              >
                获取模型列表
              </UiButton>
              <UiButton
                type="button"
                variant="secondary"
                :disabled="diagnosing"
                @click="testConnection"
              >
                测试连接
              </UiButton>
            </ProductActionRow>
          </ProductFormSection>
        </details>
      </fieldset>
    </form>
    <UiButton v-else-if="hasError" variant="secondary" @click="load">重新读取</UiButton>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId } from 'vue'
import providers from '../../../../src/shared/ai_provider_manifest.json'
import type { components } from '@/api/generated/v2'
import type { PluginSettingsApi } from '@/types/browserExtensionSettings'
import type { TextStyleSettings } from '@/types/textStyleSettings'
import { parseCompleteTextStyleSettings } from '@/defaults/textStyleDefaults'
import TextStyleForm from './TextStyleForm.vue'
import AiProviderCredentialFields from './AiProviderCredentialFields.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'

type Schema = components['schemas']
type AgentDraft = {
  modelName: string
  customBaseUrl: string
  openaiOptions: Record<string, unknown>
}
const props = defineProps<{ api: PluginSettingsApi }>()
const emit = defineEmits<{ saving: [value: boolean] }>()
const idPrefix = `browserSettings-${useId()}-`
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
  document.value!.settings.find(row => row.domain === 'text_style_defaults')!.payload = { ...value }
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
  emit('saving', true)
  try {
    const [settings, catalog] = await Promise.all([
      props.api<Schema['SettingsDocument']>(
        '/settings?domains=text_style_defaults,browser_dom_agent'
      ),
      props.api<Schema['FontList']>('/fonts'),
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
    emit('saving', false)
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
      const result = await props.api<Schema['ModelCatalogResponse']>('/model-catalog', 'POST', body)
      if (provider.value === selected) models.value = result.models
      message(`已获取 ${result.models.length} 个模型`)
    } else {
      const result = await props.api<Schema['ConnectionTestResponse']>(
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
    if (!secret && JSON.stringify(stored?.payload) === JSON.stringify(drafts.value[value])) continue
    if (secret)
      transaction.credentialEdits!.push({
        domain: 'browser_dom_agent',
        provider: value,
        secret: { api_key: secret },
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
      ...(secret
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

function applySaved(transaction: Schema['SettingsTransaction'], result: Schema['SettingsTransactionResult']) {
  for (const change of transaction.settings!) {
    const revision = result.settings.find(row => row.domain === change.domain)!.revision
    const baseline = original!.settings.find(row => row.domain === change.domain)!
    baseline.payload = change.payload
    baseline.revision = revision
    document.value!.settings.find(row => row.domain === change.domain)!.revision = revision
  }
  for (const key of result.credentials) {
    const index = document.value!.credentials.findIndex(row => row.domain === key.domain && row.provider === key.provider)
    if (index < 0) document.value!.credentials.push(key)
    else document.value!.credentials[index] = key
    const submitted = transaction.credentialEdits!.find(row => row.provider === key.provider)!
    if (secretDrafts.value[key.provider]?.trim() === submitted.secret.api_key) {
      delete secretDrafts.value[key.provider]
    }
  }
  for (const change of transaction.providerSettings!) {
    const revision = result.providerSettings.find(row => row.domain === change.domain && row.provider === change.provider)!.revision
    const key = result.credentials.find(row => row.domain === change.domain && row.provider === change.provider)
    const entry: Schema['ProviderSettingEntry'] = {
      domain: change.domain, provider: change.provider, payload: change.payload,
      schemaVersion: change.schemaVersion, revision,
      credentialVersionId: key?.credentialVersionId ?? change.credentialVersionId ?? null,
    }
    const index = document.value!.providerSettings.findIndex(row => row.domain === change.domain && row.provider === change.provider)
    if (index < 0) document.value!.providerSettings.push(entry)
    else document.value!.providerSettings[index] = entry
    if (JSON.stringify(drafts.value[change.provider]) === JSON.stringify(change.payload) && !secretDrafts.value[change.provider]?.trim()) {
      dirtyProviders.delete(change.provider)
    }
  }
}

async function persist(reportInvalid: boolean): Promise<boolean> {
  try {
    while (true) {
      clearTimeout(autoSaveTimer)
      if (form.value && !(reportInvalid ? form.value.reportValidity() : form.value.checkValidity())) return false
      const transaction = buildTransaction()
      if (!transaction.settings!.length && !transaction.providerSettings!.length) {
        saveError.value = ''
        return true
      }
      saving.value = true
      emit('saving', true)
      const result = await props.api<Schema['SettingsTransactionResult']>('/settings/transactions', 'PUT', transaction)
      applySaved(transaction, result)
      saveError.value = ''
    }
  } catch (error) {
    clearTimeout(autoSaveTimer)
    saveError.value = `自动保存失败，修改尚未保存。${error instanceof Error ? error.message : ''}`
    return false
  } finally {
    saving.value = false
    emit('saving', false)
  }
}
function save(reportInvalid = true): Promise<boolean> {
  clearTimeout(autoSaveTimer)
  if (savePromise) return savePromise
  if (busy.value) return Promise.resolve(false)
  if (!document.value) return Promise.resolve(true)
  savePromise = persist(reportInvalid).finally(() => { savePromise = null })
  return savePromise
}
onMounted(load)
onBeforeUnmount(() => { void save(false) })
defineExpose({ save })
</script>

<style scoped>
.browser-extension-settings fieldset {
  border: 0;
  padding: 0;
  margin: 0;
  min-width: 0;
}
.browser-extension-settings__intro {
  color: var(--color-text-supporting);
  font-size: 13px;
  line-height: 1.6;
  margin: 0 0 14px;
}
.browser-extension-settings__agent {
  margin-top: 25px;
}
.browser-extension-settings__agent > summary {
  cursor: pointer;
  color: var(--color-action-primary);
  font-weight: 600;
  margin-bottom: 15px;
}
</style>
