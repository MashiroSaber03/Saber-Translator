<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { getFontMatchConfiguration } from '@/api/v2/fontMatching'
import { fetchV2ModelCatalog, runV2ConnectionTest } from '@/api/v2/settings'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
import { useSettingsStore } from '@/stores/settings'
import type { FontMatchingProviderConfig } from '@/stores/settings/types'
import AiProviderCredentialFields from './AiProviderCredentialFields.vue'
import AiProviderSelectField from './AiProviderSelectField.vue'

const props = defineProps<{ active: boolean }>()
const emit = defineEmits<{ manageModels: [] }>()
const settings = useSettingsStore()
const publicAccess = usePublicUserAccess()
const defaults = {
  openai: { modelName: 'gpt-6-luna', baseUrl: 'https://api.openai.com/v1' },
  jev: { modelName: 'jev-1.14.0', baseUrl: 'https://api.typesafe.ai/v1' },
  valen: { modelName: '', baseUrl: '' },
} as const
const providers = [
  { value: '', label: '请选择决策服务商', disabled: true },
  { value: 'openai', label: 'GPT · OpenAI Decisions' },
  { value: 'jev', label: 'Jev · TypeSafe System One' },
  { value: 'valen', label: 'Valen · 本地' },
]
const valenModels = ['Valen-0.8B', 'Valen-2B'] as const
const provider = computed(() => settings.fontMatchingProvider)
const config = computed(() => settings.providerConfigs.fontMatching[provider.value] ?? {})
const defaultConfig = computed(() => defaults[provider.value as keyof typeof defaults])
const readiness = ref<Awaited<ReturnType<typeof getFontMatchConfiguration>>>()
const models = ref<string[]>([])
const pending = ref<'catalog' | 'connection' | null>(null)
const status = ref<{ success: boolean; message: string }>()
const localOptions = computed(() => [
  { value: '', label: '请选择本地模型', disabled: true },
  ...valenModels.map(model => ({
    value: model,
    label: readiness.value?.installedValen.includes(model) ? model : `${model}（未安装）`,
    disabled: !readiness.value?.installedValen.includes(model),
  })),
])
let requestVersion = 0

function update(field: keyof FontMatchingProviderConfig, value: string | number): void {
  if (typeof value !== 'string' || !defaultConfig.value) return
  settings.providerConfigs.fontMatching[provider.value] = {
    modelName: defaultConfig.value.modelName, customBaseUrl: '', ...config.value, [field]: value,
  }
}

function selectProvider(value: string): void {
  if (!Object.hasOwn(defaults, value)) return
  settings.fontMatchingProvider = value
  if (!settings.providerConfigs.fontMatching[value]) {
    update('modelName', defaults[value as keyof typeof defaults].modelName)
  }
}

watch([provider, () => config.value.customBaseUrl, () => config.value.apiKey, () => props.active], () => {
  requestVersion++
  pending.value = null
  models.value = []
  status.value = undefined
}, { flush: 'sync' })

watch(() => config.value.modelName, () => {
  if (pending.value === 'connection') {
    requestVersion++
    pending.value = null
  }
  status.value = undefined
}, { flush: 'sync' })

watch([() => props.active, provider], async ([active, selected], _, onCleanup) => {
  if (!active || selected !== 'valen') return
  const request = new AbortController()
  onCleanup(() => request.abort())
  try {
    const result = await getFontMatchConfiguration(request.signal)
    if (!request.signal.aborted) readiness.value = result
  } catch (error) {
    if (!request.signal.aborted) {
      status.value = { success: false, message: error instanceof Error ? error.message : '读取模型状态失败' }
    }
  }
}, { immediate: true })

async function diagnose(kind: 'catalog' | 'connection'): Promise<void> {
  const id = ++requestVersion
  pending.value = kind
  status.value = undefined
  const request = {
    domain: 'font_matching', provider: provider.value, baseUrl: config.value.customBaseUrl || undefined,
    ...(config.value.apiKey?.trim() ? { secret: { api_key: config.value.apiKey.trim() } } : {}),
  }
  try {
    if (kind === 'catalog') {
      const result = await fetchV2ModelCatalog(request)
      if (id !== requestVersion) return
      models.value = result.models.map(model => model.id)
      if (!models.value.length) status.value = { success: false, message: '该账户未返回支持图片和原生 Score 的模型' }
    } else {
      const result = await runV2ConnectionTest('font_matching', { ...request, model: config.value.modelName })
      if (id === requestVersion) status.value = { ...result, message: result.message ?? '连接测试已完成' }
    }
  } catch (error) {
    if (id === requestVersion) status.value = {
      success: false, message: error instanceof Error ? error.message : '无法连接字体匹配服务',
    }
  } finally {
    if (id === requestVersion) pending.value = null
  }
}
</script>

<template>
  <ProductFormSection>
    <template #title>字体匹配</template>
    <UiFormGrid>
      <AiProviderSelectField
        :model-value="provider" input-id="fontMatchProvider" :options="providers" @change="selectProvider"
      />
      <AiProviderCredentialFields
        v-if="provider && provider !== 'valen'"
        :api-key="config.apiKey ?? ''" api-key-input-id="fontMatchApiKey"
        :base-url="config.customBaseUrl ?? ''" base-url-input-id="fontMatchBaseUrl"
        :include-base-url="false"
        @update:api-key="value => update('apiKey', value)"
      />
    </UiFormGrid>
    <template v-if="provider && provider !== 'valen'">
      <AiProviderCredentialFields
        api-key="" api-key-input-id="fontMatchApiKey" :include-api-key="false"
        :base-url="config.customBaseUrl ?? ''" base-url-input-id="fontMatchBaseUrl"
        base-url-label="服务地址" :base-url-placeholder="defaultConfig?.baseUrl" show-base-url
        @update:base-url="value => update('customBaseUrl', value)"
      />
      <UiField variant="settings" label="模型" control-id="fontMatchModel">
        <UiModelPicker
          input-id="fontMatchModel" :model-value="config.modelName ?? ''" :placeholder="defaultConfig?.modelName"
          :fetching="pending === 'catalog'" :fetch-disabled="pending !== null"
          :options="models.map(id => ({ label: id, value: id }))"
          @update:model-value="value => update('modelName', value)" @fetch="diagnose('catalog')"
        />
      </UiField>
      <ProductActionRow justify="start">
        <UiButton :disabled="pending !== null || !config.modelName?.trim()" @click="diagnose('connection')">
          {{ pending === 'connection' ? '正在测试…' : '测试连接' }}
        </UiButton>
      </ProductActionRow>
    </template>
    <template v-else-if="provider === 'valen'">
      <UiField variant="settings" label="模型" control-id="fontMatchValen">
        <UiSelect
          id="fontMatchValen" :model-value="config.modelName ?? ''" :options="localOptions"
          @change="value => update('modelName', value)"
        />
      </UiField>
      <ProductStatusBanner tone="neutral" role="status">
        {{ readiness?.valenRuntimeReady ? 'Valen 推理环境已就绪' : 'Valen 需要已安装的权重和独立推理环境' }}
        <template v-if="!publicAccess.restricted()" #actions>
          <UiButton size="sm" @click="emit('manageModels')">模型管理</UiButton>
        </template>
      </ProductStatusBanner>
    </template>
    <ProductStatusBanner v-if="status" :tone="status.success ? 'success' : 'warning'" role="status">
      {{ status.message }}
    </ProductStatusBanner>
  </ProductFormSection>
</template>
