<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { computed, ref } from 'vue'
import { providerRequiresApiKey, providerRequiresBaseUrl, providerRequiresModel, getProviderBaseUrl } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import type { StoreImageGenConfig } from '@/types/insight'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import {
  IMAGE_GEN_PROVIDER_OPTIONS,
  PROVIDER_DEFAULT_MODELS,
} from './types'

const emit = defineEmits<{
  (e: 'update:config', config: StoreImageGenConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()
const initialProvider = insightStore.config.imageGen?.provider || 'gpt2api'

const provider = ref(initialProvider)
const apiKey = ref(insightStore.config.imageGen?.apiKey || '')
const model = ref(insightStore.config.imageGen?.model ?? (PROVIDER_DEFAULT_MODELS[initialProvider]?.imageGen || 'gpt-image-2'))
const baseUrl = ref(insightStore.config.imageGen?.baseUrl || '')
const transportRetries = ref(insightStore.config.imageGen?.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.imageGen?.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.imageGen?.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => providerRequiresBaseUrl(provider.value))
const showModelWarning = computed(() => providerRequiresModel(provider.value) && !model.value.trim())

function getDefaultModel(providerId: string): string {
  return PROVIDER_DEFAULT_MODELS[providerId]?.imageGen || ''
}

function onProviderChange(): void {
  const newProvider = provider.value

  applyDraftConfig(insightStore.switchImageGenProviderDraft(buildDraftConfig()))

  if (!model.value) {
    model.value = getDefaultModel(newProvider)
  }
}

function buildDraftConfig(): StoreImageGenConfig {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: baseUrl.value,
    transportRetries: transportRetries.value,
    businessRetries: businessRetries.value,
    timeoutSeconds: timeoutSeconds.value,
  }
}

function applyDraftConfig(config: StoreImageGenConfig): void {
  provider.value = config.provider || 'gpt2api'
  apiKey.value = config.apiKey || ''
  model.value = config.model ?? getDefaultModel(provider.value)
  baseUrl.value = config.baseUrl || getProviderBaseUrl(provider.value, 'imageGen')
  transportRetries.value = config.transportRetries ?? 10
  businessRetries.value = config.businessRetries ?? 10
  timeoutSeconds.value = config.timeoutSeconds ?? 0
}

useInsightSettingsDraft<StoreImageGenConfig>({
  sources: [provider, apiKey, model, baseUrl, transportRetries, businessRetries, timeoutSeconds],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.imageGen,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
})
</script>

<template>
  <InsightSettingsPanel description="生图模型服务商保留为可扩展选择器，当前支持 gpt2api 与 New API，带参考图时会自动适配到其图片编辑路由。">
    <InsightModelProviderSection
      v-model:provider="provider"
      v-model:api-key="apiKey"
      v-model:model="model"
      v-model:base-url="baseUrl"
      :provider-options="IMAGE_GEN_PROVIDER_OPTIONS"
      :show-api-key="providerRequiresApiKey(provider)"
      credential-id="insight-imagegen-api-key"
      provider-input-id="insight-imagegen-provider"
      model-input-id="insight-imagegen-model"
      base-url-input-id="insight-imagegen-base-url"
      model-placeholder="例如: gpt-image-2"
      model-hint="默认推荐使用当前服务商的默认生图模型。"
      :model-error="showModelWarning ? '当前服务商需要手动填写模型名。' : ''"
      :show-base-url="showBaseUrl"
      base-url-placeholder="例如: http://127.0.0.1:17200 或 http://127.0.0.1:17200/v1"
      :show-fetch="false"
      :show-test="false"
      @provider-change="onProviderChange"
    />

    <UiField variant="settings" label="传输重试次数" hint="网络超时、连接错误、429/5xx 的自动重试次数，默认 10" control-id="insight-imagegen-transport-retries">
      <UiNumberField v-model="transportRetries" input-id="insight-imagegen-transport-retries" :min="0" :max="100" />
    </UiField>

    <UiField variant="settings" label="业务重试次数" hint="当接口返回空图片结果或结果不可解析时的额外重试次数，默认 10" control-id="insight-imagegen-business-retries">
      <UiNumberField v-model="businessRetries" input-id="insight-imagegen-business-retries" :min="0" :max="100" />
    </UiField>

    <UiField variant="settings" label="单次请求超时（秒）" hint="0 表示不限制；大于 0 时作为单次生图 HTTP 请求超时" control-id="insight-imagegen-timeout-seconds">
      <UiNumberField v-model="timeoutSeconds" input-id="insight-imagegen-timeout-seconds" :min="0" :max="3600" :step="1" />
    </UiField>
  </InsightSettingsPanel>
</template>
