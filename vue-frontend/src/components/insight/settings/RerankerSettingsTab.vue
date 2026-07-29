<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref, computed } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { StoreRerankerConfig } from '@/types/insight'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import { useInsightModelFetch } from './useInsightModelFetch'
import {
  RERANKER_PROVIDER_OPTIONS,
  RERANKER_DEFAULT_MODELS,
} from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
  (e: 'update:config', config: StoreRerankerConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const isTesting = ref(false)

const provider = ref(insightStore.config.reranker.provider)
const apiKey = ref(insightStore.config.reranker.apiKey)
const model = ref(insightStore.config.reranker.model)
const baseUrl = ref(insightStore.config.reranker.baseUrl ?? '')
const topK = ref(insightStore.config.reranker.topK ?? 5)
const transportRetries = ref(insightStore.config.reranker.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.reranker.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.reranker.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => provider.value === 'custom')
const hasStoredCredential = computed(() => (
  insightApi.hasInsightCredential('insight_reranker', provider.value)
))
const {
  isFetchingModels,
  modelOptions,
  modelCount,
  invalidateModelFetch,
  fetchModels,
  selectModel,
} = useInsightModelFetch({
  domain: 'insight_reranker',
  provider,
  apiKey,
  baseUrl,
  model,
  requiresApiKey: () => true,
  emitMessage: (message, type) => emit('showMessage', message, type),
})

function onProviderChange(): void {
  const newProvider = provider.value
  invalidateModelFetch()

  applyDraftConfig(insightStore.switchRerankerProviderDraft(buildDraftConfig()))

  if (!model.value) {
    const defaultModel = RERANKER_DEFAULT_MODELS[newProvider]
    if (defaultModel) model.value = defaultModel
  }
}

async function testConnection(): Promise<void> {
  if (isTesting.value) return
  isTesting.value = true

  try {
    const response = await insightApi.testRerankerConnection({
      provider: provider.value,
      api_key: apiKey.value,
      model: model.value,
      base_url: baseUrl.value || undefined,
      transport_retries: transportRetries.value,
      business_retries: businessRetries.value,
      timeout_seconds: timeoutSeconds.value,
    })
    emit('showMessage', response.success ? 'Reranker 连接成功' : '连接失败: ' + (response.error || '未知错误'), response.success ? 'success' : 'error')
  } catch {
    emit('showMessage', '测试失败', 'error')
  } finally {
    isTesting.value = false
  }
}

function buildDraftConfig(): StoreRerankerConfig {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: baseUrl.value,
    topK: topK.value,
    transportRetries: transportRetries.value,
    businessRetries: businessRetries.value,
    timeoutSeconds: timeoutSeconds.value,
  }
}

function applyDraftConfig(config: StoreRerankerConfig): void {
  provider.value = config.provider
  apiKey.value = config.apiKey
  model.value = config.model
  baseUrl.value = config.baseUrl ?? ''
  topK.value = config.topK ?? 5
  transportRetries.value = config.transportRetries ?? 10
  businessRetries.value = config.businessRetries ?? 10
  timeoutSeconds.value = config.timeoutSeconds ?? 0
}

useInsightSettingsDraft<StoreRerankerConfig>({
  sources: [provider, apiKey, model, baseUrl, topK, transportRetries, businessRetries, timeoutSeconds],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.reranker,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
})
</script>

<template>
  <InsightSettingsPanel description="Reranker（重排序模型）用于对搜索结果进行重新排序，提高问答准确性。">
    <InsightModelProviderSection
      v-model:provider="provider"
      v-model:api-key="apiKey"
      v-model:model="model"
      v-model:base-url="baseUrl"
      :provider-options="RERANKER_PROVIDER_OPTIONS"
      show-api-key
      :has-stored-credential="hasStoredCredential"
      credential-id="reranker-api-key"
      provider-input-id="reranker-provider"
      model-input-id="reranker-model"
      base-url-input-id="reranker-base-url"
      model-placeholder="例如: jina-reranker-v2-base-multilingual"
      fetch-variant="primary"
      :fetching-models="isFetchingModels"
      :model-options="modelOptions"
      :model-count="modelCount"
      :show-base-url="showBaseUrl"
      :testing="isTesting"
      @provider-change="onProviderChange"
      @model-change="selectModel"
      @fetch="fetchModels"
      @test="testConnection"
    />

    <UiField variant="settings" label="Top K" hint="重排序后返回的结果数量" control-id="reranker-top-k">
      <UiNumberField v-model="topK" input-id="reranker-top-k" :min="1" :max="20" />
    </UiField>

    <UiField variant="settings" label="传输重试次数" hint="网络超时、连接错误、429/5xx 的自动重试次数，默认 10" control-id="reranker-transport-retries">
      <UiNumberField v-model="transportRetries" input-id="reranker-transport-retries" :min="0" :max="100" />
    </UiField>

    <UiField variant="settings" label="业务重试次数" hint="当重排序结果为空或结构无效时的额外重试次数，默认 10" control-id="reranker-business-retries">
      <UiNumberField v-model="businessRetries" input-id="reranker-business-retries" :min="0" :max="100" />
    </UiField>

    <UiField variant="settings" label="单次请求超时（秒）" hint="0 表示不限制；大于 0 时作为单次重排序 HTTP 请求超时" control-id="reranker-timeout-seconds">
      <UiNumberField v-model="timeoutSeconds" input-id="reranker-timeout-seconds" :min="0" :max="3600" :step="1" />
    </UiField>
  </InsightSettingsPanel>
</template>
