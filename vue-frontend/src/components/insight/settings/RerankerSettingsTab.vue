<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref, computed } from 'vue'
import { getProviderDefaultModel } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { StoreRerankerConfig } from '@/types/insight'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import { useInsightModelFetch } from './useInsightModelFetch'
import { useInsightConnectionTest } from './useInsightConnectionTest'
import { RERANKER_PROVIDER_OPTIONS } from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
  (e: 'update:config', config: StoreRerankerConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const provider = ref(insightStore.config.reranker.provider)
const apiKey = ref(insightStore.config.reranker.apiKey)
const model = ref(insightStore.config.reranker.model)
const baseUrl = ref(insightStore.config.reranker.baseUrl ?? '')
const transportRetries = ref(insightStore.config.reranker.transportRetries ?? 1)
const businessRetries = ref(insightStore.config.reranker.businessRetries ?? 0)
const timeoutSeconds = ref(insightStore.config.reranker.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => provider.value === 'custom')
const hasStoredCredential = computed(() =>
  insightApi.hasInsightCredential('insight_reranker', provider.value)
)
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

const { isTesting, testConnection } = useInsightConnectionTest({
  sources: [
    provider,
    apiKey,
    model,
    baseUrl,
    transportRetries,
    businessRetries,
    timeoutSeconds,
  ],
  snapshot: () => ({
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    transportRetries: transportRetries.value,
    businessRetries: businessRetries.value,
    timeoutSeconds: timeoutSeconds.value,
  }),
  request: snapshot => insightApi.testRerankerConnection({
    provider: snapshot.provider,
    api_key: snapshot.apiKey,
    model: snapshot.model,
    base_url: snapshot.baseUrl,
  }),
  successMessage: 'Reranker 连接成功',
  emitMessage: (message, type) => emit('showMessage', message, type),
})

function onProviderChange(): void {
  const newProvider = provider.value
  invalidateModelFetch()

  applyDraftConfig(insightStore.switchRerankerProviderDraft(buildDraftConfig()))

  if (!model.value) {
    const defaultModel = getProviderDefaultModel(newProvider, 'reranker')
    if (defaultModel) model.value = defaultModel
  }
}

function buildDraftConfig(): StoreRerankerConfig {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
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
  transportRetries.value = config.transportRetries ?? 1
  businessRetries.value = config.businessRetries ?? 0
  timeoutSeconds.value = config.timeoutSeconds ?? 0
}

useInsightSettingsDraft<StoreRerankerConfig>({
  sources: [
    provider,
    apiKey,
    model,
    baseUrl,
    transportRetries,
    businessRetries,
    timeoutSeconds,
  ],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.reranker,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
})
</script>

<template>
  <InsightSettingsPanel
    class="reranker-settings-tab"
    description="Reranker（重排序模型）用于对搜索结果进行重新排序，提高问答准确性。"
  >
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
      test-placement="panel-end"
      :testing="isTesting"
      @provider-change="onProviderChange"
      @model-change="selectModel"
      @fetch="fetchModels"
      @test="testConnection"
    />

    <UiField
      variant="settings"
      label="传输重试次数"
      hint="网络超时、连接错误、429/5xx 默认重试 1 次"
      control-id="reranker-transport-retries"
    >
      <UiNumberField
        v-model="transportRetries"
        input-id="reranker-transport-retries"
        :min="0"
      />
    </UiField>

    <UiField
      variant="settings"
      label="业务重试次数"
      hint="空结果或结构无效时默认不额外重试"
      control-id="reranker-business-retries"
    >
      <UiNumberField
        v-model="businessRetries"
        input-id="reranker-business-retries"
        :min="0"
      />
    </UiField>

    <UiField
      variant="settings"
      label="单次请求超时（秒）"
      hint="0 表示不限制；大于 0 时作为单次重排序 HTTP 请求超时"
      control-id="reranker-timeout-seconds"
    >
      <UiNumberField
        v-model="timeoutSeconds"
        input-id="reranker-timeout-seconds"
        :min="0"
        :step="0.1"
      />
    </UiField>
  </InsightSettingsPanel>
</template>

<style scoped>
.reranker-settings-tab {
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-number-field-text-align: left;
}
</style>
