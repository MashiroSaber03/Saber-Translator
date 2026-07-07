<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref, computed } from 'vue'
import { providerRequiresApiKey } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { StoreEmbeddingConfig } from '@/types/insight'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import { useInsightModelFetch } from './useInsightModelFetch'
import {
  EMBEDDING_PROVIDER_OPTIONS,
  EMBEDDING_DEFAULT_MODELS,
} from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
  (e: 'update:config', config: StoreEmbeddingConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const isTesting = ref(false)

const provider = ref(insightStore.config.embedding.provider)
const apiKey = ref(insightStore.config.embedding.apiKey)
const model = ref(insightStore.config.embedding.model)
const baseUrl = ref(insightStore.config.embedding.baseUrl)
const rpmLimit = ref(insightStore.config.embedding.rpmLimit)
const transportRetries = ref(insightStore.config.embedding.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.embedding.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.embedding.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => provider.value === 'custom')
const {
  isFetchingModels,
  modelOptions,
  modelCount,
  invalidateModelFetch,
  fetchModels,
  selectModel,
} = useInsightModelFetch({
  provider,
  apiKey,
  baseUrl,
  model,
  emitMessage: (message, type) => emit('showMessage', message, type),
})

function onProviderChange(): void {
  const newProvider = provider.value
  invalidateModelFetch()

  applyDraftConfig(insightStore.switchEmbeddingProviderDraft(buildDraftConfig()))

  if (!model.value) {
    const defaultModel = EMBEDDING_DEFAULT_MODELS[newProvider]
    if (defaultModel) model.value = defaultModel
  }
}

async function testConnection(): Promise<void> {
  if (isTesting.value) return
  isTesting.value = true

  try {
    const response = await insightApi.testEmbeddingConnection({
      provider: provider.value,
      api_key: apiKey.value,
      model: model.value,
      base_url: baseUrl.value || undefined,
      rpm_limit: rpmLimit.value,
      transport_retries: transportRetries.value,
      business_retries: businessRetries.value,
      timeout_seconds: timeoutSeconds.value,
    })
    emit('showMessage', response.success ? 'Embedding 连接成功' : '连接失败: ' + (response.error || '未知错误'), response.success ? 'success' : 'error')
  } catch {
    emit('showMessage', '测试失败', 'error')
  } finally {
    isTesting.value = false
  }
}

function buildDraftConfig(): StoreEmbeddingConfig {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    rpmLimit: rpmLimit.value,
    transportRetries: transportRetries.value,
    businessRetries: businessRetries.value,
    timeoutSeconds: timeoutSeconds.value
  }
}

function applyDraftConfig(config: StoreEmbeddingConfig): void {
  provider.value = config.provider
  apiKey.value = config.apiKey
  model.value = config.model
  baseUrl.value = config.baseUrl
  rpmLimit.value = config.rpmLimit
  transportRetries.value = config.transportRetries ?? 10
  businessRetries.value = config.businessRetries ?? 10
  timeoutSeconds.value = config.timeoutSeconds ?? 0
}

useInsightSettingsDraft<StoreEmbeddingConfig>({
  sources: [provider, apiKey, model, baseUrl, rpmLimit, transportRetries, businessRetries, timeoutSeconds],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.embedding,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
})
</script>

<template>
  <InsightSettingsPanel description="Embedding（向量化模型）用于将文本转换为向量，支持语义搜索和问答功能。">
    <InsightModelProviderSection
      v-model:provider="provider"
      v-model:api-key="apiKey"
      v-model:model="model"
      v-model:base-url="baseUrl"
      :provider-options="EMBEDDING_PROVIDER_OPTIONS"
      :show-api-key="providerRequiresApiKey(provider)"
      credential-id="insight-embedding-api-key"
      provider-input-id="insight-embedding-provider"
      model-input-id="insight-embedding-model"
      base-url-input-id="insight-embedding-base-url"
      model-placeholder="例如: text-embedding-3-small"
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

    <UiField variant="settings" label="RPM 限制" hint="每分钟最大请求数，0 表示不限制" control-id="insight-embedding-rpm-limit">
      <UiNumberField v-model="rpmLimit" input-id="insight-embedding-rpm-limit" :min="0" :max="1000" />
    </UiField>

    <UiField variant="settings" label="传输重试次数" hint="网络超时、连接错误、429/5xx 的自动重试次数，默认 10" control-id="insight-embedding-transport-retries">
      <UiNumberField v-model="transportRetries" input-id="insight-embedding-transport-retries" :min="0" :max="100" />
    </UiField>

    <UiField variant="settings" label="业务重试次数" hint="当接口返回空向量或数量不匹配时的额外重试次数，默认 10" control-id="insight-embedding-business-retries">
      <UiNumberField v-model="businessRetries" input-id="insight-embedding-business-retries" :min="0" :max="100" />
    </UiField>

    <UiField variant="settings" label="单次请求超时（秒）" hint="0 表示不限制；大于 0 时作为单次 Embedding HTTP 请求超时" control-id="insight-embedding-timeout-seconds">
      <UiNumberField v-model="timeoutSeconds" input-id="insight-embedding-timeout-seconds" :min="0" :max="3600" :step="1" />
    </UiField>
  </InsightSettingsPanel>
</template>
