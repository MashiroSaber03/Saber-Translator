<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

import UiButton from '@/components/ui/UiButton.vue'
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import {
  RERANKER_PROVIDER_OPTIONS,
  RERANKER_DEFAULT_MODELS,
  SUPPORTED_FETCH_PROVIDERS,
  type ModelInfo
} from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
}>()

const insightStore = useInsightStore()

const isTesting = ref(false)
const isFetchingModels = ref(false)
const models = ref<ModelInfo[]>([])
const modelSelectVisible = ref(false)
let modelFetchRequestId = 0

const provider = ref(insightStore.config.reranker.provider)
const apiKey = ref(insightStore.config.reranker.apiKey)
const model = ref(insightStore.config.reranker.model)
const baseUrl = ref(insightStore.config.reranker.baseUrl)
const topK = ref(insightStore.config.reranker.topK)
const transportRetries = ref(insightStore.config.reranker.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.reranker.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.reranker.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => provider.value === 'custom')

function resetModelOptions(): void {
  models.value = []
  modelSelectVisible.value = false
}

function onProviderChange(): void {
  const newProvider = provider.value
  const previousProvider = insightStore.config.reranker.provider
  modelFetchRequestId += 1
  isFetchingModels.value = false
  resetModelOptions()

  if (previousProvider !== newProvider) {
    insightStore.config.reranker.apiKey = apiKey.value
    insightStore.config.reranker.model = model.value
    insightStore.config.reranker.baseUrl = baseUrl.value
    insightStore.config.reranker.topK = topK.value
    insightStore.config.reranker.transportRetries = transportRetries.value
    insightStore.config.reranker.businessRetries = businessRetries.value
    insightStore.config.reranker.timeoutSeconds = timeoutSeconds.value
  }

  insightStore.setRerankerProvider(newProvider)

  apiKey.value = insightStore.config.reranker.apiKey
  model.value = insightStore.config.reranker.model
  baseUrl.value = insightStore.config.reranker.baseUrl
  topK.value = insightStore.config.reranker.topK
  transportRetries.value = insightStore.config.reranker.transportRetries ?? 10
  businessRetries.value = insightStore.config.reranker.businessRetries ?? 10
  timeoutSeconds.value = insightStore.config.reranker.timeoutSeconds ?? 0

  if (!model.value) {
    const defaultModel = RERANKER_DEFAULT_MODELS[newProvider]
    if (defaultModel) model.value = defaultModel
  }
}

async function fetchModels(): Promise<void> {
  if (!apiKey.value) {
    emit('showMessage', '请先填写 API Key', 'error')
    return
  }

  if (!SUPPORTED_FETCH_PROVIDERS.includes(provider.value)) {
    emit('showMessage', `${provider.value} 不支持自动获取模型列表`, 'error')
    return
  }

  if (provider.value === 'custom' && !baseUrl.value) {
    emit('showMessage', '自定义服务需要先填写 Base URL', 'error')
    return
  }

  isFetchingModels.value = true
  const requestId = ++modelFetchRequestId
  const requestProvider = provider.value
  const requestApiKey = apiKey.value
  const requestBaseUrl = baseUrl.value || undefined
  const isCurrentRequest = () => (
    modelFetchRequestId === requestId &&
    provider.value === requestProvider &&
    apiKey.value === requestApiKey &&
    (baseUrl.value || undefined) === requestBaseUrl
  )

  try {
    const response = await insightApi.fetchModels(requestProvider, requestApiKey, requestBaseUrl)
    if (!isCurrentRequest()) return

    if (response.success && response.models?.length) {
      models.value = response.models
      modelSelectVisible.value = true
      emit('showMessage', `获取到 ${response.models.length} 个模型`, 'success')
    } else {
      emit('showMessage', response.message || '未获取到模型列表', 'error')
      modelSelectVisible.value = false
    }
  } catch {
    if (isCurrentRequest()) {
      emit('showMessage', '获取模型列表失败', 'error')
      modelSelectVisible.value = false
    }
  } finally {
    if (modelFetchRequestId === requestId) {
      isFetchingModels.value = false
    }
  }
}

function onModelSelected(modelId: string): void {
  if (modelId) model.value = modelId
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

function getConfig() {
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

function syncFromStore(): void {
  provider.value = insightStore.config.reranker.provider
  apiKey.value = insightStore.config.reranker.apiKey
  model.value = insightStore.config.reranker.model
  baseUrl.value = insightStore.config.reranker.baseUrl
  topK.value = insightStore.config.reranker.topK
  transportRetries.value = insightStore.config.reranker.transportRetries ?? 10
  businessRetries.value = insightStore.config.reranker.businessRetries ?? 10
  timeoutSeconds.value = insightStore.config.reranker.timeoutSeconds ?? 0
}

defineExpose({ getConfig, syncFromStore })
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">Reranker（重排序模型）用于对搜索结果进行重新排序，提高问答准确性。</p>

    <div class="insight-settings-field">
      <label>服务商</label>
      <CustomSelect v-model="provider" :options="RERANKER_PROVIDER_OPTIONS" @change="onProviderChange" />
    </div>

    <div class="insight-settings-field">
      <label>API Key</label>
      <UiInput v-model="apiKey" data-testid="reranker-api-key" type="password" placeholder="输入 API Key" />
    </div>

    <div class="insight-settings-field">
      <label>模型</label>
      <div class="model-input-row">
        <UiInput v-model="model" data-testid="reranker-model" type="text" placeholder="例如: jina-reranker-v2-base-multilingual" class="model-field-input" />
        <UiButton variant="secondary" class="fetch-btn" :disabled="isFetchingModels" @click="fetchModels" size="sm">
          {{ isFetchingModels ? '获取中...' : '🔍 获取模型' }}
        </UiButton>
      </div>
      <div v-if="modelSelectVisible && models.length > 0" class="model-select-container">
        <UiSelect class="model-select" :model-value="model" @change="onModelSelected">
          <option value="">-- 选择模型 --</option>
          <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name || m.id }}</option>
        </UiSelect>
        <span class="model-count">共 {{ models.length }} 个模型</span>
      </div>
    </div>

    <div v-if="showBaseUrl" class="insight-settings-field">
      <label>Base URL</label>
      <UiInput v-model="baseUrl" type="text" placeholder="自定义 API 地址" />
    </div>

    <div class="insight-settings-field">
      <label>Top K</label>
      <UiInput v-model.number="topK" data-testid="reranker-top-k" type="number" min="1" max="20" />
      <p class="form-hint">重排序后返回的结果数量</p>
    </div>

    <div class="insight-settings-field">
      <label>传输重试次数</label>
      <UiInput v-model.number="transportRetries" data-testid="reranker-transport-retries" type="number" min="0" max="100" />
      <p class="form-hint">网络超时、连接错误、429/5xx 的自动重试次数，默认 10</p>
    </div>

    <div class="insight-settings-field">
      <label>业务重试次数</label>
      <UiInput v-model.number="businessRetries" data-testid="reranker-business-retries" type="number" min="0" max="100" />
      <p class="form-hint">当重排序结果为空或结构无效时的额外重试次数，默认 10</p>
    </div>

    <div class="insight-settings-field">
      <label>单次请求超时（秒）</label>
      <UiInput v-model.number="timeoutSeconds" data-testid="reranker-timeout-seconds" type="number" min="0" max="3600" step="1" />
      <p class="form-hint">0 表示不限制；大于 0 时作为单次重排序 HTTP 请求超时</p>
    </div>

    <UiButton variant="secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </UiButton>
  </div>
</template>

<style scoped>
.insight-settings-content {
  --ui-input-padding: 10px 12px;
  --ui-input-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-input-radius: 6px;
  --ui-input-font-size: 14px;
  --ui-input-background: var(--color-surface-input, var(--color-surface-base));
  --ui-input-color: var(--color-text-default);
  --ui-input-focus-border: var(--color-border-brand);
  --ui-input-focus-shadow: var(--color-focus-brand-soft);
  --ui-select-padding: 8px 12px;
  --ui-select-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-select-radius: 4px;
  --ui-select-font-size: 13px;
  --ui-select-background: var(--color-surface-input, var(--color-surface-base));
  --ui-select-color: var(--color-text-default);
  --ui-select-focus-shadow: var(--color-focus-brand-soft);
  --ui-button-padding: 10px 16px;
  --ui-button-radius: 6px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;

  padding: 16px 0;
  min-height: 300px;
}

.insight-settings-content .settings-hint {
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 13px;
  margin-bottom: 16px;
  padding: 8px 12px;
  background: var(--color-surface-muted);
  border-radius: 4px;
}

.insight-settings-content .insight-settings-field {
  margin-bottom: 16px;
}

.insight-settings-content .insight-settings-field label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
  color: var(--color-text-default);
}

.insight-settings-content .form-hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-content .model-field-input {
  flex: 1;
}

.insight-settings-content .fetch-btn {
  white-space: nowrap;
  flex-shrink: 0;
}

.insight-settings-content .model-select-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-radius: 6px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .model-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  font-size: 13px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default);
  cursor: pointer;
}

.insight-settings-content .model-select:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.insight-settings-content .model-count {
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  white-space: nowrap;
}
</style>
