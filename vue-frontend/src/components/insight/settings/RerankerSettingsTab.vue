<script setup lang="ts">
/**
 * Reranker 设置选项卡组件
 */
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

const provider = ref(insightStore.config.reranker.provider)
const apiKey = ref(insightStore.config.reranker.apiKey)
const model = ref(insightStore.config.reranker.model)
const baseUrl = ref(insightStore.config.reranker.baseUrl)
const topK = ref(insightStore.config.reranker.topK)
const transportRetries = ref(insightStore.config.reranker.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.reranker.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.reranker.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => provider.value === 'custom')

function onProviderChange(): void {
  const newProvider = provider.value
  const oldProvider = insightStore.config.reranker.provider
  
  if (oldProvider !== newProvider) {
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
  
  const apiProvider = provider.value === 'custom' ? 'custom_openai' : provider.value
  isFetchingModels.value = true
  
  try {
    const response = await insightApi.fetchModels(apiProvider, apiKey.value, baseUrl.value || undefined)
    if (response.success && response.models?.length) {
      models.value = response.models
      modelSelectVisible.value = true
      emit('showMessage', `获取到 ${response.models.length} 个模型`, 'success')
    } else {
      emit('showMessage', response.message || '未获取到模型列表', 'error')
      modelSelectVisible.value = false
    }
  } catch {
    emit('showMessage', '获取模型列表失败', 'error')
    modelSelectVisible.value = false
  } finally {
    isFetchingModels.value = false
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
    
    <div class="form-group">
      <label>服务商</label>
      <CustomSelect v-model="provider" :options="RERANKER_PROVIDER_OPTIONS" @change="onProviderChange" />
    </div>
    
    <div class="form-group">
      <label>API Key</label>
      <input v-model="apiKey" type="password" placeholder="输入 API Key">
    </div>
    
    <div class="form-group">
      <label>模型</label>
      <div class="model-input-row">
        <input v-model="model" type="text" placeholder="例如: jina-reranker-v2-base-multilingual">
        <button class="btn btn-secondary btn-sm fetch-btn" :disabled="isFetchingModels" @click="fetchModels">
          {{ isFetchingModels ? '获取中...' : '🔍 获取模型' }}
        </button>
      </div>
      <div v-if="modelSelectVisible && models.length > 0" class="model-select-container">
        <select class="model-select" :value="model" @change="onModelSelected(($event.target as HTMLSelectElement).value)">
          <option value="">-- 选择模型 --</option>
          <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name || m.id }}</option>
        </select>
        <span class="model-count">共 {{ models.length }} 个模型</span>
      </div>
    </div>
    
    <div v-if="showBaseUrl" class="form-group">
      <label>Base URL</label>
      <input v-model="baseUrl" type="text" placeholder="自定义 API 地址">
    </div>
    
    <div class="form-group">
      <label>Top K</label>
      <input v-model.number="topK" type="number" min="1" max="20">
      <p class="form-hint">重排序后返回的结果数量</p>
    </div>

    <div class="form-group">
      <label>传输重试次数</label>
      <input v-model.number="transportRetries" type="number" min="0" max="100">
      <p class="form-hint">网络超时、连接错误、429/5xx 的自动重试次数，默认 10</p>
    </div>

    <div class="form-group">
      <label>业务重试次数</label>
      <input v-model.number="businessRetries" type="number" min="0" max="100">
      <p class="form-hint">当重排序结果为空或结构无效时的额外重试次数，默认 10</p>
    </div>

    <div class="form-group">
      <label>单次请求超时（秒）</label>
      <input v-model.number="timeoutSeconds" type="number" min="0" max="3600" step="1">
      <p class="form-hint">0 表示不限制；大于 0 时作为单次重排序 HTTP 请求超时</p>
    </div>
    
    <button class="btn btn-secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </button>
  </div>
</template>
