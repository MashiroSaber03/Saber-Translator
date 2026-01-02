<script setup lang="ts">
/**
 * Embedding 设置选项卡组件
 */
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import {
  EMBEDDING_PROVIDER_OPTIONS,
  EMBEDDING_DEFAULT_MODELS,
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

const provider = ref(insightStore.config.embedding.provider)
const apiKey = ref(insightStore.config.embedding.apiKey)
const model = ref(insightStore.config.embedding.model)
const baseUrl = ref(insightStore.config.embedding.baseUrl)
const rpmLimit = ref(insightStore.config.embedding.rpmLimit)

const showBaseUrl = computed(() => provider.value === 'custom')

function onProviderChange(): void {
  const newProvider = provider.value
  const oldProvider = insightStore.config.embedding.provider
  
  if (oldProvider !== newProvider) {
    insightStore.config.embedding.apiKey = apiKey.value
    insightStore.config.embedding.model = model.value
    insightStore.config.embedding.baseUrl = baseUrl.value
    insightStore.config.embedding.rpmLimit = rpmLimit.value
  }
  
  insightStore.setEmbeddingProvider(newProvider)
  
  apiKey.value = insightStore.config.embedding.apiKey
  model.value = insightStore.config.embedding.model
  baseUrl.value = insightStore.config.embedding.baseUrl
  rpmLimit.value = insightStore.config.embedding.rpmLimit
  
  if (!model.value) {
    const defaultModel = EMBEDDING_DEFAULT_MODELS[newProvider]
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
    const response = await insightApi.testEmbeddingConnection({
      provider: provider.value,
      api_key: apiKey.value,
      model: model.value,
      base_url: baseUrl.value || undefined
    })
    emit('showMessage', response.success ? 'Embedding 连接成功' : '连接失败: ' + (response.error || '未知错误'), response.success ? 'success' : 'error')
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
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    rpmLimit: rpmLimit.value
  }
}

function syncFromStore(): void {
  provider.value = insightStore.config.embedding.provider
  apiKey.value = insightStore.config.embedding.apiKey
  model.value = insightStore.config.embedding.model
  baseUrl.value = insightStore.config.embedding.baseUrl
  rpmLimit.value = insightStore.config.embedding.rpmLimit
}

defineExpose({ getConfig, syncFromStore })
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">Embedding（向量化模型）用于将文本转换为向量，支持语义搜索和问答功能。</p>
    
    <div class="form-group">
      <label>服务商</label>
      <CustomSelect v-model="provider" :options="EMBEDDING_PROVIDER_OPTIONS" @change="onProviderChange" />
    </div>
    
    <div class="form-group">
      <label>API Key</label>
      <input v-model="apiKey" type="password" placeholder="输入 API Key">
    </div>
    
    <div class="form-group">
      <label>模型</label>
      <div class="model-input-row">
        <input v-model="model" type="text" placeholder="例如: text-embedding-3-small">
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
      <label>RPM 限制</label>
      <input v-model.number="rpmLimit" type="number" min="0" max="1000">
      <p class="form-hint">每分钟最大请求数，0 表示不限制</p>
    </div>
    
    <button class="btn btn-secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </button>
  </div>
</template>
