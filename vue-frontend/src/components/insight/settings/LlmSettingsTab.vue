<script setup lang="ts">
/**
 * LLM 设置选项卡组件
 */
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { providerRequiresApiKey } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import {
  VLM_PROVIDER_OPTIONS,
  LLM_DEFAULT_MODELS,
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

const provider = ref(insightStore.config.llm.provider)
const apiKey = ref(insightStore.config.llm.apiKey)
const model = ref(insightStore.config.llm.model)
const baseUrl = ref(insightStore.config.llm.baseUrl)
const useStream = ref(insightStore.config.llm.useStream)

const showBaseUrl = computed(() => provider.value === 'custom')

function onProviderChange(): void {
  const newProvider = provider.value
  const oldProvider = insightStore.config.llm.provider
  
  if (oldProvider !== newProvider) {
    insightStore.config.llm.apiKey = apiKey.value
    insightStore.config.llm.model = model.value
    insightStore.config.llm.baseUrl = baseUrl.value
    insightStore.config.llm.useStream = useStream.value
  }
  
  insightStore.setLlmProvider(newProvider)
  
  apiKey.value = insightStore.config.llm.apiKey
  model.value = insightStore.config.llm.model
  baseUrl.value = insightStore.config.llm.baseUrl
  useStream.value = insightStore.config.llm.useStream
  
  if (!model.value) {
    const defaultModel = LLM_DEFAULT_MODELS[newProvider]
    if (defaultModel) {
      model.value = defaultModel
    }
  }
}

async function fetchModels(): Promise<void> {
  if (providerRequiresApiKey(provider.value) && !apiKey.value) {
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
    
    if (response.success && response.models && response.models.length > 0) {
      models.value = response.models
      modelSelectVisible.value = true
      emit('showMessage', `获取到 ${response.models.length} 个模型`, 'success')
    } else {
      emit('showMessage', response.message || '未获取到模型列表', 'error')
      modelSelectVisible.value = false
    }
  } catch (error) {
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
    const response = await insightApi.testLlmConnection({
      provider: provider.value,
      api_key: apiKey.value,
      model: model.value,
      base_url: baseUrl.value || undefined
    })
    
    if (response.success) {
      emit('showMessage', 'LLM 连接成功', 'success')
    } else {
      emit('showMessage', '连接失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    emit('showMessage', '测试失败', 'error')
  } finally {
    isTesting.value = false
  }
}

function getConfig() {
  return {
    useSameAsVlm: false,
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    useStream: useStream.value
  }
}

function syncFromStore(): void {
  provider.value = insightStore.config.llm.provider
  apiKey.value = insightStore.config.llm.apiKey
  model.value = insightStore.config.llm.model
  baseUrl.value = insightStore.config.llm.baseUrl
  useStream.value = insightStore.config.llm.useStream
}

defineExpose({ getConfig, syncFromStore })
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">LLM（对话模型）用于生成故事概要、智能问答等文本生成任务。</p>
    
    <div class="form-group">
      <label>服务商</label>
      <CustomSelect v-model="provider" :options="VLM_PROVIDER_OPTIONS" @change="onProviderChange" />
    </div>
    
    <div v-if="providerRequiresApiKey(provider)" class="form-group">
      <label>API Key</label>
      <input v-model="apiKey" type="password" placeholder="输入 API Key">
    </div>
    
    <div class="form-group">
      <label>模型</label>
      <div class="model-input-row">
        <input v-model="model" type="text" placeholder="例如: gpt-4o-mini">
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
      <label class="checkbox-label">
        <input v-model="useStream" type="checkbox">
        <span>使用流式请求</span>
      </label>
    </div>
    
    <button class="btn btn-secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </button>
  </div>
</template>
