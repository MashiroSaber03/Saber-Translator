<script setup lang="ts">
/**
 * VLM 设置选项卡组件
 */
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import {
  VLM_PROVIDER_OPTIONS,
  VLM_DEFAULT_MODELS,
  SUPPORTED_FETCH_PROVIDERS,
  type ModelInfo
} from './types'

// ============================================================
// Props & Emits
// ============================================================

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
}>()

// ============================================================
// Store
// ============================================================

const insightStore = useInsightStore()

// ============================================================
// 状态
// ============================================================

const isTesting = ref(false)
const isFetchingModels = ref(false)
const models = ref<ModelInfo[]>([])
const modelSelectVisible = ref(false)

// VLM 设置（从 store 同步）
const provider = ref(insightStore.config.vlm.provider)
const apiKey = ref(insightStore.config.vlm.apiKey)
const model = ref(insightStore.config.vlm.model)
const baseUrl = ref(insightStore.config.vlm.baseUrl)
const rpmLimit = ref(insightStore.config.vlm.rpmLimit)
const temperature = ref(insightStore.config.vlm.temperature)
const forceJson = ref(insightStore.config.vlm.forceJson)
const useStream = ref(insightStore.config.vlm.useStream)
const imageMaxSize = ref(insightStore.config.vlm.imageMaxSize)

// ============================================================
// 计算属性
// ============================================================

const showBaseUrl = computed(() => provider.value === 'custom')

// ============================================================
// 方法
// ============================================================

function onProviderChange(): void {
  const newProvider = provider.value
  const oldProvider = insightStore.config.vlm.provider
  
  if (oldProvider !== newProvider) {
    insightStore.config.vlm.apiKey = apiKey.value
    insightStore.config.vlm.model = model.value
    insightStore.config.vlm.baseUrl = baseUrl.value
    insightStore.config.vlm.rpmLimit = rpmLimit.value
    insightStore.config.vlm.temperature = temperature.value
    insightStore.config.vlm.forceJson = forceJson.value
    insightStore.config.vlm.useStream = useStream.value
    insightStore.config.vlm.imageMaxSize = imageMaxSize.value
  }
  
  insightStore.setVlmProvider(newProvider)
  
  apiKey.value = insightStore.config.vlm.apiKey
  model.value = insightStore.config.vlm.model
  baseUrl.value = insightStore.config.vlm.baseUrl
  rpmLimit.value = insightStore.config.vlm.rpmLimit
  temperature.value = insightStore.config.vlm.temperature
  forceJson.value = insightStore.config.vlm.forceJson
  useStream.value = insightStore.config.vlm.useStream
  imageMaxSize.value = insightStore.config.vlm.imageMaxSize
  
  if (!model.value) {
    const defaultModel = VLM_DEFAULT_MODELS[newProvider]
    if (defaultModel) {
      model.value = defaultModel
    }
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
    
    if (response.success && response.models && response.models.length > 0) {
      models.value = response.models
      modelSelectVisible.value = true
      emit('showMessage', `获取到 ${response.models.length} 个模型`, 'success')
    } else {
      emit('showMessage', response.message || '未获取到模型列表', 'error')
      modelSelectVisible.value = false
    }
  } catch (error) {
    emit('showMessage', '获取模型列表失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    modelSelectVisible.value = false
  } finally {
    isFetchingModels.value = false
  }
}

function onModelSelected(modelId: string): void {
  if (modelId) {
    model.value = modelId
  }
}

async function testConnection(): Promise<void> {
  if (isTesting.value) return
  
  isTesting.value = true
  
  try {
    const response = await insightApi.testVlmConnection({
      provider: provider.value,
      api_key: apiKey.value,
      model: model.value,
      base_url: baseUrl.value || undefined
    })
    
    if (response.success) {
      emit('showMessage', 'VLM 连接成功', 'success')
    } else {
      emit('showMessage', '连接失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    emit('showMessage', '测试失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isTesting.value = false
  }
}

/** 获取当前配置 */
function getConfig() {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    rpmLimit: rpmLimit.value,
    temperature: temperature.value,
    forceJson: forceJson.value,
    useStream: useStream.value,
    imageMaxSize: imageMaxSize.value
  }
}

/** 从store同步 */
function syncFromStore(): void {
  provider.value = insightStore.config.vlm.provider
  apiKey.value = insightStore.config.vlm.apiKey
  model.value = insightStore.config.vlm.model
  baseUrl.value = insightStore.config.vlm.baseUrl
  rpmLimit.value = insightStore.config.vlm.rpmLimit
  temperature.value = insightStore.config.vlm.temperature
  forceJson.value = insightStore.config.vlm.forceJson
  useStream.value = insightStore.config.vlm.useStream
  imageMaxSize.value = insightStore.config.vlm.imageMaxSize
}

// 暴露方法给父组件
defineExpose({
  getConfig,
  syncFromStore
})
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">VLM（视觉语言模型）用于分析漫画图片内容，提取对话和场景信息。</p>
    
    <div class="form-group">
      <label>服务商</label>
      <CustomSelect
        v-model="provider"
        :options="VLM_PROVIDER_OPTIONS"
        @change="onProviderChange"
      />
    </div>
    
    <div class="form-group">
      <label>API Key</label>
      <input v-model="apiKey" type="password" placeholder="输入 API Key">
    </div>
    
    <div class="form-group">
      <label>模型</label>
      <div class="model-input-row">
        <input v-model="model" type="text" placeholder="例如: gemini-2.0-flash">
        <button 
          class="btn btn-secondary btn-sm fetch-btn" 
          :disabled="isFetchingModels"
          @click="fetchModels"
        >
          {{ isFetchingModels ? '获取中...' : '🔍 获取模型' }}
        </button>
      </div>
      <div v-if="modelSelectVisible && models.length > 0" class="model-select-container">
        <select 
          class="model-select"
          :value="model"
          @change="onModelSelected(($event.target as HTMLSelectElement).value)"
        >
          <option value="">-- 选择模型 --</option>
          <option v-for="m in models" :key="m.id" :value="m.id">
            {{ m.name || m.id }}
          </option>
        </select>
        <span class="model-count">共 {{ models.length }} 个模型</span>
      </div>
    </div>
    
    <div v-if="showBaseUrl" class="form-group">
      <label>Base URL</label>
      <input v-model="baseUrl" type="text" placeholder="自定义 API 地址">
    </div>
    
    <div class="form-row">
      <div class="form-group">
        <label>RPM 限制</label>
        <input v-model.number="rpmLimit" type="number" min="1" max="100">
        <p class="form-hint">每分钟最大请求数</p>
      </div>
      <div class="form-group">
        <label>温度</label>
        <input v-model.number="temperature" type="number" min="0" max="1" step="0.1">
        <p class="form-hint">0-1，越低越确定</p>
      </div>
    </div>
    
    <div class="form-group">
      <label class="checkbox-label">
        <input v-model="forceJson" type="checkbox">
        <span>强制 JSON 输出</span>
      </label>
      <p class="form-hint">对 OpenAI 兼容 API 启用 response_format: json_object</p>
    </div>
    
    <div class="form-group">
      <label class="checkbox-label">
        <input v-model="useStream" type="checkbox">
        <span>使用流式请求</span>
      </label>
      <p class="form-hint">流式请求可避免长时间等待导致的超时问题</p>
    </div>
    
    <div class="form-group">
      <label>图片压缩（最大边长）</label>
      <input v-model.number="imageMaxSize" type="number" min="0" max="4096" step="128" placeholder="0 表示不压缩">
      <p class="form-hint">发送前将图片等比例缩放到指定最大边长（像素），0 表示不压缩</p>
    </div>
    
    <button class="btn btn-secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </button>
  </div>
</template>
