<script setup lang="ts">
/**
 * 生图模型设置选项卡组件
 * 用于续写功能的图片生成配置
 */
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import {
  IMAGE_GEN_PROVIDER_OPTIONS,
  IMAGE_GEN_DEFAULT_MODELS,
  IMAGE_GEN_DEFAULT_BASE_URLS,
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

const isFetchingModels = ref(false)
const models = ref<ModelInfo[]>([])
const modelSelectVisible = ref(false)

// 生图模型设置（从 store 同步）
const provider = ref(insightStore.config.imageGen?.provider || 'siliconflow')
const apiKey = ref(insightStore.config.imageGen?.apiKey || '')
const model = ref(insightStore.config.imageGen?.model || 'stabilityai/stable-diffusion-3-5-large')
const baseUrl = ref(insightStore.config.imageGen?.baseUrl || '')
const maxRetries = ref(insightStore.config.imageGen?.maxRetries || 3)

// ============================================================
// 计算属性
// ============================================================

const showBaseUrl = computed(() => provider.value === 'custom')

// ============================================================
// 方法
// ============================================================

function onProviderChange(): void {
  const newProvider = provider.value
  
  // 更新默认模型
  const defaultModel = IMAGE_GEN_DEFAULT_MODELS[newProvider]
  if (defaultModel) {
    model.value = defaultModel
  }
  
  // 更新默认 Base URL
  if (newProvider !== 'custom') {
    baseUrl.value = IMAGE_GEN_DEFAULT_BASE_URLS[newProvider] || ''
  }
  
  // 清空模型列表
  models.value = []
  modelSelectVisible.value = false
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

/** 获取当前配置 */
function getConfig() {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    maxRetries: maxRetries.value
  }
}

/** 从store同步 */
function syncFromStore(): void {
  const imageGen = insightStore.config.imageGen
  if (imageGen) {
    provider.value = imageGen.provider || 'siliconflow'
    apiKey.value = imageGen.apiKey || ''
    model.value = imageGen.model || 'stabilityai/stable-diffusion-3-5-large'
    baseUrl.value = imageGen.baseUrl || ''
    maxRetries.value = imageGen.maxRetries || 3
  }
}

// 暴露方法给父组件
defineExpose({
  getConfig,
  syncFromStore
})
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">生图模型用于续写功能中生成漫画页面图片。</p>
    
    <div class="form-group">
      <label>服务商</label>
      <CustomSelect
        v-model="provider"
        :options="IMAGE_GEN_PROVIDER_OPTIONS"
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
        <input v-model="model" type="text" placeholder="例如: dall-e-3">
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
      <p class="form-hint">不同服务商支持的模型不同，请参考各服务商文档</p>
    </div>
    
    <div v-if="showBaseUrl" class="form-group">
      <label>Base URL</label>
      <input v-model="baseUrl" type="text" placeholder="自定义 API 地址">
    </div>
    
    <div class="form-group">
      <label>失败重试次数</label>
      <input v-model.number="maxRetries" type="number" min="1" max="10">
      <p class="form-hint">每张图片生成失败后的重试次数</p>
    </div>
    
    <div class="info-box">
      <h4>💡 服务商说明</h4>
      <ul>
        <li><strong>OpenAI DALL-E</strong>: 高质量生成，但价格较高</li>
        <li><strong>SiliconFlow</strong>: 支持 SD3.5 等开源模型，性价比高</li>
        <li><strong>通义万相</strong>: 阿里云生图服务，需申请白名单</li>
        <li><strong>火山引擎</strong>: 字节跳动生图服务</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.info-box {
  margin-top: 20px;
  padding: 16px;
  background: var(--bg-tertiary, #f5f5f5);
  border-radius: 8px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.info-box h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
}

.info-box ul {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.8;
}

.info-box li strong {
  color: var(--primary-color, #007bff);
}
</style>

