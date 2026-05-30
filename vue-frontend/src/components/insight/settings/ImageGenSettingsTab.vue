<script setup lang="ts">
/**
 * 生图模型设置选项卡组件
 * 用于续写功能的图片生成配置
 */
import { computed, ref } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { providerRequiresApiKey, providerRequiresBaseUrl, getProviderBaseUrl } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import {
  IMAGE_GEN_PROVIDER_OPTIONS,
  PROVIDER_DEFAULT_MODELS,
} from './types'

// ============================================================
// Store
// ============================================================

const insightStore = useInsightStore()

// ============================================================
// 状态
// ============================================================

const provider = ref(insightStore.config.imageGen?.provider || 'gpt2api')
const apiKey = ref(insightStore.config.imageGen?.apiKey || '')
const model = ref(insightStore.config.imageGen?.model || 'gpt-image-2')
const baseUrl = ref(insightStore.config.imageGen?.baseUrl || '')
const transportRetries = ref(insightStore.config.imageGen?.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.imageGen?.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.imageGen?.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => providerRequiresBaseUrl(provider.value))

// ============================================================
// 方法
// ============================================================

function getDefaultModel(providerId: string): string {
  return PROVIDER_DEFAULT_MODELS[providerId]?.imageGen || 'gpt-image-2'
}

function onProviderChange(): void {
  const newProvider = provider.value
  const oldProvider = insightStore.config.imageGen.provider

  if (oldProvider !== newProvider) {
    insightStore.config.imageGen.apiKey = apiKey.value
    insightStore.config.imageGen.model = model.value
    insightStore.config.imageGen.baseUrl = baseUrl.value
    insightStore.config.imageGen.transportRetries = transportRetries.value
    insightStore.config.imageGen.businessRetries = businessRetries.value
    insightStore.config.imageGen.timeoutSeconds = timeoutSeconds.value
  }

  insightStore.setImageGenProvider(newProvider)

  apiKey.value = insightStore.config.imageGen.apiKey
  model.value = insightStore.config.imageGen.model
  baseUrl.value = insightStore.config.imageGen.baseUrl || getProviderBaseUrl(newProvider, 'imageGen')
  transportRetries.value = insightStore.config.imageGen.transportRetries ?? 10
  businessRetries.value = insightStore.config.imageGen.businessRetries ?? 10
  timeoutSeconds.value = insightStore.config.imageGen.timeoutSeconds ?? 0

  if (!model.value) {
    model.value = getDefaultModel(newProvider)
  }
}

/** 获取当前配置 */
function getConfig() {
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

/** 从store同步 */
function syncFromStore(): void {
  const imageGen = insightStore.config.imageGen
  if (imageGen) {
    provider.value = imageGen.provider || 'gpt2api'
    apiKey.value = imageGen.apiKey || ''
    model.value = imageGen.model || 'gpt-image-2'
    baseUrl.value = imageGen.baseUrl || getProviderBaseUrl(provider.value, 'imageGen')
    transportRetries.value = imageGen.transportRetries ?? 10
    businessRetries.value = imageGen.businessRetries ?? 10
    timeoutSeconds.value = imageGen.timeoutSeconds ?? 0
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
    <p class="settings-hint">生图模型服务商保留为可扩展选择器，当前可选项只有 gpt2api，带参考图时会自动适配到其图片编辑路由。</p>
    
    <div class="form-group">
      <label>服务商</label>
      <CustomSelect
        v-model="provider"
        :options="IMAGE_GEN_PROVIDER_OPTIONS"
        @change="onProviderChange"
      />
    </div>
    
    <div v-if="providerRequiresApiKey(provider)" class="form-group">
      <label>API Key</label>
      <input v-model="apiKey" type="password" placeholder="输入 API Key">
    </div>
    
    <div class="form-group">
      <label>模型</label>
      <input v-model="model" type="text" placeholder="例如: gpt-image-2">
      <p class="form-hint">默认推荐使用当前服务商的默认生图模型。</p>
    </div>
    
    <div v-if="showBaseUrl" class="form-group">
      <label>Base URL</label>
      <input v-model="baseUrl" type="text" placeholder="例如: http://127.0.0.1:17200 或 http://127.0.0.1:17200/v1">
    </div>
    
    <div class="form-group">
      <label>传输重试次数</label>
      <input v-model.number="transportRetries" type="number" min="0" max="100">
      <p class="form-hint">网络超时、连接错误、429/5xx 的自动重试次数，默认 10</p>
    </div>

    <div class="form-group">
      <label>业务重试次数</label>
      <input v-model.number="businessRetries" type="number" min="0" max="100">
      <p class="form-hint">当接口返回空图片结果或结果不可解析时的额外重试次数，默认 10</p>
    </div>

    <div class="form-group">
      <label>单次请求超时（秒）</label>
      <input v-model.number="timeoutSeconds" type="number" min="0" max="3600" step="1">
      <p class="form-hint">0 表示不限制；大于 0 时作为单次生图 HTTP 请求超时</p>
    </div>
  </div>
</template>

