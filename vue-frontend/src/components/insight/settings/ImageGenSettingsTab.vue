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
const maxRetries = ref(insightStore.config.imageGen?.maxRetries || 3)
const previousProvider = ref(provider.value)

const showBaseUrl = computed(() => providerRequiresBaseUrl(provider.value))

// ============================================================
// 方法
// ============================================================

function getDefaultModel(providerId: string): string {
  return PROVIDER_DEFAULT_MODELS[providerId]?.imageGen || 'gpt-image-2'
}

function onProviderChange(): void {
  const currentProvider = provider.value
  const previousProviderId = previousProvider.value
  const defaultModel = getDefaultModel(currentProvider)
  const previousDefaultModel = getDefaultModel(previousProviderId)
  const defaultBaseUrl = getProviderBaseUrl(currentProvider, 'imageGen')
  const previousDefaultBaseUrl = getProviderBaseUrl(previousProviderId, 'imageGen')

  if (!model.value || model.value === previousDefaultModel) {
    model.value = defaultModel
  }

  if (!baseUrl.value || baseUrl.value === previousDefaultBaseUrl) {
    baseUrl.value = defaultBaseUrl
  }

  previousProvider.value = currentProvider
}

/** 获取当前配置 */
function getConfig() {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: showBaseUrl.value ? baseUrl.value : '',
    maxRetries: maxRetries.value
  }
}

/** 从store同步 */
function syncFromStore(): void {
  const imageGen = insightStore.config.imageGen
  if (imageGen) {
    provider.value = imageGen.provider || 'gpt2api'
    previousProvider.value = provider.value
    apiKey.value = imageGen.apiKey || ''
    model.value = imageGen.model || 'gpt-image-2'
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
      <label>失败重试次数</label>
      <input v-model.number="maxRetries" type="number" min="1" max="10">
      <p class="form-hint">每张图片生成失败后的重试次数</p>
    </div>
  </div>
</template>

