<script setup lang="ts">
/**
 * 生图模型设置选项卡组件
 * 用于续写功能的图片生成配置
 */
import { ref } from 'vue'
import { useInsightStore } from '@/stores/insightStore'

// ============================================================
// Store
// ============================================================

const insightStore = useInsightStore()
const providerId = 'gpt2api'

// ============================================================
// 状态
// ============================================================

const apiKey = ref(insightStore.config.imageGen?.apiKey || '')
const model = ref(insightStore.config.imageGen?.model || 'gpt-image-2')
const baseUrl = ref(insightStore.config.imageGen?.baseUrl || '')
const maxRetries = ref(insightStore.config.imageGen?.maxRetries || 3)

// ============================================================
// 方法
// ============================================================

/** 获取当前配置 */
function getConfig() {
  return {
    provider: providerId,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: baseUrl.value,
    maxRetries: maxRetries.value
  }
}

/** 从store同步 */
function syncFromStore(): void {
  const imageGen = insightStore.config.imageGen
  if (imageGen) {
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
    <p class="settings-hint">生图模型当前只支持 gpt2api 网关，带参考图时会自动适配到其图片编辑路由。</p>
    
    <div class="form-group">
      <label>服务商</label>
      <input :value="providerId" type="text" readonly>
    </div>
    
    <div class="form-group">
      <label>API Key</label>
      <input v-model="apiKey" type="password" placeholder="输入 API Key">
    </div>
    
    <div class="form-group">
      <label>模型</label>
      <input v-model="model" type="text" placeholder="例如: gpt-image-2">
      <p class="form-hint">默认推荐使用 gpt2api 暴露的 `gpt-image-2`。</p>
    </div>
    
    <div class="form-group">
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

