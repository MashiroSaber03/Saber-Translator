<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

/**
 * 生图模型设置选项卡组件
 * 用于续写功能的图片生成配置
 */
import { computed, ref } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { providerRequiresApiKey, providerRequiresBaseUrl, providerRequiresModel, getProviderBaseUrl } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import {
  IMAGE_GEN_PROVIDER_OPTIONS,
  PROVIDER_DEFAULT_MODELS,
} from './types'

// ============================================================
// Store
// ============================================================

const insightStore = useInsightStore()
const initialProvider = insightStore.config.imageGen?.provider || 'gpt2api'

// ============================================================
// 状态
// ============================================================

const provider = ref(initialProvider)
const apiKey = ref(insightStore.config.imageGen?.apiKey || '')
const model = ref(insightStore.config.imageGen?.model ?? (PROVIDER_DEFAULT_MODELS[initialProvider]?.imageGen || 'gpt-image-2'))
const baseUrl = ref(insightStore.config.imageGen?.baseUrl || '')
const transportRetries = ref(insightStore.config.imageGen?.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.imageGen?.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.imageGen?.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => providerRequiresBaseUrl(provider.value))
const showModelWarning = computed(() => providerRequiresModel(provider.value) && !model.value.trim())

// ============================================================
// 方法
// ============================================================

function getDefaultModel(providerId: string): string {
  return PROVIDER_DEFAULT_MODELS[providerId]?.imageGen || ''
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
    model.value = imageGen.model ?? getDefaultModel(provider.value)
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
    <p class="settings-hint">生图模型服务商保留为可扩展选择器，当前支持 gpt2api 与 New API，带参考图时会自动适配到其图片编辑路由。</p>
    
    <div class="insight-settings-field">
      <label>服务商</label>
      <CustomSelect
        v-model="provider"
        :options="IMAGE_GEN_PROVIDER_OPTIONS"
        @change="onProviderChange"
      />
    </div>
    
    <div v-if="providerRequiresApiKey(provider)" class="insight-settings-field">
      <label>API Key</label>
      <UiInput v-model="apiKey" type="password" placeholder="输入 API Key" />
    </div>
    
    <div class="insight-settings-field">
      <label>模型</label>
      <UiInput v-model="model" type="text" placeholder="例如: gpt-image-2" />
      <p class="form-hint">默认推荐使用当前服务商的默认生图模型。</p>
      <p v-if="showModelWarning" class="form-hint warning-text">当前服务商需要手动填写模型名。</p>
    </div>
    
    <div v-if="showBaseUrl" class="insight-settings-field">
      <label>Base URL</label>
      <UiInput v-model="baseUrl" type="text" placeholder="例如: http://127.0.0.1:17200 或 http://127.0.0.1:17200/v1" />
    </div>
    
    <div class="insight-settings-field">
      <label>传输重试次数</label>
      <UiInput v-model.number="transportRetries" type="number" min="0" max="100" />
      <p class="form-hint">网络超时、连接错误、429/5xx 的自动重试次数，默认 10</p>
    </div>

    <div class="insight-settings-field">
      <label>业务重试次数</label>
      <UiInput v-model.number="businessRetries" type="number" min="0" max="100" />
      <p class="form-hint">当接口返回空图片结果或结果不可解析时的额外重试次数，默认 10</p>
    </div>

    <div class="insight-settings-field">
      <label>单次请求超时（秒）</label>
      <UiInput v-model.number="timeoutSeconds" type="number" min="0" max="3600" step="1" />
      <p class="form-hint">0 表示不限制；大于 0 时作为单次生图 HTTP 请求超时</p>
    </div>
  </div>
</template>

<style scoped>.insight-settings-content {
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
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .insight-settings-field input[type="text"],
.insight-settings-content .insight-settings-field input[type="password"],
.insight-settings-content .insight-settings-field input[type="number"],
.insight-settings-content .insight-settings-field select,
.insight-settings-content .insight-settings-field textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 6px;
  font-size: 14px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default, var(--color-text-default));
  transition: border-color 0.2s, box-shadow 0.2s;
}

.insight-settings-content .insight-settings-field input:focus,
.insight-settings-content .insight-settings-field select:focus,
.insight-settings-content .insight-settings-field textarea:focus {
  outline: none;
  border-color: var(--color-border-brand);
  box-shadow: 0 0 0 3px var(--color-focus-brand-soft);
}

.insight-settings-content .form-hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: normal;
}

.insight-settings-content .ui-checkbox-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.insight-settings-content .ui-button {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.insight-settings-content .ui-button--primary {
  background: var(--color-surface-brand);
  color: white;
}

.insight-settings-content .ui-button--primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.insight-settings-content .ui-button--primary:hover:not(:disabled) {
  background: var(--color-surface-brand-strong);
}

.insight-settings-content .ui-button--secondary {
  background: var(--color-surface-muted);
  color: var(--color-text-default, var(--color-text-default));
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .ui-button--secondary:hover:not(:disabled) {
  background: var(--color-surface-hover);
}

.insight-settings-content .form-row {
  display: flex;
  gap: 16px;
}

.insight-settings-content .form-row .insight-settings-field {
  flex: 1;
}

.insight-settings-content .placeholder-text {
  color: var(--color-text-supporting, var(--color-text-secondary));
  text-align: center;
  padding: 40px;
}

.insight-settings-content .prompts-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.insight-settings-content .prompt-editor {
  width: 100%;
  min-height: 200px;
  font-family: Consolas, Monaco, monospace;
  font-size: 13px;
  line-height: 1.5;
  padding: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  background: var(--color-surface-muted);
  color: var(--color-text-default, var(--color-text-default));
  resize: vertical;
}

.insight-settings-content .prompt-editor:focus {
  outline: none;
  border-color: var(--color-border-brand);
}

.insight-settings-content .prompt-actions-bar {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.insight-settings-content .ui-button--sm {
  padding: 6px 12px;
  font-size: 13px;
}

.insight-settings-content .section-divider {
  border: none;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  margin: 16px 0;
}

.insight-settings-content .prompts-library-section {
  margin-top: 8px;
}

.insight-settings-content .library-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.insight-settings-content .library-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.insight-settings-content .library-actions {
  display: flex;
  gap: 8px;
}

.insight-settings-content .saved-prompts-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 4px;
  background: var(--color-surface-muted);
}

.insight-settings-content .saved-prompt-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
  transition: background 0.2s;
}

.insight-settings-content .saved-prompt-item:last-child {
  border-bottom: none;
}

.insight-settings-content .saved-prompt-item:hover {
  background: var(--color-surface-hover);
}

.insight-settings-content .prompt-name {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.insight-settings-content .prompt-type-badge {
  font-size: 11px;
  padding: 2px 6px;
  background: var(--color-focus-brand-soft);
  color: var(--color-text-brand);
  border-radius: 4px;
  white-space: nowrap;
}

.insight-settings-content .button-icon-sm {
  padding: 2px 6px;
  background: none;
  border: none;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.insight-settings-content .button-icon-sm:hover {
  opacity: 1;
}

.insight-settings-content .loading-text {
  text-align: center;
  padding: 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.insight-settings-content .batch-info-box {
  margin-top: 16px;
  padding: 12px;
  background: var(--color-surface-subtle);
  border-radius: 8px;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
}

.insight-settings-content .batch-info-box h4 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .layers-preview-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}

.insight-settings-content .layers-preview-list li {
  margin-bottom: 4px;
}

.insight-settings-content .align-badge {
  color: var(--color-text-brand);
  font-size: 12px;
}

.insight-settings-content .batch-estimate-box {
  margin-top: 12px;
  padding: 10px 12px;
  background: linear-gradient(135deg, var(--color-focus-brand-soft), var(--image-gen-settings-tab-surface-base));
  border-radius: 6px;
  border: 1px solid var(--image-gen-settings-tab-border-default);
}

.insight-settings-content .batch-estimate-box p {
  margin: 0;
  font-size: 13px;
  color: var(--color-text-default, var(--color-text-default));
}

.insight-settings-content .batch-estimate-box strong {
  color: var(--color-text-brand);
}

.insight-settings-content .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-content .model-input-row input {
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
  color: var(--color-text-default, var(--color-text-default));
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
