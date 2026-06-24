<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import { computed, ref } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { providerRequiresApiKey, providerRequiresBaseUrl, providerRequiresModel, getProviderBaseUrl } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import {
  IMAGE_GEN_PROVIDER_OPTIONS,
  PROVIDER_DEFAULT_MODELS,
} from './types'

const insightStore = useInsightStore()
const initialProvider = insightStore.config.imageGen?.provider || 'gpt2api'

const provider = ref(initialProvider)
const apiKey = ref(insightStore.config.imageGen?.apiKey || '')
const model = ref(insightStore.config.imageGen?.model ?? (PROVIDER_DEFAULT_MODELS[initialProvider]?.imageGen || 'gpt-image-2'))
const baseUrl = ref(insightStore.config.imageGen?.baseUrl || '')
const transportRetries = ref(insightStore.config.imageGen?.transportRetries ?? 10)
const businessRetries = ref(insightStore.config.imageGen?.businessRetries ?? 10)
const timeoutSeconds = ref(insightStore.config.imageGen?.timeoutSeconds ?? 0)

const showBaseUrl = computed(() => providerRequiresBaseUrl(provider.value))
const showModelWarning = computed(() => providerRequiresModel(provider.value) && !model.value.trim())

function getDefaultModel(providerId: string): string {
  return PROVIDER_DEFAULT_MODELS[providerId]?.imageGen || ''
}

function onProviderChange(): void {
  const newProvider = provider.value
  const previousProvider = insightStore.config.imageGen.provider

  if (previousProvider !== newProvider) {
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

</style>
