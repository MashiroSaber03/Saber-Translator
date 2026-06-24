<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

import UiButton from '@/components/ui/UiButton.vue'
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import { providerRequiresApiKey } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import {
  VLM_PROVIDER_OPTIONS,
  VLM_DEFAULT_MODELS,
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
let modelFetchRequestId = 0

const provider = ref(insightStore.config.vlm.provider)
const apiKey = ref(insightStore.config.vlm.apiKey)
const model = ref(insightStore.config.vlm.model)
const baseUrl = ref(insightStore.config.vlm.baseUrl)
const rpmLimit = ref(insightStore.config.vlm.openaiOptions.execution.rpmLimit)
const transportRetries = ref(insightStore.config.vlm.openaiOptions.execution.transportRetries)
const businessRetries = ref(insightStore.config.vlm.openaiOptions.execution.businessRetries)
const temperature = ref(insightStore.config.vlm.openaiOptions.request.temperature)
const forceJsonOutput = ref(insightStore.config.vlm.openaiOptions.request.forceJsonOutput)
const extraBody = ref(insightStore.config.vlm.openaiOptions.request.extraBody)
const useStream = ref(insightStore.config.vlm.openaiOptions.execution.useStream)
const imageMaxSize = ref(insightStore.config.vlm.imageMaxSize)

const showBaseUrl = computed(() => provider.value === 'custom')

function resetModelOptions(): void {
  models.value = []
  modelSelectVisible.value = false
}

function onProviderChange(): void {
  const newProvider = provider.value
  const previousProvider = insightStore.config.vlm.provider
  modelFetchRequestId += 1
  isFetchingModels.value = false
  resetModelOptions()

  if (previousProvider !== newProvider) {
    insightStore.config.vlm.apiKey = apiKey.value
    insightStore.config.vlm.model = model.value
    insightStore.config.vlm.baseUrl = baseUrl.value
    insightStore.config.vlm.openaiOptions.execution.rpmLimit = rpmLimit.value
    insightStore.config.vlm.openaiOptions.execution.transportRetries = transportRetries.value
    insightStore.config.vlm.openaiOptions.execution.businessRetries = businessRetries.value
    insightStore.config.vlm.openaiOptions.request.temperature = temperature.value
    insightStore.config.vlm.openaiOptions.request.forceJsonOutput = forceJsonOutput.value
    insightStore.config.vlm.openaiOptions.request.extraBody = extraBody.value
    insightStore.config.vlm.openaiOptions.execution.useStream = useStream.value
    insightStore.config.vlm.imageMaxSize = imageMaxSize.value
  }

  insightStore.setVlmProvider(newProvider)

  apiKey.value = insightStore.config.vlm.apiKey
  model.value = insightStore.config.vlm.model
  baseUrl.value = insightStore.config.vlm.baseUrl
  rpmLimit.value = insightStore.config.vlm.openaiOptions.execution.rpmLimit
  transportRetries.value = insightStore.config.vlm.openaiOptions.execution.transportRetries
  businessRetries.value = insightStore.config.vlm.openaiOptions.execution.businessRetries
  temperature.value = insightStore.config.vlm.openaiOptions.request.temperature
  forceJsonOutput.value = insightStore.config.vlm.openaiOptions.request.forceJsonOutput
  extraBody.value = insightStore.config.vlm.openaiOptions.request.extraBody
  useStream.value = insightStore.config.vlm.openaiOptions.execution.useStream
  imageMaxSize.value = insightStore.config.vlm.imageMaxSize

  if (!model.value) {
    const defaultModel = VLM_DEFAULT_MODELS[newProvider]
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

  isFetchingModels.value = true
  const requestId = ++modelFetchRequestId
  const requestProvider = provider.value
  const requestApiKey = apiKey.value
  const requestBaseUrl = baseUrl.value || undefined
  const isCurrentRequest = () => (
    modelFetchRequestId === requestId &&
    provider.value === requestProvider &&
    apiKey.value === requestApiKey &&
    (baseUrl.value || undefined) === requestBaseUrl
  )

  try {
    const response = await insightApi.fetchModels(requestProvider, requestApiKey, requestBaseUrl)
    if (!isCurrentRequest()) return

    if (response.success && response.models && response.models.length > 0) {
      models.value = response.models
      modelSelectVisible.value = true
      emit('showMessage', `获取到 ${response.models.length} 个模型`, 'success')
    } else {
      emit('showMessage', response.message || '未获取到模型列表', 'error')
      modelSelectVisible.value = false
    }
  } catch (error) {
    if (isCurrentRequest()) {
      emit('showMessage', '获取模型列表失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
      modelSelectVisible.value = false
    }
  } finally {
    if (modelFetchRequestId === requestId) {
      isFetchingModels.value = false
    }
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

function getConfig() {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    openaiOptions: {
      request: {
        forceJsonOutput: forceJsonOutput.value,
        temperature: temperature.value,
        extraBody: extraBody.value
      },
      execution: {
        useStream: useStream.value,
        rpmLimit: rpmLimit.value,
        transportRetries: transportRetries.value,
        businessRetries: businessRetries.value
      }
    },
    imageMaxSize: imageMaxSize.value
  }
}

function syncFromStore(): void {
  provider.value = insightStore.config.vlm.provider
  apiKey.value = insightStore.config.vlm.apiKey
  model.value = insightStore.config.vlm.model
  baseUrl.value = insightStore.config.vlm.baseUrl
  rpmLimit.value = insightStore.config.vlm.openaiOptions.execution.rpmLimit
  transportRetries.value = insightStore.config.vlm.openaiOptions.execution.transportRetries
  businessRetries.value = insightStore.config.vlm.openaiOptions.execution.businessRetries
  temperature.value = insightStore.config.vlm.openaiOptions.request.temperature
  forceJsonOutput.value = insightStore.config.vlm.openaiOptions.request.forceJsonOutput
  extraBody.value = insightStore.config.vlm.openaiOptions.request.extraBody
  useStream.value = insightStore.config.vlm.openaiOptions.execution.useStream
  imageMaxSize.value = insightStore.config.vlm.imageMaxSize
}

defineExpose({
  getConfig,
  syncFromStore
})
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">VLM（视觉语言模型）用于分析漫画图片内容，提取对话和场景信息。</p>

    <div class="insight-settings-field">
      <label>服务商</label>
      <CustomSelect
        v-model="provider"
        :options="VLM_PROVIDER_OPTIONS"
        @change="onProviderChange"
      />
    </div>

    <div v-if="providerRequiresApiKey(provider)" class="insight-settings-field">
      <label>API Key</label>
      <UiInput v-model="apiKey" type="password" placeholder="输入 API Key" />
    </div>

    <div class="insight-settings-field">
      <label>模型</label>
      <div class="model-input-row">
        <UiInput v-model="model" type="text" placeholder="例如: gemini-2.0-flash" class="model-field-input" />
        <UiButton
          variant="secondary"
          class="fetch-btn"
          :disabled="isFetchingModels"
          @click="fetchModels" size="sm"
        >
          {{ isFetchingModels ? '获取中...' : '🔍 获取模型' }}
        </UiButton>
      </div>
      <div v-if="modelSelectVisible && models.length > 0" class="model-select-container">
        <UiSelect
          class="model-select"
          :model-value="model"
          @change="onModelSelected"
        >
          <option value="">-- 选择模型 --</option>
          <option v-for="m in models" :key="m.id" :value="m.id">
            {{ m.name || m.id }}
          </option>
        </UiSelect>
        <span class="model-count">共 {{ models.length }} 个模型</span>
      </div>
    </div>

    <div v-if="showBaseUrl" class="insight-settings-field">
      <label>Base URL</label>
      <UiInput v-model="baseUrl" type="text" placeholder="自定义 API 地址" />
    </div>

    <div class="form-row">
      <div class="insight-settings-field">
        <label>RPM 限制</label>
        <UiInput v-model.number="rpmLimit" type="number" min="1" max="100" />
        <p class="form-hint">每分钟最大请求数</p>
      </div>
      <div class="insight-settings-field">
        <label>传输重试</label>
        <UiInput v-model.number="transportRetries" type="number" min="0" max="10" />
        <p class="form-hint">网络超时/429/5xx</p>
      </div>
      <div class="insight-settings-field">
        <label>业务重试</label>
        <UiInput v-model.number="businessRetries" type="number" min="0" max="10" />
        <p class="form-hint">空结果/结构解析失败</p>
      </div>
      <div class="insight-settings-field">
        <label>温度</label>
        <UiInput v-model.number="temperature" type="number" min="0" max="1" step="0.1" />
        <p class="form-hint">0-1，越低越确定</p>
      </div>
    </div>

    <div class="insight-settings-field">
      <label class="ui-checkbox-label">
        <UiInput v-model="forceJsonOutput" type="checkbox" class="vlm-settings-tab__checkbox-input" />
        <span>强制 JSON 输出</span>
      </label>
      <p class="form-hint">对 OpenAI 兼容 API 启用 response_format: json_object</p>
    </div>

    <div class="insight-settings-field">
      <label class="ui-checkbox-label">
        <UiInput v-model="useStream" type="checkbox" class="vlm-settings-tab__checkbox-input" />
        <span>使用流式请求</span>
      </label>
      <p class="form-hint">流式请求可避免长时间等待导致的超时问题</p>
    </div>

    <div class="insight-settings-field">
      <OpenAIExtraBodyEditor v-model="extraBody" />
    </div>

    <div class="insight-settings-field">
      <label>图片压缩（最大边长）</label>
      <UiInput v-model.number="imageMaxSize" type="number" min="0" max="4096" step="128" placeholder="0 表示不压缩" />
      <p class="form-hint">发送前将图片等比例缩放到指定最大边长（像素），0 表示不压缩</p>
    </div>

    <UiButton variant="secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </UiButton>
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
  --ui-select-padding: 8px 12px;
  --ui-select-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-select-radius: 4px;
  --ui-select-font-size: 13px;
  --ui-select-background: var(--color-surface-input, var(--color-surface-base));
  --ui-select-color: var(--color-text-default);
  --ui-select-focus-shadow: var(--color-focus-brand-soft);
  --ui-button-padding: 10px 16px;
  --ui-button-radius: 6px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;

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

.insight-settings-content .ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: normal;
}

.vlm-settings-tab__checkbox-input {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.insight-settings-content .form-row {
  display: flex;
  gap: 16px;
}

.insight-settings-content .form-row .insight-settings-field {
  flex: 1;
}

.insight-settings-content .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-content .model-field-input {
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
  color: var(--color-text-default);
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
