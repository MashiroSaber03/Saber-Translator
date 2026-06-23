<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

import UiButton from '@/components/ui/UiButton.vue'
/**
 * LLM 设置选项卡组件
 */
import { ref, computed } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
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
let modelFetchRequestId = 0

const provider = ref(insightStore.config.llm.provider)
const apiKey = ref(insightStore.config.llm.apiKey)
const model = ref(insightStore.config.llm.model)
const baseUrl = ref(insightStore.config.llm.baseUrl)
const forceJsonOutput = ref(insightStore.config.llm.openaiOptions.request.forceJsonOutput)
const extraBody = ref(insightStore.config.llm.openaiOptions.request.extraBody)
const useStream = ref(insightStore.config.llm.openaiOptions.execution.useStream)
const rpmLimit = ref(insightStore.config.llm.openaiOptions.execution.rpmLimit)
const transportRetries = ref(insightStore.config.llm.openaiOptions.execution.transportRetries)
const businessRetries = ref(insightStore.config.llm.openaiOptions.execution.businessRetries)

const showBaseUrl = computed(() => provider.value === 'custom')

function resetModelOptions(): void {
  models.value = []
  modelSelectVisible.value = false
}

function onProviderChange(): void {
  const newProvider = provider.value
  const oldProvider = insightStore.config.llm.provider
  modelFetchRequestId += 1
  isFetchingModels.value = false
  resetModelOptions()

  if (oldProvider !== newProvider) {
    insightStore.config.llm.apiKey = apiKey.value
    insightStore.config.llm.model = model.value
    insightStore.config.llm.baseUrl = baseUrl.value
    insightStore.config.llm.openaiOptions.request.forceJsonOutput = forceJsonOutput.value
    insightStore.config.llm.openaiOptions.request.extraBody = extraBody.value
    insightStore.config.llm.openaiOptions.execution.useStream = useStream.value
    insightStore.config.llm.openaiOptions.execution.rpmLimit = rpmLimit.value
    insightStore.config.llm.openaiOptions.execution.transportRetries = transportRetries.value
    insightStore.config.llm.openaiOptions.execution.businessRetries = businessRetries.value
  }

  insightStore.setLlmProvider(newProvider)

  apiKey.value = insightStore.config.llm.apiKey
  model.value = insightStore.config.llm.model
  baseUrl.value = insightStore.config.llm.baseUrl
  forceJsonOutput.value = insightStore.config.llm.openaiOptions.request.forceJsonOutput
  extraBody.value = insightStore.config.llm.openaiOptions.request.extraBody
  useStream.value = insightStore.config.llm.openaiOptions.execution.useStream
  rpmLimit.value = insightStore.config.llm.openaiOptions.execution.rpmLimit
  transportRetries.value = insightStore.config.llm.openaiOptions.execution.transportRetries
  businessRetries.value = insightStore.config.llm.openaiOptions.execution.businessRetries

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
  } catch {
    if (isCurrentRequest()) {
      emit('showMessage', '获取模型列表失败', 'error')
      modelSelectVisible.value = false
    }
  } finally {
    if (modelFetchRequestId === requestId) {
      isFetchingModels.value = false
    }
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
    openaiOptions: {
      request: {
        forceJsonOutput: forceJsonOutput.value,
        temperature: insightStore.config.llm.openaiOptions.request.temperature,
        extraBody: extraBody.value
      },
      execution: {
        useStream: useStream.value,
        rpmLimit: rpmLimit.value,
        transportRetries: transportRetries.value,
        businessRetries: businessRetries.value
      }
    }
  }
}

function syncFromStore(): void {
  provider.value = insightStore.config.llm.provider
  apiKey.value = insightStore.config.llm.apiKey
  model.value = insightStore.config.llm.model
  baseUrl.value = insightStore.config.llm.baseUrl
  forceJsonOutput.value = insightStore.config.llm.openaiOptions.request.forceJsonOutput
  extraBody.value = insightStore.config.llm.openaiOptions.request.extraBody
  useStream.value = insightStore.config.llm.openaiOptions.execution.useStream
  rpmLimit.value = insightStore.config.llm.openaiOptions.execution.rpmLimit
  transportRetries.value = insightStore.config.llm.openaiOptions.execution.transportRetries
  businessRetries.value = insightStore.config.llm.openaiOptions.execution.businessRetries
}

defineExpose({ getConfig, syncFromStore })
</script>

<template>
  <div class="insight-settings-content">
    <p class="settings-hint">LLM（对话模型）用于生成故事概要、智能问答等文本生成任务。</p>

    <div class="insight-settings-field">
      <label>服务商</label>
      <CustomSelect v-model="provider" :options="VLM_PROVIDER_OPTIONS" @change="onProviderChange" />
    </div>

    <div v-if="providerRequiresApiKey(provider)" class="insight-settings-field">
      <label>API Key</label>
      <UiInput v-model="apiKey" type="password" placeholder="输入 API Key" />
    </div>

    <div class="insight-settings-field">
      <label>模型</label>
      <div class="model-input-row">
        <UiInput v-model="model" type="text" placeholder="例如: gpt-4o-mini" />
        <UiButton variant="secondary" class="fetch-btn" :disabled="isFetchingModels" @click="fetchModels" size="sm">
          {{ isFetchingModels ? '获取中...' : '🔍 获取模型' }}
        </UiButton>
      </div>
      <div v-if="modelSelectVisible && models.length > 0" class="model-select-container">
        <UiSelect class="model-select" :model-value="model" @change="onModelSelected">
          <option value="">-- 选择模型 --</option>
          <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name || m.id }}</option>
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
        <UiInput v-model.number="rpmLimit" type="number" min="0" max="100" />
      </div>
      <div class="insight-settings-field">
        <label>传输重试</label>
        <UiInput v-model.number="transportRetries" type="number" min="0" max="10" />
      </div>
      <div class="insight-settings-field">
        <label>业务重试</label>
        <UiInput v-model.number="businessRetries" type="number" min="0" max="10" />
      </div>
    </div>

    <div class="insight-settings-field">
      <label class="ui-checkbox-label">
        <UiInput v-model="forceJsonOutput" type="checkbox" />
        <span>强制 JSON 输出</span>
      </label>
      <p class="form-hint">对 OpenAI 兼容 API 启用 response_format: json_object</p>
    </div>

    <div class="insight-settings-field">
      <label class="ui-checkbox-label">
        <UiInput v-model="useStream" type="checkbox" />
        <span>使用流式请求</span>
      </label>
    </div>

    <div class="insight-settings-field">
      <OpenAIExtraBodyEditor v-model="extraBody" />
    </div>

    <UiButton variant="secondary" :disabled="isTesting" @click="testConnection">
      {{ isTesting ? '测试中...' : '测试连接' }}
    </UiButton>
  </div>
</template>

<style scoped>
.insight-settings-content {
  --llm-settings-tab-border-default: rgba(99, 102, 241, .2);
  --llm-settings-tab-surface-base: rgba(99, 102, 241, .05);
}

.insight-settings-content {
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

.insight-settings-content {
  --ui-button-padding: 10px 16px;
  --ui-button-radius: 6px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default, var(--color-text-default));
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;
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
  background: linear-gradient(135deg, var(--color-focus-brand-soft), var(--llm-settings-tab-surface-base));
  border-radius: 6px;
  border: 1px solid var(--llm-settings-tab-border-default);
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
