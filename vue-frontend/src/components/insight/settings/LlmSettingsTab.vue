<script setup lang="ts">
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref, computed } from 'vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import { providerRequiresApiKey } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { StoreLlmConfig } from '@/types/insight'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import { useInsightModelFetch } from './useInsightModelFetch'
import {
  LLM_PROVIDER_OPTIONS,
  LLM_DEFAULT_MODELS,
} from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
  (e: 'update:config', config: StoreLlmConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const isTesting = ref(false)

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
const {
  isFetchingModels,
  modelOptions,
  modelCount,
  invalidateModelFetch,
  fetchModels,
  selectModel,
} = useInsightModelFetch({
  domain: 'insight_chat',
  provider,
  apiKey,
  baseUrl,
  model,
  emitMessage: (message, type) => emit('showMessage', message, type),
})

function onProviderChange(): void {
  const newProvider = provider.value
  invalidateModelFetch()

  applyDraftConfig(insightStore.switchLlmProviderDraft(buildDraftConfig()))

  if (!model.value) {
    const defaultModel = LLM_DEFAULT_MODELS[newProvider]
    if (defaultModel) {
      model.value = defaultModel
    }
  }
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

function buildDraftConfig(): StoreLlmConfig {
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

function applyDraftConfig(config: StoreLlmConfig): void {
  provider.value = config.provider
  apiKey.value = config.apiKey
  model.value = config.model
  baseUrl.value = config.baseUrl
  forceJsonOutput.value = config.openaiOptions.request.forceJsonOutput
  extraBody.value = config.openaiOptions.request.extraBody
  useStream.value = config.openaiOptions.execution.useStream
  rpmLimit.value = config.openaiOptions.execution.rpmLimit
  transportRetries.value = config.openaiOptions.execution.transportRetries
  businessRetries.value = config.openaiOptions.execution.businessRetries
}

useInsightSettingsDraft<StoreLlmConfig>({
  sources: [provider, apiKey, model, baseUrl, forceJsonOutput, extraBody, useStream, rpmLimit, transportRetries, businessRetries],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.llm,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
  deep: true,
})
</script>

<template>
  <InsightSettingsPanel description="LLM（对话模型）用于生成故事概要、智能问答等文本生成任务。">
    <InsightModelProviderSection
      v-model:provider="provider"
      v-model:api-key="apiKey"
      v-model:model="model"
      v-model:base-url="baseUrl"
      :provider-options="LLM_PROVIDER_OPTIONS"
      :show-api-key="providerRequiresApiKey(provider)"
      credential-id="insight-llm-api-key"
      provider-input-id="insight-llm-provider"
      model-input-id="insight-llm-model"
      base-url-input-id="insight-llm-base-url"
      model-placeholder="例如: gpt-4o-mini"
      fetch-variant="primary"
      :fetching-models="isFetchingModels"
      :model-options="modelOptions"
      :model-count="modelCount"
      :show-base-url="showBaseUrl"
      :testing="isTesting"
      @provider-change="onProviderChange"
      @model-change="selectModel"
      @fetch="fetchModels"
      @test="testConnection"
    />

    <UiFormGrid>
      <UiField variant="settings" label="RPM 限制" control-id="insight-llm-rpm-limit">
        <UiNumberField v-model="rpmLimit" input-id="insight-llm-rpm-limit" :min="0" :max="100" />
      </UiField>
      <UiField variant="settings" label="传输重试" control-id="insight-llm-transport-retries">
        <UiNumberField v-model="transportRetries" input-id="insight-llm-transport-retries" :min="0" :max="10" />
      </UiField>
      <UiField variant="settings" label="业务重试" control-id="insight-llm-business-retries">
        <UiNumberField v-model="businessRetries" input-id="insight-llm-business-retries" :min="0" :max="10" />
      </UiField>
    </UiFormGrid>

    <UiField variant="settings" control="checkbox" hint="对 OpenAI 兼容 API 启用 response_format: json_object">
      <UiCheckbox
        v-model="forceJsonOutput"
        label="强制 JSON 输出"
      />
    </UiField>

    <UiField variant="settings" control="checkbox">
      <UiCheckbox
        v-model="useStream"
        label="使用流式请求"
      />
    </UiField>

    <UiField variant="settings">
      <OpenAIExtraBodyEditor v-model="extraBody" />
    </UiField>
  </InsightSettingsPanel>
</template>
