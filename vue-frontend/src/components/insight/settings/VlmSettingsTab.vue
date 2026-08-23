<script setup lang="ts">
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref, computed } from 'vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import { getProviderDefaultModel, providerRequiresApiKey } from '@/config/aiProviders'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { StoreVlmConfig } from '@/types/insight'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { useInsightSettingsDraft } from './useInsightSettingsDraft'
import { useInsightModelFetch } from './useInsightModelFetch'
import { useInsightConnectionTest } from './useInsightConnectionTest'
import { VLM_PROVIDER_OPTIONS } from './types'

const emit = defineEmits<{
  (e: 'showMessage', message: string, type: 'success' | 'error'): void
  (e: 'update:config', config: StoreVlmConfig): void
}>()

const props = defineProps<{
  syncRequestId?: number
}>()

const insightStore = useInsightStore()

const provider = ref(insightStore.config.vlm.provider)
const apiKey = ref(insightStore.config.vlm.apiKey)
const model = ref(insightStore.config.vlm.model)
const baseUrl = ref(insightStore.config.vlm.baseUrl ?? '')
const rpmLimit = ref(insightStore.config.vlm.openaiOptions.execution.rpmLimit ?? 0)
const transportRetries = ref(insightStore.config.vlm.openaiOptions.execution.transportRetries)
const businessRetries = ref(insightStore.config.vlm.openaiOptions.execution.businessRetries)
const temperature = ref(insightStore.config.vlm.openaiOptions.request.temperature ?? 0.3)
const forceJsonOutput = ref(insightStore.config.vlm.openaiOptions.request.forceJsonOutput)
const extraBody = ref(insightStore.config.vlm.openaiOptions.request.extraBody)
const useStream = ref(insightStore.config.vlm.openaiOptions.execution.useStream)
const imageMaxSize = ref(insightStore.config.vlm.imageMaxSize ?? 0)

const showBaseUrl = computed(() => provider.value === 'custom')
const {
  isFetchingModels,
  modelOptions,
  modelCount,
  invalidateModelFetch,
  fetchModels,
  selectModel,
} = useInsightModelFetch({
  domain: 'insight_vlm',
  provider,
  apiKey,
  baseUrl,
  model,
  emitMessage: (message, type) => emit('showMessage', message, type),
})

const { isTesting, testConnection } = useInsightConnectionTest({
  sources: [provider, apiKey, model, baseUrl],
  snapshot: () => ({
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
  }),
  request: snapshot => insightApi.testVlmConnection({
    provider: snapshot.provider,
    api_key: snapshot.apiKey,
    model: snapshot.model,
    base_url: snapshot.baseUrl,
  }),
  successMessage: 'VLM 连接成功',
  emitMessage: (message, type) => emit('showMessage', message, type),
})

function onProviderChange(): void {
  const newProvider = provider.value
  invalidateModelFetch()

  applyDraftConfig(insightStore.switchVlmProviderDraft(buildDraftConfig()))

  if (!model.value) {
    const defaultModel = getProviderDefaultModel(newProvider, 'vlm')
    if (defaultModel) {
      model.value = defaultModel
    }
  }
}

function buildDraftConfig(): StoreVlmConfig {
  return {
    provider: provider.value,
    apiKey: apiKey.value,
    model: model.value,
    baseUrl: provider.value === 'custom' ? baseUrl.value : '',
    openaiOptions: {
      request: {
        forceJsonOutput: forceJsonOutput.value,
        temperature: temperature.value,
        extraBody: extraBody.value,
      },
      execution: {
        useStream: useStream.value,
        rpmLimit: rpmLimit.value,
        transportRetries: transportRetries.value,
        businessRetries: businessRetries.value,
      },
    },
    imageMaxSize: imageMaxSize.value,
  }
}

function applyDraftConfig(config: StoreVlmConfig): void {
  provider.value = config.provider
  apiKey.value = config.apiKey
  model.value = config.model
  baseUrl.value = config.baseUrl ?? ''
  rpmLimit.value = config.openaiOptions.execution.rpmLimit ?? 0
  transportRetries.value = config.openaiOptions.execution.transportRetries
  businessRetries.value = config.openaiOptions.execution.businessRetries
  temperature.value = config.openaiOptions.request.temperature ?? 0.3
  forceJsonOutput.value = config.openaiOptions.request.forceJsonOutput
  extraBody.value = config.openaiOptions.request.extraBody
  useStream.value = config.openaiOptions.execution.useStream
  imageMaxSize.value = config.imageMaxSize ?? 0
}

useInsightSettingsDraft<StoreVlmConfig>({
  sources: [
    provider,
    apiKey,
    model,
    baseUrl,
    rpmLimit,
    transportRetries,
    businessRetries,
    temperature,
    forceJsonOutput,
    extraBody,
    useStream,
    imageMaxSize,
  ],
  buildDraft: buildDraftConfig,
  applyDraft: applyDraftConfig,
  loadDraft: () => insightStore.config.vlm,
  emitDraft: config => emit('update:config', config),
  syncRequestId: () => props.syncRequestId,
  deep: true,
})
</script>

<template>
  <InsightSettingsPanel
    class="vlm-settings-tab"
    description="VLM（视觉语言模型）用于分析漫画图片内容，提取对话和场景信息。"
  >
    <InsightModelProviderSection
      v-model:provider="provider"
      v-model:api-key="apiKey"
      v-model:model="model"
      v-model:base-url="baseUrl"
      :provider-options="VLM_PROVIDER_OPTIONS"
      :show-api-key="providerRequiresApiKey(provider)"
      credential-id="insight-vlm-api-key"
      provider-input-id="insight-vlm-provider"
      model-input-id="insight-vlm-model"
      base-url-input-id="insight-vlm-base-url"
      model-placeholder="例如: gemini-2.0-flash"
      fetch-variant="primary"
      :fetching-models="isFetchingModels"
      :model-options="modelOptions"
      :model-count="modelCount"
      :show-base-url="showBaseUrl"
      test-placement="panel-end"
      :testing="isTesting"
      @provider-change="onProviderChange"
      @model-change="selectModel"
      @fetch="fetchModels"
      @test="testConnection"
    />

    <UiFormGrid class="vlm-settings-tab__execution-grid">
      <UiField
        variant="settings"
        label="RPM 限制"
        hint="每分钟最大请求数，0 表示不限制"
        control-id="insight-vlm-rpm-limit"
      >
        <UiNumberField v-model="rpmLimit" input-id="insight-vlm-rpm-limit" :min="0" />
      </UiField>
      <UiField
        variant="settings"
        label="传输重试"
        hint="网络超时/429/5xx"
        control-id="insight-vlm-transport-retries"
      >
        <UiNumberField
          v-model="transportRetries"
          input-id="insight-vlm-transport-retries"
          :min="0"
        />
      </UiField>
      <UiField
        variant="settings"
        label="业务重试"
        hint="空结果/结构解析失败"
        control-id="insight-vlm-business-retries"
      >
        <UiNumberField
          v-model="businessRetries"
          input-id="insight-vlm-business-retries"
          :min="0"
        />
      </UiField>
      <UiField
        variant="settings"
        label="温度"
        hint="0-2，越低越确定"
        control-id="insight-vlm-temperature"
      >
        <UiNumberField
          v-model="temperature"
          input-id="insight-vlm-temperature"
          :min="0"
          :max="2"
          :step="0.1"
        />
      </UiField>
    </UiFormGrid>

    <UiField
      variant="settings"
      control="checkbox"
      hint="对 OpenAI 兼容 API 启用 response_format: json_object"
    >
      <UiCheckbox v-model="forceJsonOutput" label="强制 JSON 输出" />
    </UiField>

    <UiField variant="settings" control="checkbox" hint="流式请求可避免长时间等待导致的超时问题">
      <UiCheckbox v-model="useStream" label="使用流式请求" />
    </UiField>

    <UiField variant="settings">
      <OpenAIExtraBodyEditor v-model="extraBody" />
    </UiField>

    <UiField
      variant="settings"
      label="图片压缩（最大边长）"
      hint="发送前将图片等比例缩放到指定最大边长（像素），0 表示不压缩"
      control-id="insight-vlm-image-max-size"
    >
      <UiNumberField
        v-model="imageMaxSize"
        input-id="insight-vlm-image-max-size"
        :min="0"
        :step="128"
      />
    </UiField>
  </InsightSettingsPanel>
</template>

<style scoped>
.vlm-settings-tab {
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-number-field-text-align: left;
}

.vlm-settings-tab__execution-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

@media (--breakpoint-md-down) {
  .vlm-settings-tab__execution-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
