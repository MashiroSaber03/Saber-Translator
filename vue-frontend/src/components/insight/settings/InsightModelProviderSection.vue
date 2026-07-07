<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

withDefaults(defineProps<{
  provider: string
  providerOptions: UiSelectOption[]
  apiKey?: string
  model: string
  baseUrl?: string
  showApiKey?: boolean
  credentialId: string
  providerInputId: string
  modelInputId: string
  baseUrlInputId: string
  modelPlaceholder: string
  modelHint?: string
  modelError?: string
  showBaseUrl?: boolean
  baseUrlPlaceholder?: string
  showFetch?: boolean
  fetchVariant?: 'primary' | 'secondary'
  fetchingModels?: boolean
  modelOptions?: UiSelectOption[]
  modelCount?: number
  showTest?: boolean
  testing?: boolean
  testLabel?: string
  testingLabel?: string
}>(), {
  apiKey: '',
  baseUrl: '',
  showApiKey: false,
  modelHint: '',
  modelError: '',
  showBaseUrl: false,
  baseUrlPlaceholder: '自定义 API 地址',
  showFetch: true,
  fetchVariant: 'secondary',
  fetchingModels: false,
  modelOptions: () => [],
  modelCount: 0,
  showTest: true,
  testing: false,
  testLabel: '测试连接',
  testingLabel: '测试中...',
})

const emit = defineEmits<{
  'update:provider': [value: string]
  'update:apiKey': [value: string]
  'update:model': [value: string]
  'update:baseUrl': [value: string]
  'provider-change': [value: string]
  'model-change': [value: string]
  fetch: []
  test: []
}>()

function asString(value: UiSelectValue | string | number | boolean): string {
  return String(value)
}

function handleProviderUpdate(value: UiSelectValue): void {
  emit('update:provider', asString(value))
}

function handleProviderChange(value: UiSelectValue): void {
  emit('provider-change', asString(value))
}

function handleApiKeyUpdate(value: string): void {
  emit('update:apiKey', value)
}

function handleModelUpdate(value: UiSelectValue): void {
  emit('update:model', asString(value))
}

function handleModelChange(value: UiSelectValue): void {
  emit('model-change', asString(value))
}

function handleBaseUrlUpdate(value: string | number | boolean): void {
  emit('update:baseUrl', asString(value))
}
</script>

<template>
  <div class="insight-model-provider-section">
    <UiField variant="settings" label="服务商" :control-id="providerInputId">
      <UiSelect
        :id="providerInputId"
        :model-value="provider"
        :options="providerOptions"
        @update:model-value="handleProviderUpdate"
        @change="handleProviderChange"
      />
    </UiField>

    <UiField
      v-if="showApiKey"
      variant="settings"
      label="API Key"
      :control-id="credentialId"
    >
      <UiPasswordField
        :model-value="apiKey"
        :input-id="credentialId"
        placeholder="输入 API Key"
        @update:model-value="handleApiKeyUpdate"
      />
    </UiField>

    <UiField
      variant="settings"
      label="模型"
      :hint="modelHint"
      :error="modelError"
      :control-id="modelInputId"
    >
      <UiModelPicker
        :model-value="model"
        :input-id="modelInputId"
        :placeholder="modelPlaceholder"
        :fetch-variant="fetchVariant"
        :show-fetch="showFetch"
        :fetching="fetchingModels"
        :fetch-disabled="fetchingModels"
        :options="modelOptions"
        :model-count="modelCount"
        @update:model-value="handleModelUpdate"
        @change="handleModelChange"
        @fetch="$emit('fetch')"
      />
    </UiField>

    <UiField v-if="showBaseUrl" variant="settings" label="Base URL" :control-id="baseUrlInputId">
      <UiInput
        :id="baseUrlInputId"
        :model-value="baseUrl"
        type="text"
        :placeholder="baseUrlPlaceholder"
        @update:model-value="handleBaseUrlUpdate"
      />
    </UiField>

    <UiButton
      v-if="showTest"
      variant="secondary"
      :disabled="testing"
      @click="$emit('test')"
    >
      {{ testing ? testingLabel : testLabel }}
    </UiButton>
  </div>
</template>

<style scoped>
.insight-model-provider-section {
  display: block;
}
</style>
