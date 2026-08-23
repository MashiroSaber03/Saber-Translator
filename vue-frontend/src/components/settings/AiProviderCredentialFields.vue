<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'

withDefaults(defineProps<{
  apiKey: string
  apiKeyInputId: string
  baseUrl: string
  baseUrlInputId: string
  showApiKey?: boolean
  showBaseUrl?: boolean
  includeApiKey?: boolean
  includeBaseUrl?: boolean
  apiKeyLabel?: string
  apiKeyPlaceholder?: string
  apiKeyShowLabel?: string
  apiKeyHideLabel?: string
  baseUrlLabel?: string
  baseUrlPlaceholder?: string
  disabled?: boolean
  fieldClass?: string
}>(), {
  showApiKey: true,
  showBaseUrl: false,
  includeApiKey: true,
  includeBaseUrl: true,
  apiKeyLabel: 'API Key',
  apiKeyPlaceholder: '输入 API Key',
  apiKeyShowLabel: '显示 API Key',
  apiKeyHideLabel: '隐藏 API Key',
  baseUrlLabel: 'Base URL',
  baseUrlPlaceholder: '自定义 API 地址',
  disabled: false,
  fieldClass: '',
})

const emit = defineEmits<{
  'update:apiKey': [value: string]
  'update:baseUrl': [value: string]
}>()

function updateBaseUrl(value: string | number | boolean): void {
  if (typeof value === 'string') emit('update:baseUrl', value)
}

</script>

<template>
  <UiField
    v-if="includeApiKey"
    v-show="showApiKey"
    :class="fieldClass"
    variant="settings"
    :label="apiKeyLabel"
    :control-id="apiKeyInputId"
  >
    <UiPasswordField
      :model-value="apiKey"
      :input-id="apiKeyInputId"
      :placeholder="apiKeyPlaceholder"
      :disabled="disabled"
      :show-label="apiKeyShowLabel"
      :hide-label="apiKeyHideLabel"
      @update:model-value="emit('update:apiKey', $event)"
    />
  </UiField>

  <UiField
    v-if="includeBaseUrl"
    v-show="showBaseUrl"
    :class="fieldClass"
    variant="settings"
    :label="baseUrlLabel"
    :control-id="baseUrlInputId"
  >
    <UiInput
      :id="baseUrlInputId"
      :model-value="baseUrl"
      type="text"
      :placeholder="baseUrlPlaceholder"
      :disabled="disabled"
      @update:model-value="updateBaseUrl"
    />
  </UiField>
</template>
