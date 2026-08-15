<script setup lang="ts">
import { computed } from 'vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'

const props = withDefaults(defineProps<{
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
  hasStoredCredential?: boolean
  storedCredentialHint?: string
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
  hasStoredCredential: false,
  storedCredentialHint: '凭据已安全保存在后端；留空表示保持不变，输入新值可替换',
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

const storedCredentialMessage = computed(() => (
  !props.apiKey && props.hasStoredCredential
    ? props.storedCredentialHint
    : ''
))

const resolvedApiKeyPlaceholder = computed(() => (
  storedCredentialMessage.value
    ? '已保存在后端，留空保持不变'
    : props.apiKeyPlaceholder
))
</script>

<template>
  <UiField
    v-if="includeApiKey"
    v-show="showApiKey"
    :class="fieldClass"
    variant="settings"
    :label="apiKeyLabel"
    :control-id="apiKeyInputId"
    :hint="storedCredentialMessage"
  >
    <UiPasswordField
      :model-value="apiKey"
      :input-id="apiKeyInputId"
      :placeholder="resolvedApiKeyPlaceholder"
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
