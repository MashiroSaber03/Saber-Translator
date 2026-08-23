<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'
import type { CustomAiProfile, CustomAiProfileKind } from '@/types/customAiProfile'
import CustomAiProfilePicker from './CustomAiProfilePicker.vue'

withDefaults(defineProps<{
  modelValue: string
  inputId: string
  options: UiSelectOption[]
  label?: string
  hint?: string
  disabled?: boolean
  fieldClass?: string
  customProfileKind?: CustomAiProfileKind
  customProfileApiKey?: string
  customProfileBaseUrl?: string
  customProfileModel?: string
}>(), {
  label: '服务商',
  hint: '',
  disabled: false,
  fieldClass: '',
  customProfileKind: undefined,
  customProfileApiKey: '',
  customProfileBaseUrl: '',
  customProfileModel: '',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: [value: string]
  'apply-custom-profile': [profile: CustomAiProfile]
}>()

function emitModelValue(value: UiSelectValue): void {
  if (typeof value === 'string') emit('update:modelValue', value)
}

function emitChange(value: UiSelectValue): void {
  if (typeof value === 'string') emit('change', value)
}
</script>

<template>
  <UiField :class="fieldClass" variant="settings" :label="label" :hint="hint" :control-id="inputId">
    <UiSelect
      :id="inputId"
      :model-value="modelValue"
      :options="options"
      :disabled="disabled"
      @update:model-value="emitModelValue"
      @change="emitChange"
    />
  </UiField>
  <CustomAiProfilePicker
    v-if="modelValue === 'custom' && customProfileKind"
    :input-id="`${inputId}CustomProfile`"
    :kind="customProfileKind"
    :api-key="customProfileApiKey"
    :base-url="customProfileBaseUrl"
    :model="customProfileModel"
    :disabled="disabled"
    @apply="emit('apply-custom-profile', $event)"
  />
</template>
