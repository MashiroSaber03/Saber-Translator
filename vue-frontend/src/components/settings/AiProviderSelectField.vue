<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

withDefaults(defineProps<{
  modelValue: string
  inputId: string
  options: UiSelectOption[]
  label?: string
  hint?: string
  disabled?: boolean
  fieldClass?: string
}>(), {
  label: '服务商',
  hint: '',
  disabled: false,
  fieldClass: '',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: [value: string]
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
</template>
