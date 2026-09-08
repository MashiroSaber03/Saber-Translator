<script setup lang="ts">
defineProps<{
  modelValue: number
  label: string
  min?: number
  max?: number
  step?: number
  disabled?: boolean
}>()
const emit = defineEmits<{ 'update:modelValue': [number] }>()
function input(event: Event) {
  const node = event.target as HTMLInputElement
  if (node.validity.valid && Number.isFinite(node.valueAsNumber))
    emit('update:modelValue', node.valueAsNumber)
}
</script>
<template>
  <label class="field"
    >{{ label
    }}<input
      type="number"
      :aria-label="label"
      :value="modelValue"
      :min="min"
      :max="max"
      :step="step ?? 1"
      :disabled="disabled"
      @input="input"
  /></label>
</template>
