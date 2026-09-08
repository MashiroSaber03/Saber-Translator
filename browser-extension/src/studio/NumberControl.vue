<script setup lang="ts">
const props = defineProps<{
  modelValue: number
  label: string
  min?: number
  max?: number
  step?: number
  spinStep?: number
  disabled?: boolean
}>()
const emit = defineEmits<{ 'update:modelValue': [number] }>()
function input(event: Event) {
  const node = event.target as HTMLInputElement
  if (node.validity.valid && Number.isFinite(node.valueAsNumber))
    emit('update:modelValue', node.valueAsNumber)
}
function spin(direction: -1 | 1) {
  if (props.disabled) return
  const value = Number((props.modelValue + direction * (props.spinStep ?? props.step ?? 1)).toPrecision(12))
  emit('update:modelValue', Math.min(props.max ?? Infinity, Math.max(props.min ?? -Infinity, value)))
}
function arrow(event: KeyboardEvent) {
  if (props.spinStep === undefined || !['ArrowUp', 'ArrowDown'].includes(event.key)) return
  event.preventDefault()
  spin(event.key === 'ArrowUp' ? 1 : -1)
}
</script>
<template>
  <label class="field"
    >{{ label
    }}<span class="number-control" :class="{ 'number-control--custom-spin': spinStep !== undefined }"><input
      type="number"
      :aria-label="label"
      :value="modelValue"
      :min="min"
      :max="max"
      :step="step ?? 1"
      :disabled="disabled"
      @input="input"
      @keydown="arrow"
  /><span v-if="spinStep !== undefined" class="number-control__arrows">
      <button type="button" :aria-label="`增加${label}`" :disabled="disabled || (max !== undefined && modelValue >= max)" @click="spin(1)">▴</button>
      <button type="button" :aria-label="`减少${label}`" :disabled="disabled || (min !== undefined && modelValue <= min)" @click="spin(-1)">▾</button>
  </span></span></label>
</template>
<style scoped>
.number-control { display: contents; }
.number-control--custom-spin { display: block; position: relative; }
.number-control--custom-spin input { appearance: textfield; padding-right: 28px; }
.number-control--custom-spin input::-webkit-inner-spin-button,
.number-control--custom-spin input::-webkit-outer-spin-button { appearance: none; margin: 0; }
.number-control__arrows { position: absolute; right: 4px; top: 4px; bottom: 4px; display: flex; flex-direction: column; width: 22px; }
.number-control__arrows button { flex: 1; border: 0; padding: 0; border-radius: 4px; background: transparent; color: inherit; font-size: 12px; line-height: 1; cursor: pointer; }
.number-control__arrows button:hover:not(:disabled) { background: #ec489922; }
.number-control__arrows button:disabled { opacity: 0.4; cursor: default; }
</style>
