<script setup lang="ts">
import { ref } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: string
  inputId?: string
  disabled?: boolean
  hidden?: boolean
  ariaLabel?: string
  size?: 'sm' | 'md'
  title?: string
}>(), {
  inputId: undefined,
  disabled: false,
  hidden: false,
  ariaLabel: undefined,
  size: 'md',
  title: undefined,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: [value: string]
}>()

const inputRef = ref<HTMLInputElement | null>(null)

function handleInput(event: Event): void {
  const value = (event.target as HTMLInputElement).value
  emit('update:modelValue', value)
  emit('change', value)
}

function click(): void {
  inputRef.value?.click()
}

function focus(): void {
  inputRef.value?.focus()
}

defineExpose({ click, focus })
</script>

<template>
  <input
    ref="inputRef"
    :id="inputId"
    class="ui-color-input"
    :class="`ui-color-input--${props.size}`"
    type="color"
    :value="modelValue"
    :disabled="disabled"
    :hidden="hidden"
    :aria-label="ariaLabel"
    :title="title"
    @input="handleInput"
  />
</template>

<style scoped>
.ui-color-input {
  width: var(--ui-colorpicker-width, 72px);
  height: var(--ui-colorpicker-height, 34px);
  padding: var(--ui-colorpicker-padding, 2px);
  border: var(--ui-colorpicker-border, 1px solid var(--color-border-input));
  border-radius: var(--ui-colorpicker-radius, 8px);
  background: var(--ui-colorpicker-background, var(--color-surface-base));
  cursor: pointer;
}

.ui-color-input--sm {
  --ui-colorpicker-width: 58px;
}

.ui-color-input:disabled {
  opacity: var(--ui-colorpicker-disabled-opacity, 0.5);
  cursor: not-allowed;
}

</style>
