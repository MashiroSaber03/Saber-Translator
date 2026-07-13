<script setup lang="ts">
import { computed } from 'vue'

import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiInput from '@/components/ui/UiInput.vue'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  controls?: boolean
  decrementLabel?: string
  disabled?: boolean
  incrementLabel?: string
  inputId?: string
  max?: number
  min?: number
  modelValue: number | null
  nullable?: boolean
  size?: 'lg' | 'md' | 'sm' | 'xs'
  step?: number
  title?: string
  variant?: 'default' | 'editor' | 'studio'
}>(), {
  ariaLabel: undefined,
  controls: false,
  decrementLabel: '减少数值',
  disabled: false,
  incrementLabel: '增加数值',
  inputId: undefined,
  max: undefined,
  min: undefined,
  nullable: false,
  size: 'sm',
  step: 1,
  title: '',
  variant: 'default',
})

const emit = defineEmits<{
  'update:modelValue': [value: number | null]
  change: [value: number | null]
}>()

const canDecrement = computed(() => !props.disabled && (
  props.modelValue !== null && (props.min === undefined || props.modelValue > props.min)
))
const canIncrement = computed(() => !props.disabled && (
  props.modelValue === null || props.max === undefined || props.modelValue < props.max
))
const inputValue = computed(() => props.modelValue ?? '')

function clampValue(value: number): number {
  if (!Number.isFinite(value)) return props.min ?? 0
  const lowerBounded = props.min === undefined ? value : Math.max(props.min, value)
  return props.max === undefined ? lowerBounded : Math.min(props.max, lowerBounded)
}

function commitValue(value: number): void {
  const nextValue = clampValue(value)
  emit('update:modelValue', nextValue)
  emit('change', nextValue)
}

function handleInputValue(value: string | number | boolean): void {
  if (props.nullable && value === '') {
    emit('update:modelValue', null)
    emit('change', null)
    return
  }
  commitValue(Number(value))
}

function stepBy(direction: -1 | 1): void {
  if ((direction < 0 && !canDecrement.value) || (direction > 0 && !canIncrement.value)) return
  if (props.modelValue === null) {
    commitValue(props.min ?? 0)
    return
  }
  commitValue(props.modelValue + props.step * direction)
}
</script>

<template>
  <div
    class="ui-number-field"
    :class="[
      `ui-number-field--${size}`,
      { 'ui-number-field--with-controls': controls },
    ]"
  >
    <UiButton
      v-if="controls"
      variant="secondary"
      icon
      :size="size"
      :aria-label="decrementLabel"
      :disabled="!canDecrement"
      @click="stepBy(-1)"
    >
      <UiIcon name="minus" />
    </UiButton>

    <UiInput
      :id="inputId"
      class="ui-number-field__input"
      type="number"
      :model-value="inputValue"
      :min="min"
      :max="max"
      :step="step"
      :size="size"
      :variant="variant"
      :disabled="disabled"
      :aria-label="ariaLabel"
      :title="title || undefined"
      @update:model-value="handleInputValue"
    />

    <UiButton
      v-if="controls"
      variant="secondary"
      icon
      :size="size"
      :aria-label="incrementLabel"
      :disabled="!canIncrement"
      @click="stepBy(1)"
    >
      <UiIcon name="plus" />
    </UiButton>
  </div>
</template>

<style scoped>
.ui-number-field {
  --internal-ui-number-field-input-width: 120px;

  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: fit-content;
}

.ui-number-field__input {
  width: var(--ui-number-field-input-width, var(--internal-ui-number-field-input-width));
  text-align: center;
}

.ui-number-field--xs {
  --internal-ui-number-field-input-width: 72px;
}

.ui-number-field--sm {
  --internal-ui-number-field-input-width: 96px;
}

.ui-number-field--with-controls {
  --ui-button-icon-width: 28px;
  --ui-button-icon-height: 28px;
}

.ui-number-field__input::-webkit-inner-spin-button,
.ui-number-field__input::-webkit-outer-spin-button {
  appearance: none;
  margin: 0;
}
</style>
