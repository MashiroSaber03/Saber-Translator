<script setup lang="ts">
import { computed } from 'vue'

import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiInput from '@/components/ui/UiInput.vue'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  controls?: boolean
  controlsPlacement?: 'split' | 'after'
  decrementLabel?: string
  decrementText?: string
  disabled?: boolean
  incrementLabel?: string
  incrementText?: string
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
  controlsPlacement: 'split',
  decrementLabel: '减少数值',
  decrementText: '',
  disabled: false,
  incrementLabel: '增加数值',
  incrementText: '',
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

function handleInputValue(value: string | number): void {
  if (props.nullable && value === '') {
    emit('update:modelValue', null)
    emit('change', null)
    return
  }
  if (value === '') return
  commitValue(Number(value))
}

function restoreEmptyInput(event: FocusEvent): void {
  if (props.nullable) return
  const target = event.target
  if (!(target instanceof HTMLInputElement) || target.value !== '') return
  target.value = String(props.modelValue ?? props.min ?? 0)
}

function stepBy(direction: -1 | 1): void {
  if ((direction < 0 && !canDecrement.value) || (direction > 0 && !canIncrement.value)) return
  if (props.modelValue === null) {
    commitValue(props.min ?? 0)
    return
  }
  commitValue(Number((props.modelValue + props.step * direction).toPrecision(12)))
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
      v-if="controls && controlsPlacement === 'split'"
      variant="secondary"
      icon
      :size="size"
      :aria-label="decrementLabel"
      :disabled="!canDecrement"
      @click="stepBy(-1)"
    >
      <span v-if="decrementText" class="ui-number-field__control-text">{{ decrementText }}</span>
      <UiIcon v-else name="minus" />
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
      @blur="restoreEmptyInput"
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
      <span v-if="incrementText" class="ui-number-field__control-text">{{ incrementText }}</span>
      <UiIcon v-else name="plus" />
    </UiButton>

    <UiButton
      v-if="controls && controlsPlacement === 'after'"
      variant="secondary"
      icon
      :size="size"
      :aria-label="decrementLabel"
      :disabled="!canDecrement"
      @click="stepBy(-1)"
    >
      <span v-if="decrementText" class="ui-number-field__control-text">{{ decrementText }}</span>
      <UiIcon v-else name="minus" />
    </UiButton>
  </div>
</template>

<style scoped>
.ui-number-field {
  --internal-ui-number-field-input-width: 120px;

  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: var(--ui-number-field-width, fit-content);
}

.ui-number-field__input {
  width: var(--ui-number-field-input-width, var(--internal-ui-number-field-input-width));
  text-align: var(--ui-number-field-text-align, center);
}

.ui-number-field--xs {
  --internal-ui-number-field-input-width: 72px;
}

.ui-number-field--sm {
  --internal-ui-number-field-input-width: 96px;
}

.ui-number-field--with-controls {
  --ui-button-icon-width: var(--ui-number-field-control-width, 28px);
  --ui-button-icon-height: var(--ui-number-field-control-height, 28px);
}

.ui-number-field__control-text {
  font-size: 13px;
  font-weight: 600;
  line-height: 1;
}

.ui-number-field__input::-webkit-inner-spin-button,
.ui-number-field__input::-webkit-outer-spin-button {
  appearance: none;
  margin: 0;
}
</style>
