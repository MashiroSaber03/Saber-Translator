<script setup lang="ts">
import { computed, ref, useAttrs } from 'vue'

defineOptions({ inheritAttrs: false })

const props = withDefaults(defineProps<{
  modelValue?: string | number | boolean
  value?: string | number | boolean
  checked?: boolean
  type?: string
  placeholder?: string
  disabled?: boolean
  readonly?: boolean
  error?: boolean | string
  size?: 'lg' | 'md' | 'sm' | 'xs'
}>(), {
  modelValue: undefined,
  value: undefined,
  checked: undefined,
  type: 'text',
  placeholder: '',
  disabled: false,
  readonly: false,
  error: false,
  size: 'md',
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number | boolean]
  input: [event: Event]
}>()

const attrs = useAttrs()
const isComposing = ref(false)
const inputValue = computed(() => props.modelValue ?? props.value ?? '')
const textCompositionInputTypes = new Set([
  'text',
  'search',
  'url',
  'tel',
  'email',
  'password',
])
const supportsComposition = computed(() => textCompositionInputTypes.has((props.type || 'text').toLowerCase()))

function handleInput(event: Event) {
  const target = event.target as HTMLInputElement & { composing?: boolean }
  if (supportsComposition.value && (isComposing.value || target.composing)) return
  if (props.type === 'checkbox') {
    emit('update:modelValue', target.checked)
    emit('input', event)
    return
  }
  emit('update:modelValue', props.type === 'number' && target.value !== '' ? Number(target.value) : target.value)
  emit('input', event)
}

function handleCompositionStart(event: CompositionEvent) {
  if (!supportsComposition.value) return
  isComposing.value = true
  ;(event.target as HTMLInputElement & { composing?: boolean }).composing = true
}

function handleCompositionEnd(event: CompositionEvent) {
  if (!supportsComposition.value) return
  const target = event.target as HTMLInputElement & { composing?: boolean }
  if (!isComposing.value && !target.composing) return
  isComposing.value = false
  target.composing = false
  target.dispatchEvent(new Event('input', { bubbles: true }))
}
</script>

<template>
  <input
    v-bind="attrs"
    class="ui-input"
    :class="[`ui-input--${size}`, { 'ui-input--error': Boolean(error) }]"
    :value="inputValue"
    :checked="type === 'checkbox' ? checked ?? Boolean(modelValue) : undefined"
    :type="type"
    :placeholder="placeholder"
    :disabled="disabled"
    :readonly="readonly"
    :aria-invalid="Boolean(error) ? 'true' : undefined"
    @compositionstart="handleCompositionStart"
    @compositionend="handleCompositionEnd"
    @input="handleInput"
  >
</template>

<style scoped>
:where(.ui-input) {
  box-sizing: border-box;
  width: 100%;
  min-height: var(--ui-input-min-height, 38px);
  padding: var(--ui-input-padding, 9px 12px);
  border: var(--ui-input-border, 1px solid var(--color-border-muted));
  border-radius: var(--ui-input-radius, 6px);
  background: var(--ui-input-background, var(--color-surface-input, var(--color-surface-card)));
  color: var(--ui-input-color, var(--color-text-default));
  font-family: inherit;
  font-size: var(--ui-input-font-size, inherit);
  line-height: var(--ui-input-line-height, normal);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

:where(.ui-input):focus {
  outline: none;
  border-color: var(--ui-input-focus-border, var(--color-action-primary));
  box-shadow: 0 0 0 3px var(--ui-input-shadow-default);
}

:where(.ui-input[type='checkbox']),
:where(.ui-input[type='radio']),
:where(.ui-input[type='range']) {
  width: auto;
  min-height: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
  margin: var(--ui-input-control-margin);
}

:where(.ui-input[type='checkbox']):focus,
:where(.ui-input[type='radio']):focus,
:where(.ui-input[type='range']):focus {
  box-shadow: none;
}

:where(.ui-input--lg) {
  min-height: 44px;
  padding: 11px 14px;
  font-size: 1rem;
}

:where(.ui-input--sm) {
  min-height: 32px;
  padding: 6px 10px;
  font-size: 0.85rem;
}

:where(.ui-input--xs) {
  min-height: 28px;
  padding: 4px 8px;
  font-size: 0.78rem;
}

:where(.ui-input--error) {
  border-color: var(--color-status-error, var(--ui-input-border-default));
}

:where(.ui-input):disabled {
  opacity: 0.65;
  cursor: not-allowed;
}
</style>
