<script setup lang="ts">
import { computed, ref, useAttrs } from 'vue'

defineOptions({ inheritAttrs: false })

const props = withDefaults(defineProps<{
  modelValue?: string | number | boolean
  type?: string
  placeholder?: string
  disabled?: boolean
  readonly?: boolean
  error?: boolean | string
  size?: 'lg' | 'md' | 'sm' | 'xs'
  variant?: 'default' | 'editor' | 'studio' | 'embedded'
}>(), {
  modelValue: undefined,
  type: 'text',
  placeholder: '',
  disabled: false,
  readonly: false,
  error: false,
  size: 'md',
  variant: 'default',
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number | boolean]
  input: [event: Event]
}>()

const attrs = useAttrs()
const inputRef = ref<HTMLInputElement | null>(null)
const isComposing = ref(false)
const inputValue = computed(() => props.modelValue ?? '')
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

function focus() {
  inputRef.value?.focus()
}

defineExpose({ focus })
</script>

<template>
  <input
    ref="inputRef"
    v-bind="attrs"
    class="ui-input"
    :class="[`ui-input--${size}`, `ui-input--${variant}`, { 'ui-input--error': Boolean(error) }]"
    :value="inputValue"
    :checked="type === 'checkbox' ? Boolean(modelValue) : undefined"
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
  border-color: var(--ui-input-focus-border, var(--color-border-brand));
  box-shadow: 0 0 0 3px var(--ui-input-focus-shadow);
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

:where(.ui-input--editor) {
  border: var(--ui-input-editor-border, 1px solid var(--color-border-input));
  background: var(--ui-input-editor-background, var(--color-surface-base));
  color: var(--ui-input-editor-color, var(--color-text-strong));
}

:where(.ui-input--editor):focus {
  border-color: var(--ui-input-editor-focus-border, var(--color-border-brand-gradient));
  box-shadow: 0 0 0 3px var(--ui-input-editor-focus-shadow, color-mix(in srgb, var(--color-action-brand) 16%, transparent));
}

:where(.ui-input--studio) {
  min-height: 38px;
  padding: var(--ui-input-studio-padding, 10px 12px);
  border: var(--ui-input-studio-border, 1px solid var(--color-border-muted));
  border-radius: var(--ui-input-studio-radius, 14px);
  background: var(--ui-input-studio-background, var(--color-surface-input, var(--color-surface-card)));
  color: var(--ui-input-studio-color, var(--color-text-default));
  font-size: var(--ui-input-studio-font-size, 13px);
}

:where(.ui-input--studio.ui-input--lg) {
  min-height: 44px;
  padding: var(--ui-input-studio-lg-padding, 12px 14px);
  border-radius: var(--ui-input-studio-lg-radius, 16px);
}

:where(.ui-input--studio):focus {
  border-color: var(--ui-input-studio-focus-border, var(--color-border-brand));
  box-shadow: 0 0 0 3px var(--ui-input-studio-focus-shadow, var(--color-focus-brand-soft));
}

:where(.ui-input--embedded) {
  padding: var(--ui-input-embedded-padding, 8px);
  border: var(--ui-input-embedded-border, none);
  border-radius: var(--ui-input-embedded-radius, 0);
  background: var(--ui-input-embedded-background, transparent);
  font-size: var(--ui-input-embedded-font-size, 14px);
}

:where(.ui-input--error) {
  border-color: var(--color-status-error, var(--ui-input-error-border));
}

:where(.ui-input):disabled {
  opacity: 0.65;
  cursor: not-allowed;
}
</style>
