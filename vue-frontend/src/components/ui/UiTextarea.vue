<script setup lang="ts">
import { computed, ref, useAttrs } from 'vue'

defineOptions({ inheritAttrs: false })

const props = withDefaults(defineProps<{
  modelValue?: string
  value?: string
  placeholder?: string
  rows?: number | string
  disabled?: boolean
  readonly?: boolean
  error?: boolean | string
  size?: 'lg' | 'md' | 'sm' | 'xs'
  variant?: 'default' | 'panel' | 'studio'
}>(), {
  modelValue: undefined,
  value: undefined,
  placeholder: '',
  rows: 4,
  disabled: false,
  readonly: false,
  error: false,
  size: 'md',
  variant: 'default',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  input: [event: Event]
}>()

const attrs = useAttrs()
const isComposing = ref(false)
const textareaRef = ref<HTMLTextAreaElement | null>(null)
const textareaValue = computed(() => props.modelValue ?? props.value ?? '')

function handleCompositionStart(event: CompositionEvent) {
  isComposing.value = true
  ;(event.target as HTMLTextAreaElement & { composing?: boolean }).composing = true
}

function handleCompositionEnd(event: CompositionEvent) {
  const target = event.target as HTMLTextAreaElement & { composing?: boolean }
  if (!isComposing.value && !target.composing) return
  isComposing.value = false
  target.composing = false
  target.dispatchEvent(new Event('input', { bubbles: true }))
}

function handleInput(event: Event) {
  const target = event.target as HTMLTextAreaElement & { composing?: boolean }
  if (isComposing.value || target.composing) return
  emit('update:modelValue', target.value)
  emit('input', event)
}

defineExpose({
  focus() {
    textareaRef.value?.focus()
  },
  get selectionStart() {
    return textareaRef.value?.selectionStart ?? null
  },
  set selectionStart(value: number | null) {
    if (textareaRef.value && value !== null) {
      textareaRef.value.selectionStart = value
    }
  },
  get selectionEnd() {
    return textareaRef.value?.selectionEnd ?? null
  },
  set selectionEnd(value: number | null) {
    if (textareaRef.value && value !== null) {
      textareaRef.value.selectionEnd = value
    }
  },
})
</script>

<template>
  <textarea
    ref="textareaRef"
    v-bind="attrs"
    class="ui-textarea"
    :class="[
      `ui-textarea--${size}`,
      `ui-textarea--${variant}`,
      { 'ui-textarea--error': Boolean(error) },
    ]"
    :value="textareaValue"
    :placeholder="placeholder"
    :rows="rows"
    :disabled="disabled"
    :readonly="readonly"
    :aria-invalid="Boolean(error) ? 'true' : undefined"
    @compositionstart="handleCompositionStart"
    @compositionend="handleCompositionEnd"
    @input="handleInput"
  />
</template>

<style scoped>
:where(.ui-textarea) {
  box-sizing: border-box;
  width: 100%;
  min-height: var(--ui-textarea-min-height, 96px);
  padding: var(--ui-textarea-padding, 10px 12px);
  border: var(--ui-textarea-border, 1px solid var(--color-border-muted));
  border-radius: var(--ui-textarea-radius, 6px);
  resize: vertical;
  background: var(--ui-textarea-background, var(--color-surface-input, var(--color-surface-card)));
  color: var(--ui-textarea-color, var(--color-text-default));
  font-family: inherit;
  font-size: var(--ui-textarea-font-size, inherit);
  line-height: var(--ui-textarea-line-height, normal);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

:where(.ui-textarea):focus {
  outline: none;
  border-color: var(--ui-textarea-focus-border, var(--color-border-brand));
  box-shadow: 0 0 0 3px var(--ui-textarea-focus-shadow);
}

:where(.ui-textarea--lg) {
  min-height: 128px;
  padding: 12px 14px;
  font-size: 1rem;
}

:where(.ui-textarea--sm) {
  min-height: 72px;
  padding: 8px 10px;
  font-size: 0.85rem;
}

:where(.ui-textarea--xs) {
  min-height: 56px;
  padding: 6px 8px;
  font-size: 0.78rem;
}

:where(.ui-textarea--panel) {
  padding: 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default);
  line-height: 1.6;
}

:where(.ui-textarea--panel.ui-textarea--lg) {
  padding: 16px;
  font-size: 14px;
}

:where(.ui-textarea--panel):focus {
  border-color: var(--color-border-brand);
  box-shadow: 0 0 0 3px var(--color-focus-brand-soft);
}

:where(.ui-textarea--studio) {
  padding: var(--ui-textarea-studio-padding, 10px 12px);
  border: var(--ui-textarea-studio-border, 1px solid var(--color-border-muted));
  border-radius: var(--ui-textarea-studio-radius, 14px);
  background: var(--ui-textarea-studio-background, var(--color-surface-input, var(--color-surface-card)));
  color: var(--ui-textarea-studio-color, var(--color-text-default));
  font-size: var(--ui-textarea-studio-font-size, 13px);
  line-height: var(--ui-textarea-studio-line-height, 1.7);
}

:where(.ui-textarea--studio.ui-textarea--lg) {
  padding: var(--ui-textarea-studio-lg-padding, 12px 14px);
  border-radius: var(--ui-textarea-studio-lg-radius, 16px);
}

:where(.ui-textarea--studio):focus {
  border-color: var(--ui-textarea-studio-focus-border, var(--color-border-brand));
  box-shadow: 0 0 0 3px var(--ui-textarea-studio-focus-shadow, var(--color-focus-brand-soft));
}

:where(.ui-textarea--error) {
  border-color: var(--color-status-error, var(--ui-textarea-error-border));
}

:where(.ui-textarea):disabled {
  opacity: 0.65;
  cursor: not-allowed;
}
</style>
