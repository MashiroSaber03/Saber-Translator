<script setup lang="ts">
import { useAttrs } from 'vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

defineOptions({ inheritAttrs: false })

const props = defineProps<{
  modelValue?: UiSelectValue
  options?: UiSelectOption[]
  disabled?: boolean
  error?: boolean | string
  size?: 'lg' | 'md' | 'sm' | 'xs'
  variant?: 'default' | 'studio'
}>()

const emit = defineEmits<{
  'update:modelValue': [value: UiSelectValue]
  change: [value: UiSelectValue]
}>()

const attrs = useAttrs()

function handleChange(event: Event) {
  const rawValue = (event.target as HTMLSelectElement).value
  const value = props.options?.find(option => String(option.value) === rawValue)?.value ?? rawValue
  emit('update:modelValue', value)
  emit('change', value)
}
</script>

<template>
  <select
    v-bind="attrs"
    class="ui-select"
    :class="[`ui-select--${size || 'md'}`, `ui-select--${variant || 'default'}`, { 'ui-select--error': Boolean(error) }]"
    :value="modelValue"
    :disabled="disabled"
    :aria-invalid="Boolean(error) ? 'true' : undefined"
    @change="handleChange"
  >
    <slot>
      <option
        v-for="option in options || []"
        :key="String(option.value)"
        :value="option.value"
        :disabled="option.disabled"
      >
        {{ option.label }}
      </option>
    </slot>
  </select>
</template>

<style scoped>
:where(.ui-select) {
  box-sizing: border-box;
  width: 100%;
  min-height: var(--ui-select-min-height, 38px);
  padding: var(--ui-select-padding, 9px 12px);
  border: var(--ui-select-border, 1px solid var(--color-border-muted));
  border-radius: var(--ui-select-radius, 6px);
  background: var(--ui-select-background, var(--color-surface-input, var(--color-surface-card)));
  color: var(--ui-select-color, var(--color-text-default));
  font-family: inherit;
  font-size: var(--ui-select-font-size, inherit);
  line-height: var(--ui-select-line-height, normal);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

:where(.ui-select):focus {
  outline: none;
  border-color: var(--color-action-primary);
  box-shadow: 0 0 0 3px var(--ui-select-focus-shadow);
}

:where(.ui-select--lg) {
  min-height: 44px;
  padding: 11px 14px;
  font-size: 1rem;
}

:where(.ui-select--sm) {
  min-height: 32px;
  padding: 6px 10px;
  font-size: 0.85rem;
}

:where(.ui-select--xs) {
  min-height: 28px;
  padding: 4px 8px;
  font-size: 0.78rem;
}

:where(.ui-select--studio) {
  min-height: 38px;
  padding: var(--ui-select-studio-padding, 10px 12px);
  border: var(--ui-select-studio-border, 1px solid var(--color-border-muted));
  border-radius: var(--ui-select-studio-radius, 14px);
  background: var(--ui-select-studio-background, var(--color-surface-input, var(--color-surface-card)));
  color: var(--ui-select-studio-color, var(--color-text-default));
  font-size: var(--ui-select-studio-font-size, 13px);
}

:where(.ui-select--studio.ui-select--lg) {
  min-height: 44px;
  padding: var(--ui-select-studio-lg-padding, 12px 14px);
  border-radius: var(--ui-select-studio-lg-radius, 16px);
}

:where(.ui-select--studio):focus {
  border-color: var(--ui-select-studio-focus-border, var(--color-border-brand));
  box-shadow: 0 0 0 3px var(--ui-select-studio-focus-shadow, var(--color-focus-brand-soft));
}

:where(.ui-select--error) {
  border-color: var(--color-status-error, var(--ui-select-error-border));
}

:where(.ui-select):disabled {
  opacity: 0.65;
  cursor: not-allowed;
}
</style>
