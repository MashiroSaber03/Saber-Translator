<script setup lang="ts">
import { useAttrs } from 'vue'

defineOptions({ inheritAttrs: false })

type OptionValue = string | number

defineProps<{
  modelValue?: OptionValue
  options?: Array<{ label: string; value: OptionValue; disabled?: boolean }>
  disabled?: boolean
  error?: boolean | string
  size?: 'lg' | 'md' | 'sm' | 'xs'
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: [value: string]
}>()

const attrs = useAttrs()

function handleChange(event: Event) {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
  emit('change', value)
}
</script>

<template>
  <select
    v-bind="attrs"
    class="ui-select"
    :class="[`ui-select--${size || 'md'}`, { 'ui-select--error': Boolean(error) }]"
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
  box-shadow: 0 0 0 3px var(--ui-select-shadow-default);
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

:where(.ui-select--error) {
  border-color: var(--color-status-error, var(--ui-select-border-default));
}

:where(.ui-select):disabled {
  opacity: 0.65;
  cursor: not-allowed;
}
</style>
