<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

export interface UiColorSwatchOption {
  label: string
  value: string
}

withDefaults(defineProps<{
  ariaLabel?: string
  modelValue: string
  options: UiColorSwatchOption[]
}>(), {
  ariaLabel: '颜色选项',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: [value: string]
}>()

function selectColor(value: string): void {
  emit('update:modelValue', value)
  emit('change', value)
}
</script>

<template>
  <div class="ui-color-swatch-group" role="group" :aria-label="ariaLabel">
    <UiButton
      v-for="option in options"
      :key="option.value"
      variant="toolbar"
      icon
      type="button"
      class="ui-color-swatch-group__swatch"
      :class="{ 'ui-color-swatch-group__swatch--selected': option.value === modelValue }"
      :style="{ '--ui-swatch-background': option.value }"
      :aria-label="option.label"
      :aria-pressed="option.value === modelValue ? 'true' : 'false'"
      :title="option.label"
      @click="selectColor(option.value)"
    />
  </div>
</template>

<style scoped>
.ui-color-swatch-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

button.ui-color-swatch-group__swatch {
  --ui-button-icon-width: 32px;
  --ui-button-icon-height: 32px;
  --ui-button-padding: 0;

  width: var(--ui-button-icon-width);
  height: var(--ui-button-icon-height);
  padding: var(--ui-button-padding);
  background: var(--ui-swatch-background);
  border: 2px solid transparent;
  border-radius: 6px;
  box-shadow: none;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.ui-color-swatch-group__swatch:hover {
  transform: scale(1.08);
}

.ui-color-swatch-group__swatch--selected {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 0 2px var(--color-focus-ring-soft);
}
</style>
