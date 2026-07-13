<script setup lang="ts">
import { computed } from 'vue'

import UiButton from '@/components/ui/UiButton.vue'

const props = withDefaults(defineProps<{
  modelValue?: boolean
  ariaLabel: string
  title?: string
  disabled?: boolean
  size?: 'sm' | 'md'
}>(), {
  modelValue: false,
  title: '',
  disabled: false,
  size: 'md',
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  change: [value: boolean]
}>()

const checkedText = computed(() => props.modelValue ? 'true' : 'false')

function toggle(): void {
  if (props.disabled) return

  const nextValue = !props.modelValue
  emit('update:modelValue', nextValue)
  emit('change', nextValue)
}
</script>

<template>
  <UiButton
    variant="toolbar"
    class="ui-switch"
    :class="[
      `ui-switch--${size}`,
      { 'ui-switch--checked': modelValue },
    ]"
    role="switch"
    :aria-label="ariaLabel"
    :aria-checked="checkedText"
    :title="title || undefined"
    :disabled="disabled"
    @click="toggle"
  >
    <span class="ui-switch__track" aria-hidden="true">
      <span class="ui-switch__thumb"></span>
    </span>
  </UiButton>
</template>

<style scoped>
.ui-switch {
  --internal-ui-switch-width: 40px;
  --internal-ui-switch-height: 22px;
  --internal-ui-switch-thumb-size: 16px;
  --internal-ui-switch-thumb-offset: 3px;
  --internal-ui-switch-thumb-translate: 18px;

  position: relative;
  display: inline-flex;
  width: var(--ui-switch-width, var(--internal-ui-switch-width));
  height: var(--ui-switch-height, var(--internal-ui-switch-height));
  padding: 0;
  border: 0;
  border-radius: var(--ui-switch-height, var(--internal-ui-switch-height));
  background: transparent;
  flex-shrink: 0;
}

.ui-switch--sm {
  --internal-ui-switch-width: 32px;
  --internal-ui-switch-height: 18px;
  --internal-ui-switch-thumb-size: 14px;
  --internal-ui-switch-thumb-offset: 2px;
  --internal-ui-switch-thumb-translate: 14px;
}

.ui-switch__track {
  position: absolute;
  inset: 0;
  border-radius: var(--ui-switch-height, var(--internal-ui-switch-height));
  background: var(--ui-switch-track-background, var(--color-border-muted));
  transition: background 0.2s ease;
}

.ui-switch__thumb {
  position: absolute;
  width: var(--ui-switch-thumb-size, var(--internal-ui-switch-thumb-size));
  height: var(--ui-switch-thumb-size, var(--internal-ui-switch-thumb-size));
  left: var(--ui-switch-thumb-offset, var(--internal-ui-switch-thumb-offset));
  bottom: var(--ui-switch-thumb-offset, var(--internal-ui-switch-thumb-offset));
  border-radius: 50%;
  background: var(--ui-switch-thumb-background, var(--color-surface-base));
  box-shadow: var(--ui-switch-thumb-shadow, 0 1px 3px var(--shadow-medium));
  transition: transform 0.2s ease;
}

.ui-switch--checked .ui-switch__track {
  background: var(--ui-switch-track-checked-background, linear-gradient(135deg, var(--color-action-success) 0%, var(--color-action-success-strong) 100%));
}

.ui-switch--checked .ui-switch__thumb {
  transform: translateX(var(--ui-switch-thumb-translate, var(--internal-ui-switch-thumb-translate)));
}
</style>
