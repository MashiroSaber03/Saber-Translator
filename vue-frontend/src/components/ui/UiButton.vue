<script setup lang="ts">
import { computed, unref } from 'vue'

const props = withDefaults(defineProps<{
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'link' | 'toolbar' | 'card-action' | 'tab' | 'plain-danger'
  tone?: 'neutral' | 'primary' | 'danger' | 'success' | 'warning'
  size?: 'lg' | 'md' | 'sm' | 'xs'
  type?: 'button' | 'submit' | 'reset'
  disabled?: unknown
  loading?: boolean
  block?: boolean
  icon?: boolean
}>(), {
  variant: 'secondary',
  tone: 'neutral',
  size: 'md',
  type: 'button',
  disabled: false,
  loading: false,
  block: false,
  icon: false,
})

const isDisabled = computed(() => Boolean(unref(props.disabled)) || props.loading)
const isBareStyledVariant = computed(() => ['link', 'toolbar', 'card-action', 'tab', 'plain-danger'].includes(props.variant))
const buttonClasses = computed(() => {
  return [
    'ui-button',
    `ui-button--${props.variant}`,
    `ui-button--${props.size}`,
    `ui-button--tone-${props.tone}`,
    {
      'ui-button--bare': isBareStyledVariant.value,
      'ui-button--block': props.block,
      'ui-button--icon': props.icon,
      'ui-button--loading': props.loading,
    },
  ]
})
</script>

<template>
  <button
    :class="buttonClasses"
    :type="type"
    :disabled="isDisabled"
    :aria-busy="loading ? 'true' : undefined"
  >
    <slot />
  </button>
</template>

<style scoped>
.ui-button:not(:where(.ui-button--bare)) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  white-space: nowrap;
  user-select: none;
}

.ui-button:not(:where(.ui-button--bare)):disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ui-button:not(:where(.ui-button--bare)):active:not(:disabled) {
  transform: scale(0.97);
}

.ui-button--primary {
  background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
  color: white;
  box-shadow: 0 2px 8px var(--ui-button-shadow-default);
}

.ui-button--primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px var(--ui-button-shadow-raised);
}

.ui-button--secondary {
  background: var(--color-surface-card);
  color: var(--color-text-default);
  border: 1px solid var(--color-border-muted);
}

.ui-button--secondary:hover:not(:disabled) {
  background: var(--color-surface-interactive-hover);
  border-color: var(--color-text-supporting);
}

:where(.ui-button--bare) {
  border: 0;
  padding: 0;
  background: none;
  color: inherit;
  font: inherit;
  line-height: normal;
  cursor: pointer;
}

:where(.ui-button--bare):disabled {
  cursor: not-allowed;
}

.ui-button--ghost {
  border: 1px solid var(--color-border-studio);
  background: var(--color-surface-raised);
  color: var(--color-text-default);
}

.ui-button--ghost:hover:not(:disabled) {
  background: var(--color-surface-studio-tint);
  border-color: var(--ui-button-border-default);
}

.ui-button--danger {
  background: linear-gradient(135deg, var(--ui-button-surface-base) 0%, var(--ui-button-surface-raised) 100%);
  color: white;
  box-shadow: 0 2px 8px var(--ui-button-shadow-floating);
}

.ui-button--danger:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--ui-button-surface-muted) 0%, var(--ui-button-surface-subtle) 100%);
  box-shadow: 0 6px 20px var(--ui-button-shadow-strong);
}

.ui-button--lg:not(:where(.ui-button--bare)) {
  padding: 14px 28px;
  font-size: 1rem;
}

.ui-button--sm:not(:where(.ui-button--bare)) {
  padding: 6px 14px;
  font-size: 0.85rem;
}

.ui-button--xs:not(:where(.ui-button--bare)) {
  padding: 4px 10px;
  font-size: 0.78rem;
}

.ui-button--block:not(:where(.ui-button--bare)) {
  width: 100%;
}

.ui-button--icon:not(:where(.ui-button--bare)) {
  width: 36px;
  height: 36px;
  padding: 0;
}

.ui-button--tone-danger:not(:where(.ui-button--bare)) {
  color: var(--color-status-error, var(--color-text-danger));
}

.ui-button--tone-success:not(:where(.ui-button--bare)) {
  color: var(--status-success, var(--ui-button-text-primary));
}

.ui-button--tone-warning:not(:where(.ui-button--bare)) {
  color: var(--status-warning, var(--ui-button-text-secondary));
}

.ui-button--tone-primary:not(:where(.ui-button--bare)) {
  color: var(--color-action-primary);
}

.ui-button--loading:not(:where(.ui-button--bare)) {
  pointer-events: none;
}

</style>
