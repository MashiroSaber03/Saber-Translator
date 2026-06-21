<script setup lang="ts">
withDefaults(defineProps<{
  label: string
  variant?: 'plain' | 'soft' | 'primary' | 'danger'
  size?: 'sm' | 'md'
  type?: 'button' | 'submit' | 'reset'
  disabled?: boolean
}>(), {
  variant: 'soft',
  size: 'md',
  type: 'button',
  disabled: false,
})
</script>

<template>
  <button
    class="ui-icon-button"
    :class="[`ui-icon-button--${variant}`, `ui-icon-button--${size}`]"
    :type="type"
    :disabled="disabled"
    :aria-label="label"
    :title="label"
  >
    <slot />
  </button>
</template>

<style scoped>
.ui-icon-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  border: 1px solid transparent;
  border-radius: 8px;
  font: inherit;
  cursor: pointer;
  transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.ui-icon-button--md {
  width: 36px;
  height: 36px;
}

.ui-icon-button--sm {
  width: 30px;
  height: 30px;
  font-size: 0.85rem;
}

.ui-icon-button--plain {
  background: transparent;
  color: var(--color-text-supporting);
}

.ui-icon-button--soft {
  background: var(--color-surface-interactive-hover);
  border-color: var(--color-border-muted);
  color: var(--color-text-default);
}

.ui-icon-button--primary {
  background: var(--color-action-primary);
  color: white;
}

.ui-icon-button--danger {
  background: var(--ui-icon-button-surface-base);
  color: var(--color-status-error, var(--color-text-danger));
}

.ui-icon-button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.ui-icon-button:hover:not(:disabled) {
  transform: translateY(-1px);
}
</style>
