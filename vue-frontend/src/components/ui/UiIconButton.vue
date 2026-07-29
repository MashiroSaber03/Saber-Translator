<script setup lang="ts">
withDefaults(defineProps<{
  label: string
  title?: string
  variant?: 'plain' | 'soft' | 'primary' | 'danger' | 'inverse'
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  shape?: 'rounded' | 'circle'
  elevated?: boolean
  active?: boolean
  pressed?: boolean
  type?: 'button' | 'submit' | 'reset'
  disabled?: boolean
}>(), {
  variant: 'soft',
  size: 'md',
  shape: 'rounded',
  elevated: false,
  active: false,
  pressed: undefined,
  type: 'button',
  disabled: false,
})
</script>

<template>
  <button
    class="ui-icon-button"
    :class="[
      `ui-icon-button--${variant}`,
      `ui-icon-button--${size}`,
      `ui-icon-button--${shape}`,
      {
        'ui-icon-button--active': active,
        'ui-icon-button--elevated': elevated,
      },
    ]"
    :type="type"
    :disabled="disabled"
    :aria-label="label"
    :aria-pressed="pressed"
    :title="title || label"
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

.ui-icon-button--lg {
  width: 44px;
  height: 44px;
}

.ui-icon-button--xl {
  width: 48px;
  height: 48px;
}

.ui-icon-button--sm {
  width: 30px;
  height: 30px;
  font-size: 0.85rem;
}

.ui-icon-button--xs {
  width: 24px;
  height: 24px;
  font-size: 0.78rem;
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
  color: var(--color-text-inverse);
}

.ui-icon-button--danger {
  background: var(--ui-icon-button-danger-soft-background);
  color: var(--color-status-error, var(--color-text-danger));
}

.ui-icon-button--inverse {
  background: var(--color-overlay-inverse-subtle);
  border-color: transparent;
  color: var(--color-text-inverse);
}

.ui-icon-button--circle {
  border-radius: 50%;
}

.ui-icon-button--elevated.ui-icon-button--primary {
  box-shadow: 0 4px 12px var(--shadow-action-brand);
}

.ui-icon-button--active {
  border-color: var(--ui-icon-button-active-border, var(--color-action-primary));
  background: var(--ui-icon-button-active-background, var(--color-action-primary));
  color: var(--ui-icon-button-active-color, var(--color-text-inverse));
}

.ui-icon-button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.ui-icon-button:hover:not(:disabled) {
  transform: translateY(-1px);
}

.ui-icon-button--inverse:hover:not(:disabled) {
  background: var(--color-overlay-inverse-muted);
}

.ui-icon-button--active:hover:not(:disabled) {
  background: var(--ui-icon-button-active-hover-background, var(--color-action-primary-hover));
  border-color: var(--ui-icon-button-active-hover-border, var(--color-action-primary-hover));
  color: var(--ui-icon-button-active-color, var(--color-text-inverse));
}

.ui-icon-button--elevated.ui-icon-button--primary:hover:not(:disabled) {
  box-shadow: 0 6px 16px var(--shadow-action-brand);
}
</style>
