<script setup lang="ts">
type ProductActionRowJustify = 'start' | 'center' | 'end' | 'between'
type ProductActionRowVariant = 'default' | 'dialog' | 'toolbar'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  divider?: boolean
  justify?: ProductActionRowJustify
  variant?: ProductActionRowVariant
}>(), {
  ariaLabel: undefined,
  divider: false,
  justify: 'end',
  variant: 'default',
})
</script>

<template>
  <div
    class="product-action-row"
    :class="[
      `product-action-row--${props.justify}`,
      `product-action-row--${props.variant}`,
      { 'product-action-row--divider': props.divider },
    ]"
    role="group"
    :aria-label="props.ariaLabel"
  >
    <slot />
  </div>
</template>

<style scoped>
.product-action-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}

.product-action-row--divider {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
}

.product-action-row--start {
  justify-content: flex-start;
}

.product-action-row--center {
  justify-content: center;
}

.product-action-row--end {
  justify-content: flex-end;
}

.product-action-row--between {
  justify-content: space-between;
}

.product-action-row--dialog {
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-secondary-hover-border-color: var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-color: var(--color-text-default);
}

.product-action-row--toolbar {
  --ui-button-font-size: 12px;
  --ui-button-sm-padding: 8px 12px;
  --ui-button-xs-padding: 6px 10px;
  --ui-button-xs-font-size: 12px;
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: none;
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-secondary-hover-color: var(--color-text-default);
}
</style>
