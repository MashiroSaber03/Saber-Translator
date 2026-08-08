<script setup lang="ts">
type ProductActionRowJustify = 'start' | 'center' | 'end' | 'between'
type ProductActionRowVariant = 'default' | 'dialog' | 'toolbar'
type ProductActionRowAppearance = 'default' | 'accent'

const props = withDefaults(defineProps<{
  appearance?: ProductActionRowAppearance
  ariaLabel?: string
  divider?: boolean
  justify?: ProductActionRowJustify
  variant?: ProductActionRowVariant
}>(), {
  appearance: 'default',
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
      `product-action-row--appearance-${props.appearance}`,
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

.product-action-row--appearance-accent {
  --ui-button-primary-background: linear-gradient(
    135deg,
    var(--color-text-link-strong) 0%,
    color-mix(in srgb, var(--color-text-link-strong) 65%, var(--color-action-brand)) 100%
  );
  --ui-button-primary-hover-background: linear-gradient(
    135deg,
    var(--color-text-link-strong) 0%,
    color-mix(in srgb, var(--color-text-link-strong) 65%, var(--color-action-brand)) 100%
  );
  --ui-button-primary-shadow: 0 2px 8px color-mix(in srgb, var(--color-text-link-strong) 22%, transparent);
  --ui-button-primary-hover-shadow: 0 6px 20px color-mix(in srgb, var(--color-text-link-strong) 26%, transparent);
  --ui-button-secondary-background: color-mix(in srgb, var(--color-text-heading) 7%, transparent);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: none;
  --ui-button-secondary-hover-background: color-mix(in srgb, var(--color-text-link-strong) 14%, transparent);
  --ui-button-secondary-hover-border-color: transparent;
  --ui-button-disabled-border: none;
  --ui-button-disabled-background: color-mix(in srgb, var(--color-text-heading) 7%, transparent);
  --ui-button-disabled-opacity: 0.68;
}
</style>
