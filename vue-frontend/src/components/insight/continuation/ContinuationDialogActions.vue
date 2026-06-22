<template>
  <div class="continuation-dialog-actions">
    <slot />
  </div>
</template>

<style scoped>
.continuation-dialog-actions {
  --ui-button-padding: 10px 20px;
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

  display: flex;
  justify-content: flex-end;
  gap: 12px;
}
</style>
