<template>
  <div class="continuation-dialog-form">
    <slot />
  </div>
</template>

<style scoped>
.continuation-dialog-form {
  --ui-input-min-height: 46px;
  --ui-input-padding: 10px 12px;
  --ui-input-border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  --ui-input-radius: 8px;
  --ui-input-background: var(--color-surface-base);
  --ui-input-focus-border: var(--color-border-brand);
  --ui-input-line-height: inherit;
  --ui-textarea-min-height: 98px;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  --ui-textarea-radius: 8px;
  --ui-textarea-background: var(--color-surface-base);
  --ui-textarea-focus-border: var(--color-border-brand);
  --ui-textarea-line-height: inherit;

  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
