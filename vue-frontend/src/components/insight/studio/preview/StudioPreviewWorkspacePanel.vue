<template>
  <section class="studio-preview-workspace-panel">
    <slot />
  </section>
</template>

<style scoped>
.studio-preview-workspace-panel {
  --studio-preview-workspace-panel-background: color-mix(in srgb, var(--color-surface-card) 92%, transparent);
  --studio-preview-workspace-panel-border: var(--studio-border-default);
  --studio-preview-workspace-panel-shadow: var(--studio-shadow-floating);

  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  width: 100%;
  min-height: 0;
  padding: 14px;
  border: 1px solid var(--studio-preview-workspace-panel-border);
  border-radius: 24px;
  background: var(--studio-preview-workspace-panel-background);
  box-shadow: 0 24px 40px var(--studio-preview-workspace-panel-shadow);
}
</style>
