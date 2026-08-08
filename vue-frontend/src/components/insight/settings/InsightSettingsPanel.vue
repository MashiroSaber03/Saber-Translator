<script setup lang="ts">
defineProps<{
  description: string
}>()
</script>

<template>
  <section class="insight-settings-panel">
    <p class="insight-settings-panel__hint">{{ description }}</p>
    <div class="insight-settings-panel__content">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.insight-settings-panel {
  padding: 28px 0 16px;
  min-height: 300px;
}

.insight-settings-panel__content {
  --ui-input-sm-min-height: 38px;
  --ui-input-sm-padding: 9px 12px;

  display: flex;
  flex-direction: column;
  min-width: 0;
}

.insight-settings-panel__hint {
  margin: 0 0 16px;
  padding: 8px 12px;
  border-radius: 4px;
  background: var(--color-surface-muted);
  color: var(--color-text-supporting);
  font-size: 13px;
  line-height: 1.45;
}
</style>
