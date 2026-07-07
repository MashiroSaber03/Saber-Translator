<script setup lang="ts">
defineProps<{
  description: string
}>()
</script>

<template>
  <section class="insight-settings-panel">
    <p class="insight-settings-panel__hint">{{ description }}</p>
    <slot />
  </section>
</template>

<style scoped>
.insight-settings-panel {
  padding: 16px 0;
  min-height: 300px;
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
