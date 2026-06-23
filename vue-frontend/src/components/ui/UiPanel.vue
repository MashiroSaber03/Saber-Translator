<script setup lang="ts">
withDefaults(defineProps<{
  padded?: boolean
  variant?: 'default' | 'settings'
}>(), {
  padded: true,
  variant: 'default',
})
</script>

<template>
  <section class="ui-panel" :class="[`ui-panel--${variant}`, { 'ui-panel--padded': padded }]">
    <div v-if="$slots.title" class="ui-panel__title">
      <slot name="title" />
    </div>
    <slot />
  </section>
</template>

<style scoped>
.ui-panel {
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--color-surface-card);
  box-shadow: var(--shadow-sm, 0 1px 3px var(--ui-panel-subtle-shadow-color));
}

.ui-panel--padded {
  padding: 16px;
}

.ui-panel--settings {
  margin-bottom: 25px;
  background-color: var(--color-surface-input, var(--ui-panel-settings-background));
  border-color: var(--color-border-muted, var(--color-border-muted));
  border-radius: 10px;
  box-shadow: none;
}

.ui-panel--settings:last-child {
  margin-bottom: 0;
}

.ui-panel--settings.ui-panel--padded {
  padding: 20px;
}

.ui-panel__title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-muted));
  color: var(--color-action-primary, var(--color-text-link));
  font-size: 1.1em;
  font-weight: 600;
}

.ui-panel :slotted(.ui-form-hint) {
  margin-top: 6px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 0.85em;
  line-height: 1.45;
}
</style>
