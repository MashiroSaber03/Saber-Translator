<script setup lang="ts">
withDefaults(defineProps<{
  label?: string
  value?: string
  title?: string
}>(), {
  label: undefined,
  value: undefined,
  title: undefined,
})
</script>

<template>
  <span class="product-header-meta-pill" :title="title">
    <span v-if="label" class="product-header-meta-pill__label">{{ label }}</span>
    <span v-if="value" class="product-header-meta-pill__value">{{ value }}</span>
    <span v-if="$slots.actions" class="product-header-meta-pill__actions">
      <slot name="actions" />
    </span>
  </span>
</template>

<style scoped>
.product-header-meta-pill {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  min-width: 0;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 20px;
  background: var(--product-header-meta-pill-background, var(--color-surface-muted));
  color: var(--product-header-meta-pill-text, var(--color-text-default));
  font-family: var(--product-header-meta-pill-font, var(--font-mono, 'Consolas', 'Monaco', monospace));
  font-size: 0.85rem;
  line-height: 1.2;
  backdrop-filter: blur(4px);
}

.product-header-meta-pill__label {
  flex: 0 0 auto;
}

.product-header-meta-pill__value {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.product-header-meta-pill__actions {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
}
</style>
