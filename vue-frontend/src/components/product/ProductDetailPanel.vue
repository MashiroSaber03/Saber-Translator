<script setup lang="ts">
withDefaults(defineProps<{
  ariaLabel?: string
}>(), {
  ariaLabel: undefined,
})
</script>

<template>
  <section class="product-detail-panel" :aria-label="ariaLabel">
    <slot />
  </section>
</template>

<style scoped>
.product-detail-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 12px;
  background: var(--color-surface-muted);
}
</style>
