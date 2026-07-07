<script setup lang="ts">
withDefaults(defineProps<{
  ariaLabel?: string
}>(), {
  ariaLabel: '搜索和筛选',
})
</script>

<template>
  <section class="product-search-toolbar" :aria-label="ariaLabel">
    <div class="product-search-toolbar__search">
      <slot name="search" />
    </div>

    <div v-if="$slots.filters" class="product-search-toolbar__filters">
      <slot name="filters" />
    </div>
  </section>
</template>

<style scoped>
.product-search-toolbar {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 24px;
  padding: 16px;
  border: 1px solid var(--color-border-muted);
  border-radius: var(--radius-md);
  background: var(--color-surface-card);
  box-shadow: var(--card-shadow);
}

.product-search-toolbar__search {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.product-search-toolbar__filters {
  min-width: 0;
}

@media (--breakpoint-sm-down) {
  .product-search-toolbar__search {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
