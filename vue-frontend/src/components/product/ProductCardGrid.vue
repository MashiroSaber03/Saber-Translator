<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  gap?: string
  minItemWidth?: string
}>(), {
  ariaLabel: '卡片列表',
  gap: '24px',
  minItemWidth: '160px',
})

const gridStyle = computed(() => ({
  '--product-card-grid-gap': props.gap,
  '--product-card-grid-min-item-width': props.minItemWidth,
}))
</script>

<template>
  <div
    class="product-card-grid"
    role="group"
    :aria-label="ariaLabel"
    :style="gridStyle"
  >
    <slot />
  </div>
</template>

<style scoped>
.product-card-grid {
  --product-card-grid-gap: 24px;
  --product-card-grid-min-item-width: 160px;

  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--product-card-grid-min-item-width)), 1fr));
  gap: var(--product-card-grid-gap);
  align-items: start;
}
</style>
