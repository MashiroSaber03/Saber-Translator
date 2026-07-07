<script setup lang="ts">
withDefaults(defineProps<{
  ariaLabel?: string
  role?: 'list' | 'region' | 'toolbar'
}>(), {
  ariaLabel: undefined,
  role: 'region',
})
</script>

<template>
  <div
    class="product-horizontal-scroll-strip"
    :role="role"
    :aria-label="ariaLabel"
  >
    <slot />
  </div>
</template>

<style scoped>
.product-horizontal-scroll-strip {
  --product-horizontal-scroll-strip-scrollbar-track: var(--color-overlay-inverse-subtle);
  --product-horizontal-scroll-strip-scrollbar-thumb: var(--color-overlay-inverse-prominent);
  --product-horizontal-scroll-strip-padding: 0;

  display: flex;
  min-width: 0;
  max-width: 100%;
  padding: var(--product-horizontal-scroll-strip-padding);
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-color:
    var(--product-horizontal-scroll-strip-scrollbar-thumb)
    var(--product-horizontal-scroll-strip-scrollbar-track);
  scrollbar-width: thin;
}

.product-horizontal-scroll-strip::-webkit-scrollbar {
  height: 6px;
}

.product-horizontal-scroll-strip::-webkit-scrollbar-track {
  border-radius: 999px;
  background: var(--product-horizontal-scroll-strip-scrollbar-track);
}

.product-horizontal-scroll-strip::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--product-horizontal-scroll-strip-scrollbar-thumb);
}
</style>
