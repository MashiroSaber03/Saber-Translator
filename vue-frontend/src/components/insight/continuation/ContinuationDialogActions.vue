<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
</script>

<template>
  <ProductActionRow aria-label="弹窗操作" justify="end" variant="dialog">
    <slot />
  </ProductActionRow>
</template>
