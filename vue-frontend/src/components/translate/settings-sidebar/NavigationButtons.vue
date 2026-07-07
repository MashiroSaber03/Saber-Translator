<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'

defineProps<{
  canGoNext: boolean
  canGoPrevious: boolean
}>()

defineEmits<{
  (event: 'next'): void
  (event: 'previous'): void
}>()
</script>

<template>
  <ProductActionRow
    class="settings-navigation-buttons"
    justify="between"
    aria-label="图片导航"
  >
    <UiButton
      class="settings-navigation-buttons__action"
      variant="secondary"
      block
      :disabled="!canGoPrevious"
      @click="$emit('previous')"
    >
      上一张
    </UiButton>
    <UiButton
      class="settings-navigation-buttons__action"
      variant="secondary"
      block
      :disabled="!canGoNext"
      @click="$emit('next')"
    >
      下一张
    </UiButton>
  </ProductActionRow>
</template>

<style scoped>
.settings-navigation-buttons {
  margin-top: 16px;
}

.settings-navigation-buttons__action {
  flex: 1;
}
</style>
