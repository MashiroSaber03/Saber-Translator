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
  --ui-button-padding: 0;
  --ui-button-radius: 8px;
  --ui-button-font-size: 13px;
  --ui-button-secondary-background: var(--color-text-supporting);
  --ui-button-secondary-color: var(--color-text-inverse);
  --ui-button-secondary-border: 0;
  --ui-button-secondary-hover-background: var(--color-text-secondary);
  --ui-button-secondary-hover-color: var(--color-text-inverse);
  --ui-button-disabled-background: color-mix(in srgb, var(--color-text-muted) 45%, var(--color-surface-base));
  --ui-button-disabled-color: var(--color-text-inverse);
  --ui-button-disabled-border: 0;

  gap: 10px;
  margin-top: 16px;
}

.settings-navigation-buttons__action {
  flex: 1;
  min-height: 38px;
  font-weight: 600;
}
</style>
