<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import type { ExtractResult, WebImportState } from '@/types/webImport'

defineProps<{
  extractResult: ExtractResult | null
  isProcessing: boolean
  selectedCount: number
  status: WebImportState['status']
}>()

defineEmits<{
  (event: 'close'): void
  (event: 'import'): void
}>()
</script>

<template>
  <ProductActionRow variant="dialog" aria-label="网页导入操作">
    <UiButton
      variant="secondary"
      :disabled="status === 'downloading'"
      @click="$emit('close')"
    >
      取消
    </UiButton>
    <UiButton
      variant="primary"
      class="web-import-footer-actions__primary"
      :disabled="!extractResult?.success || selectedCount === 0 || isProcessing"
      @click="$emit('import')"
    >
      <UiSpinner v-if="status === 'downloading'" label="下载中" :decorative="false" />
      <span v-else aria-hidden="true">📥</span>
      {{ status === 'downloading' ? '下载中...' : '导入' }}
    </UiButton>
  </ProductActionRow>
</template>

<style scoped>
.web-import-footer-actions__primary {
  --ui-button-primary-background: var(--color-action-primary);
  --ui-button-primary-hover-background: var(--color-action-primary-hover);
  --ui-button-primary-shadow: none;
  --ui-button-primary-disabled-background: var(--color-action-primary);
  --ui-button-primary-disabled-opacity: 0.6;
}
</style>
