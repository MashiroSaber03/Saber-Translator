<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
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
      :disabled="!extractResult?.success || selectedCount === 0 || isProcessing"
      @click="$emit('import')"
    >
      <UiSpinner v-if="status === 'downloading'" label="下载中" :decorative="false" />
      <UiIcon v-else name="download" />
      {{ status === 'downloading' ? '下载中...' : '导入' }}
    </UiButton>
  </ProductActionRow>
</template>
