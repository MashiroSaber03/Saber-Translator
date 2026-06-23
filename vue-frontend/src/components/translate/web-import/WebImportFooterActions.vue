<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
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
  <UiButton
    variant="toolbar"
    class="cancel-btn"
    :disabled="status === 'downloading'"
    @click="$emit('close')"
  >
    取消
  </UiButton>
  <UiButton
    variant="toolbar"
    class="import-btn"
    :disabled="!extractResult?.success || selectedCount === 0 || isProcessing"
    @click="$emit('import')"
  >
    <span v-if="status === 'downloading'" class="loading-spinner"></span>
    <span v-else>📥</span>
    {{ status === 'downloading' ? '下载中...' : '导入' }}
  </UiButton>
</template>

<style scoped>
.cancel-btn,
.import-btn {
  --web-import-footer-cancel-surface: #f0f0f0;
  --web-import-footer-cancel-surface-hover: #e5e5e5;
  --web-import-footer-import-surface: #4a90d9;
  --web-import-footer-import-surface-hover: #3a7fc8;

  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.cancel-btn {
  border: 1px solid var(--color-border-muted);
  background: var(--web-import-footer-cancel-surface);
  color: var(--color-text-default);
}

.cancel-btn:hover:not(:disabled) {
  background: var(--web-import-footer-cancel-surface-hover);
}

.import-btn {
  border: none;
  background: var(--web-import-footer-import-surface);
  color: var(--color-text-inverse);
}

.import-btn:hover:not(:disabled) {
  background: var(--web-import-footer-import-surface-hover);
}

.import-btn:disabled,
.cancel-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid transparent;
  border-top-color: currentcolor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
</style>
