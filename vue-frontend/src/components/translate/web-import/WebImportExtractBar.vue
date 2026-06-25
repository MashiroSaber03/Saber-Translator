<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { WebImportEngine, WebImportState } from '@/types/webImport'

defineProps<{
  checkingSupport: boolean
  galleryDLAvailable: boolean
  galleryDLSupported: boolean
  isProcessing: boolean
  selectedEngine: WebImportEngine
  status: WebImportState['status']
  urlInput: string
}>()

defineEmits<{
  (event: 'extract'): void
  (event: 'update:selectedEngine', value: WebImportEngine): void
  (event: 'update:urlInput', value: string): void
}>()
</script>

<template>
  <div class="web-import-extract-bar">
    <div class="url-section">
      <UiInput
        :model-value="urlInput"
        type="url"
        class="url-input"
        placeholder="输入漫画网页 URL，如 https://example.com/chapter-1"
        :disabled="isProcessing"
        @update:model-value="$emit('update:urlInput', String($event))"
        @keyup.enter="$emit('extract')"
      />
      <UiSelect
        :model-value="selectedEngine"
        class="engine-select"
        :disabled="isProcessing"
        @update:model-value="$emit('update:selectedEngine', $event as WebImportEngine)"
      >
        <option value="auto">自动选择</option>
        <option value="gallery-dl">Gallery-DL</option>
        <option value="ai-agent">AI Agent</option>
      </UiSelect>
      <UiButton
        variant="toolbar"
        class="extract-btn"
        :disabled="isProcessing || !urlInput.trim()"
        @click="$emit('extract')"
      >
        <span v-if="status === 'extracting'" class="loading-spinner"></span>
        <span v-else>🔍</span>
        {{ status === 'extracting' ? '提取中...' : '开始提取' }}
      </UiButton>
    </div>

    <div v-if="urlInput.trim() && !isProcessing" class="engine-hint">
      <span v-if="checkingSupport" class="hint-checking">检查中...</span>
      <span v-else-if="galleryDLSupported" class="hint-supported">✓ 该网站支持 Gallery-DL 高速下载</span>
      <span v-else-if="galleryDLAvailable" class="hint-unsupported">该网站将使用 AI Agent 模式</span>
    </div>

    <div class="notice">
      ⚠️ 请仅爬取您有权访问的内容，并遵守目标网站的使用条款。
    </div>
  </div>
</template>

<style scoped>
.web-import-extract-bar {
  --web-import-extract-action-background: #4a90d9;
  --web-import-extract-action-hover-background: #3a7fc8;
  --web-import-extract-supported-text: #28a745;
  --web-import-extract-notice-border: #ffe0a0;
  --web-import-extract-notice-text: #856404;
}

.url-section {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.url-input {
  flex: 1 1 auto;
  min-width: 0;
  padding: 10px 14px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.url-input:focus {
  border-color: var(--color-action-primary);
}

.engine-select {
  flex: 0 0 120px;
  width: 120px;
  min-width: 120px;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background: var(--color-surface-base);
  font-size: 14px;
  outline: none;
  cursor: pointer;
}

.engine-select:focus {
  border-color: var(--color-action-primary);
}

.engine-select:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.extract-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 18px;
  border: none;
  border-radius: 8px;
  background: var(--web-import-extract-action-background);
  color: var(--color-text-inverse);
  font-weight: 500;
  font-size: 14px;
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.2s;
}

.extract-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.extract-btn:hover:not(:disabled) {
  background: var(--web-import-extract-action-hover-background);
}

.engine-hint {
  margin-bottom: 12px;
  padding: 0 2px;
  font-size: 12px;
}

.hint-checking,
.hint-unsupported {
  color: var(--color-text-supporting);
}

.hint-supported {
  color: var(--web-import-extract-supported-text);
}

.notice {
  margin-bottom: 16px;
  padding: 10px 14px;
  border: 1px solid var(--web-import-extract-notice-border);
  border-radius: 6px;
  background: var(--color-status-warning-surface-soft);
  color: var(--web-import-extract-notice-text);
  font-size: 13px;
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
