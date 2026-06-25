<script setup lang="ts">
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import type { ExtractResult, WebImportState } from '@/types/webImport'

defineProps<{
  downloadProgress: { current: number; total: number }
  downloadProgressPercent: number
  engineDisplayName: string
  error: string | null
  extractResult: ExtractResult | null
  isAllSelected: boolean
  selectedCount: number
  selectedPages: Set<number>
  status: WebImportState['status']
  previewUrlFor: (url: string) => string
}>()

defineEmits<{
  (event: 'toggleAll'): void
  (event: 'togglePage', pageNum: number): void
}>()
</script>

<template>
  <div class="web-import-results-grid">
    <div v-if="error" class="error-section">
      <span class="error-icon">❌</span>
      <span class="error-message">{{ error }}</span>
    </div>

    <div v-if="extractResult?.success" class="result-section">
      <div class="result-header">
        <span class="result-title">
          📖 《{{ extractResult.comicTitle }}》- {{ extractResult.chapterTitle }}
        </span>
        <span class="result-meta">
          <span class="result-count">共 {{ extractResult.totalPages }} 张</span>
          <span v-if="engineDisplayName" class="result-engine">| 引擎: {{ engineDisplayName }}</span>
        </span>
      </div>

      <div class="select-control">
        <UiCheckbox :model-value="isAllSelected" label="全选" @change="$emit('toggleAll')" />
        <span class="selected-count">已选: {{ selectedCount }} 张</span>
      </div>

      <div class="image-grid">
        <label
          v-for="page in extractResult.pages"
          :key="page.pageNumber"
          class="image-item"
          :class="{ selected: selectedPages.has(page.pageNumber) }"
        >
          <div class="image-checkbox">
            <UiCheckbox
              :aria-label="`选择第 ${page.pageNumber} 页`"
              :model-value="selectedPages.has(page.pageNumber)"
              @change="$emit('togglePage', page.pageNumber)"
            />
          </div>
          <div class="image-preview">
            <img :src="previewUrlFor(page.imageUrl)" :alt="`第${page.pageNumber}页`" loading="lazy">
          </div>
          <div class="image-label">第 {{ page.pageNumber }} 页</div>
        </label>
      </div>
    </div>

    <div v-if="status === 'downloading'" class="progress-section">
      <div class="progress-label">
        下载进度: {{ downloadProgress.current }}/{{ downloadProgress.total }}
      </div>
      <div
        class="progress-bar"
        role="progressbar"
        aria-label="网页导入下载进度"
        aria-valuemin="0"
        :aria-valuemax="downloadProgress.total"
        :aria-valuenow="downloadProgress.current"
      >
        <div class="progress-fill" :style="{ width: `${downloadProgressPercent}%` }"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.web-import-results-grid {
  --web-import-results-error-border: #ffc0c0;
  --web-import-results-error-text: #c00;
  --web-import-results-selected-shadow: rgba(74, 144, 217, .2);
  --web-import-results-progress-track: #eee;
}

.error-section {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  padding: 12px 14px;
  border: 1px solid var(--web-import-results-error-border);
  border-radius: 6px;
  background: var(--color-surface-neutral-soft);
  color: var(--web-import-results-error-text);
}

.result-section {
  margin-bottom: 16px;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.result-title {
  color: var(--color-text-default);
  font-weight: 500;
  font-size: 15px;
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.result-count {
  color: var(--color-text-supporting);
  font-size: 13px;
}

.result-engine {
  color: var(--color-text-supporting);
  font-size: 12px;
}

.select-control {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
}

.selected-count {
  color: var(--color-text-supporting);
  font-size: 13px;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
  max-height: 300px;
  padding: 4px;
  overflow-y: auto;
}

.image-item {
  position: relative;
  overflow: hidden;
  border: 2px solid var(--color-border-muted);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.image-item:hover,
.image-item.selected {
  border-color: var(--color-action-primary);
}

.image-item.selected {
  box-shadow: 0 0 0 2px var(--web-import-results-selected-shadow);
}

.image-checkbox {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: var(--z-local);
}

.image-preview {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  overflow: hidden;
  background: var(--color-surface-subtle);
  aspect-ratio: 3/4;
}

.image-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-label {
  padding: 6px;
  background: var(--color-surface-base);
  color: var(--color-text-supporting);
  font-size: 12px;
  text-align: center;
}

.progress-section {
  margin-bottom: 16px;
}

.progress-label {
  margin-bottom: 8px;
  color: var(--color-text-supporting);
  font-size: 13px;
}

.progress-bar {
  height: 8px;
  overflow: hidden;
  border-radius: 4px;
  background: var(--web-import-results-progress-track);
}

.progress-fill {
  height: 100%;
  background: var(--color-action-primary);
  transition: width 0.3s ease;
}
</style>
