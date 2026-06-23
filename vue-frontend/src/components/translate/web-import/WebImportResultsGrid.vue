<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
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
      <label class="select-all">
        <UiInput
          type="checkbox"
          :checked="isAllSelected"
          @change="$emit('toggleAll')"
        />
        全选
      </label>
      <span class="selected-count">已选: {{ selectedCount }} 张</span>
    </div>

    <div class="image-grid">
      <div
        v-for="page in extractResult.pages"
        :key="page.pageNumber"
        class="image-item"
        :class="{ selected: selectedPages.has(page.pageNumber) }"
        @click="$emit('togglePage', page.pageNumber)"
      >
        <div class="image-checkbox">
          <UiInput
            type="checkbox"
            :checked="selectedPages.has(page.pageNumber)"
            @click.stop
            @change="$emit('togglePage', page.pageNumber)"
          />
        </div>
        <div class="image-preview">
          <img :src="previewUrlFor(page.imageUrl)" :alt="`第${page.pageNumber}页`" loading="lazy">
        </div>
        <div class="image-label">第 {{ page.pageNumber }} 页</div>
      </div>
    </div>
  </div>

  <div v-if="status === 'downloading'" class="progress-section">
    <div class="progress-label">
      下载进度: {{ downloadProgress.current }}/{{ downloadProgress.total }}
    </div>
    <div class="progress-bar">
      <div class="progress-fill" :style="{ width: `${downloadProgressPercent}%` }"></div>
    </div>
  </div>
</template>

<style scoped>
.error-section {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  padding: 12px 14px;
  border: 1px solid var(--web-import-modal-settings-border-default);
  border-radius: 6px;
  background: var(--color-surface-neutral-soft);
  color: var(--web-import-modal-settings-text-brand);
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

.select-all {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  cursor: pointer;
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
  box-shadow: 0 0 0 2px var(--web-import-modal-settings-shadow-default);
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
  background: var(--web-import-modal-settings-surface-muted);
}

.progress-fill {
  height: 100%;
  background: var(--color-action-primary);
  transition: width 0.3s ease;
}
</style>
