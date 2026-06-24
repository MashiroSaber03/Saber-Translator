<template>
  <div class="export-panel">
    <h3>📦 导出成品</h3>

    <div class="export-options">
      <div class="export-summary">
        <p>共生成 <strong>{{ generatedCount }}</strong> 页图片，可导出为以下格式：</p>
      </div>

      <div class="export-formats">
        <UiButton
          variant="toolbar"
          type="button"
          class="format-card"
          :class="{ selected: selectedFormat === 'images' }"
          :aria-pressed="String(selectedFormat === 'images')"
          @click="selectedFormat = 'images'"
        >
          <span class="format-icon">🖼️</span>
          <span class="format-name">图片 ZIP</span>
          <span class="format-desc">所有页面打包下载</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          type="button"
          class="format-card"
          :class="{ selected: selectedFormat === 'pdf' }"
          :aria-pressed="String(selectedFormat === 'pdf')"
          @click="selectedFormat = 'pdf'"
        >
          <span class="format-icon">📄</span>
          <span class="format-name">PDF 文档</span>
          <span class="format-desc">方便阅读和分享</span>
        </UiButton>
      </div>

      <UiButton
        variant="primary"
        class="export-download-action"
        block
        :disabled="isExporting"
        size="lg"
        @click="handleExport"
      >
        {{ isExporting ? '导出中...' : '📥 下载' }}
      </UiButton>

      <div class="export-actions">
        <UiButton variant="secondary" @click="clearAndRestart">🗑️ 清空并重新开始</UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import { ref } from 'vue'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'
import * as continuationApi from '@/api/continuation'

const props = defineProps<{
  bookId: string
  generatedCount: number
  state: ContinuationState
}>()

const emit = defineEmits<{
  'clear-and-restart': []
}>()

const state = props.state
const selectedFormat = ref<'images' | 'pdf'>('images')
const isExporting = ref(false)

async function handleExport() {
  if (!props.bookId || state.pages.value.length === 0) {
    state.showMessage('没有可导出的页面', 'error')
    return
  }

  isExporting.value = true

  try {
    let blob: Blob
    let filename: string

    if (selectedFormat.value === 'images') {
      blob = await continuationApi.exportAsImages(props.bookId)
      filename = `continuation_${Date.now()}.zip`
    } else {
      blob = await continuationApi.exportAsPdf(props.bookId)
      filename = `continuation_${Date.now()}.pdf`
    }

    const url = window.URL.createObjectURL(blob)
    try {
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
    } finally {
      window.URL.revokeObjectURL(url)
    }

    state.showMessage('导出成功', 'success')
  } catch (error) {
    state.showMessage('导出失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isExporting.value = false
  }
}

async function clearAndRestart() {
  if (!confirm('确定要清空所有续写数据并重新开始吗？此操作不可恢复。')) {
    return
  }

  emit('clear-and-restart')
}
</script>

<style scoped>
.export-panel {
  --export-panel-selected-format-background: rgba(99, 102, 241, .05);
  --ui-button-padding: 10px 20px;
  --ui-button-radius: 8px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-secondary-background: var(--color-surface-muted);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: 1px solid var(--color-border-muted, var(--color-border-default));
  --ui-button-secondary-hover-background: var(--color-surface-hover);
  --ui-button-disabled-opacity: 0.5;

  padding: 24px;
}

.export-panel h3 {
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 600;
}

.export-options {
  max-width: 600px;
  margin: 0 auto;
}

.export-summary {
  margin-bottom: 24px;
  text-align: center;
}

.export-summary p {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.export-summary strong {
  color: var(--color-text-brand);
  font-size: 20px;
}

.export-formats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.format-card {
  display: block;
  width: 100%;
  padding: 24px;
  border: 2px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}

.format-card:hover {
  border-color: var(--color-border-brand);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--color-focus-brand-soft);
}

.format-card.selected {
  border-color: var(--color-border-brand);
  background: var(--export-panel-selected-format-background);
}

.format-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 12px;
}

.format-name {
  display: block;
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
}

.format-desc {
  display: block;
  font-size: 14px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.export-download-action {
  margin-bottom: 16px;
}

.export-actions {
  text-align: center;
}
</style>
