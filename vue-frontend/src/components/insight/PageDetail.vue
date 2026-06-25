<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'

import { ref, computed, onUnmounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { PageAnalysisData } from '@/api/insight'

const insightStore = useInsightStore()

const pageAnalysis = ref<PageAnalysisData | null>(null)
const isLoading = ref(false)
const isReanalyzing = ref(false)
const pendingReanalyzePage = ref<number | null>(null)
const showImagePreview = ref(false)
const errorMessage = ref('')
let pageDetailRequestSequence = 0
let isPageDetailMounted = true

const selectedPageNum = computed(() => insightStore.selectedPageNum)
const totalPages = computed(() => insightStore.totalPageCount)

const hasPrevPage = computed(() => {
  return selectedPageNum.value !== null && selectedPageNum.value > 1
})

const hasNextPage = computed(() => {
  return selectedPageNum.value !== null && selectedPageNum.value < totalPages.value
})

const pageImageUrl = computed(() => {
  if (!insightStore.currentBookId || !selectedPageNum.value) return ''
  return insightApi.getPageImageUrl(insightStore.currentBookId, selectedPageNum.value)
})

const dialogues = computed(() => {
  if (!pageAnalysis.value?.panels) return []
  const result: Array<{ speaker: string; text: string; originalText?: string }> = []
  for (const panel of pageAnalysis.value.panels) {
    if (panel.dialogues) {
      for (const d of panel.dialogues) {
        const text = d.translated_text || d.text
        if (text) {
          result.push({
            speaker: d.speaker_name || d.character || '未知',
            text: text,
            originalText: d.text !== d.translated_text ? d.text : undefined
          })
        }
      }
    }
  }
  return result
})

const isPageAnalyzed = computed(() => {
  return pageAnalysis.value?.analyzed === true || !!pageAnalysis.value?.page_summary
})

const sceneDescription = computed(() => pageAnalysis.value?.scene || '')
const moodDescription = computed(() => pageAnalysis.value?.mood || '')

const isReanalyzeTaskRunning = computed(() => {
  return (
    pendingReanalyzePage.value !== null &&
    pendingReanalyzePage.value === selectedPageNum.value &&
    insightStore.analysisStatus === 'running'
  )
})

async function loadPageDetail(): Promise<void> {
  const bookId = insightStore.currentBookId
  const pageNum = selectedPageNum.value
  const requestId = ++pageDetailRequestSequence

  if (!bookId || !pageNum) {
    pageAnalysis.value = null
    errorMessage.value = ''
    isLoading.value = false
    return
  }

  isLoading.value = true
  errorMessage.value = ''

  try {
    const response = await insightApi.getPageData(bookId, pageNum)

    if (!isCurrentPageDetailRequest(requestId, bookId, pageNum)) return

    if (response.success) {
      pageAnalysis.value = response.analysis ?? response.page ?? null
    } else {
      pageAnalysis.value = null
      if (response.error) {
        errorMessage.value = response.error
      }
    }
  } catch (error) {
    if (!isCurrentPageDetailRequest(requestId, bookId, pageNum)) return
    pageAnalysis.value = null
    errorMessage.value = error instanceof Error ? error.message : '加载失败'
  } finally {
    if (isCurrentPageDetailRequest(requestId, bookId, pageNum)) {
      isLoading.value = false
    }
  }
}

function isCurrentPageDetailRequest(requestId: number, bookId: string, pageNum: number): boolean {
  return (
    isPageDetailMounted &&
    requestId === pageDetailRequestSequence &&
    insightStore.currentBookId === bookId &&
    selectedPageNum.value === pageNum
  )
}

function navigatePrev(): void {
  if (hasPrevPage.value && selectedPageNum.value) {
    insightStore.selectPage(selectedPageNum.value - 1)
  }
}

function navigateNext(): void {
  if (hasNextPage.value && selectedPageNum.value) {
    insightStore.selectPage(selectedPageNum.value + 1)
  }
}

async function reanalyzePage(): Promise<void> {
  if (!insightStore.currentBookId || !selectedPageNum.value) return

  isReanalyzing.value = true
  errorMessage.value = ''

  try {
    const response = await insightApi.reanalyzePage(
      insightStore.currentBookId,
      selectedPageNum.value
    )

    if (response.success) {
      if (response.task_id) {
        insightStore.setCurrentTaskId(response.task_id)
      }
      pendingReanalyzePage.value = selectedPageNum.value
      insightStore.setAnalysisStatus('running')
    } else {
      errorMessage.value = response.error || '重新分析失败'
    }
  } catch (error) {
    const message = (error as { message?: string })?.message
    errorMessage.value = message || '重新分析失败'
  } finally {
    isReanalyzing.value = false
  }
}

function openImagePreview(): void {
  showImagePreview.value = true
}

function closeImagePreview(): void {
  showImagePreview.value = false
}

function handlePreviewKeydown(event: KeyboardEvent): void {
  if (!showImagePreview.value) return

  switch (event.key) {
    case 'Escape':
      closeImagePreview()
      break
    case 'ArrowLeft':
      if (hasPrevPage.value) {
        navigatePrev()
      }
      break
    case 'ArrowRight':
      if (hasNextPage.value) {
        navigateNext()
      }
      break
  }
}

const isExporting = ref(false)

async function exportPageData(): Promise<void> {
  if (!insightStore.currentBookId || !selectedPageNum.value || !pageAnalysis.value) {
    return
  }

  isExporting.value = true

  try {
    let markdown = `# 第 ${selectedPageNum.value} 页分析数据\n\n`

    if (pageAnalysis.value.page_summary) {
      markdown += `## 📝 页面摘要\n\n${pageAnalysis.value.page_summary}\n\n`
    }

    if (pageAnalysis.value.scene) {
      markdown += `## 🎬 场景\n\n${pageAnalysis.value.scene}\n\n`
    }
    if (pageAnalysis.value.mood) {
      markdown += `## 🎭 氛围\n\n${pageAnalysis.value.mood}\n\n`
    }

    if (dialogues.value.length > 0) {
      markdown += `## 💬 对话内容\n\n`
      for (const d of dialogues.value) {
        markdown += `**${d.speaker}**: ${d.text}\n\n`
        if (d.originalText) {
          markdown += `> 原文: ${d.originalText}\n\n`
        }
      }
    }

    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    try {
      const a = document.createElement('a')
      a.href = url
      a.download = `${insightStore.currentBookId}_page_${selectedPageNum.value}.md`
      a.click()
    } finally {
      URL.revokeObjectURL(url)
    }

  } catch {
    errorMessage.value = '导出失败'
  } finally {
    isExporting.value = false
  }
}

watch(selectedPageNum, () => {
  loadPageDetail()
}, { immediate: true })

watch(() => insightStore.dataRefreshKey, async (newKey) => {
  if (newKey <= 0 || !selectedPageNum.value) return

  if (pendingReanalyzePage.value !== null) {
    pendingReanalyzePage.value = null
  }
  await loadPageDetail()
})

onUnmounted(() => {
  isPageDetailMounted = false
  pageDetailRequestSequence += 1
})
</script>

<template>
  <div class="workspace-section page-detail-section">
    <h3 class="section-title">📄 页面详情</h3>

    <div class="page-detail">
      <div v-if="!selectedPageNum" class="placeholder-text">
        <div class="empty-icon">📄</div>
        <p>点击左侧导航树中的页面查看详情</p>
      </div>

      <div v-else-if="isLoading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载中...</p>
      </div>

      <div v-else class="page-detail-content">
        <div class="page-detail-header">
          <h4>📄 第 {{ selectedPageNum }} 页</h4>
          <div class="page-nav-buttons">
            <UiButton
              variant="toolbar"
              class="btn-page-nav"
              :class="{ disabled: !hasPrevPage }"
              :disabled="!hasPrevPage"
              title="上一页 (←)"
              @click="navigatePrev"
            >
              ◀ 上一张
            </UiButton>
            <span class="page-indicator">{{ selectedPageNum }} / {{ totalPages }}</span>
            <UiButton
              variant="toolbar"
              class="btn-page-nav"
              :class="{ disabled: !hasNextPage }"
              :disabled="!hasNextPage"
              title="下一页 (→)"
              @click="navigateNext"
            >
              下一张 ▶
            </UiButton>
          </div>
        </div>

        <div v-if="errorMessage" class="error-message">
          ⚠️ {{ errorMessage }}
        </div>

        <UiButton
          variant="toolbar"
          class="page-detail-image"
          :aria-label="`预览第 ${selectedPageNum} 页图片`"
          @click="openImagePreview"
        >
          <img
            :src="pageImageUrl"
            :alt="`第${selectedPageNum}页`"
            @error="($event.target as HTMLImageElement).style.display = 'none'"
          >
          <div class="image-overlay">
            <span class="zoom-hint">🔍 点击放大</span>
          </div>
        </UiButton>

        <div class="analysis-status-tag" :class="{ analyzed: isPageAnalyzed }">
          {{ isPageAnalyzed ? '✓ 已分析' : '○ 未分析' }}
        </div>

        <div v-if="pageAnalysis?.page_summary" class="page-summary">
          <h5>📝 页面摘要</h5>
          <p>{{ pageAnalysis.page_summary }}</p>
        </div>
        <div v-else class="page-summary empty">
          <p>此页尚未分析，点击下方按钮开始分析</p>
        </div>

        <div v-if="sceneDescription || moodDescription" class="scene-mood-info">
          <div v-if="sceneDescription" class="info-item">
            <span class="info-label">🎬 场景：</span>
            <span class="info-value">{{ sceneDescription }}</span>
          </div>
          <div v-if="moodDescription" class="info-item">
            <span class="info-label">🎭 氛围：</span>
            <span class="info-value">{{ moodDescription }}</span>
          </div>
        </div>

        <div v-if="dialogues.length > 0" class="dialogues-section">
          <h5>💬 对话内容 ({{ dialogues.length }})</h5>
          <div
            v-for="(dialogue, index) in dialogues"
            :key="index"
            class="dialogue-item"
          >
            <div class="dialogue-speaker">
              <span class="speaker-icon">👤</span>
              {{ dialogue.speaker }}
            </div>
            <div class="dialogue-text">{{ dialogue.text }}</div>
            <div v-if="dialogue.originalText" class="dialogue-original">
              <span class="original-label">原文：</span>{{ dialogue.originalText }}
            </div>
          </div>
        </div>
        <div v-else-if="isPageAnalyzed" class="dialogues-section empty">
          <p>此页没有检测到对话内容</p>
        </div>

        <div class="page-detail-actions">
          <UiButton
            variant="secondary"
            size="sm"
            :disabled="isReanalyzing || isReanalyzeTaskRunning"
            :loading="isReanalyzing || isReanalyzeTaskRunning"
            @click="reanalyzePage"
          >
            <span v-if="isReanalyzing || isReanalyzeTaskRunning" class="btn-spinner"></span>
            {{ isReanalyzing ? '启动中...' : (isReanalyzeTaskRunning ? '分析中...' : '🔄 重新分析') }}
          </UiButton>
          <UiButton
            v-if="isPageAnalyzed"
            variant="secondary"
            size="sm"
            :disabled="isExporting"
            @click="exportPageData"
          >
            {{ isExporting ? '导出中...' : '📄 导出此页' }}
          </UiButton>
        </div>
      </div>
    </div>

    <OverlayLayer
      v-if="showImagePreview"
      class="image-preview-modal"
      tabindex="0"
      @backdrop="closeImagePreview"
      @keydown="handlePreviewKeydown"
    >
      <div class="image-preview-content" @click.stop>
        <UiButton variant="toolbar" class="preview-close" title="关闭 (Esc)" @click="closeImagePreview">&times;</UiButton>
        <img :src="pageImageUrl" :alt="`第${selectedPageNum}页`">
        <div class="preview-nav">
          <UiButton
            variant="toolbar"
            class="preview-nav-btn prev"
            :disabled="!hasPrevPage"
            title="上一页 (←)"
            @click.stop="navigatePrev"
          >
            ◀
          </UiButton>
          <span class="preview-page-info">{{ selectedPageNum }} / {{ totalPages }}</span>
          <UiButton
            variant="toolbar"
            class="preview-nav-btn next"
            :disabled="!hasNextPage"
            title="下一页 (→)"
            @click.stop="navigateNext"
          >
            ▶
          </UiButton>
        </div>
      </div>
    </OverlayLayer>
  </div>
</template>

<style scoped>
.workspace-section.page-detail-section {
  --page-detail-error-background: rgba(239, 68, 68, .1);
  --page-detail-image-overlay-background: rgba(0, 0, 0, 0);
  --page-detail-image-overlay-hover-background: rgba(0, 0, 0, .3);
  --page-detail-analyzed-background: rgba(34, 197, 94, .1);
  --page-detail-preview-backdrop: rgba(0, 0, 0, .95);
  --page-detail-preview-control-background: rgba(255, 255, 255, .2);
  --page-detail-preview-control-hover-background: rgba(255, 255, 255, .3);
  --page-detail-error-text: #ef4444;
  --page-detail-success-text: #22c55e;
  --ui-button-padding: 10px 18px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--insight-action-primary);
  --ui-button-primary-hover-background: var(--insight-action-primary-strong);
  --ui-button-secondary-background: var(--insight-surface-tertiary);
  --ui-button-secondary-color: var(--insight-text-primary);
  --ui-button-secondary-border: 1px solid var(--color-border-muted);
  --ui-button-secondary-hover-background: var(--color-border-muted);
  --ui-button-sm-padding: 8px 14px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;

  padding: 20px 18px;
}

.page-detail-section .placeholder-text {
  text-align: center;
  padding: 24px;
  color: var(--insight-text-secondary);
}

.page-detail-section .placeholder-text p {
  max-width: 220px;
  margin: 0 auto;
}

.page-detail-section .empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.page-detail-section .loading-state {
  text-align: center;
  padding: 24px;
  color: var(--insight-text-secondary);
}

.page-detail-section .loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--color-border-muted);
  border-top-color: var(--insight-action-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 12px;
}

.page-detail-section .page-detail-header h4 {
  margin: 0;
  font-size: 16px;
}

.page-detail-section .page-nav-buttons {
  display: flex;
  align-items: center;
  gap: 8px;
}

.page-detail-section .btn-page-nav {
  padding: 4px 12px;
  font-size: 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
  background: var(--insight-surface-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.page-detail-section .btn-page-nav:hover:not(.disabled) {
  background: var(--insight-surface-hover);
  border-color: var(--insight-action-primary);
}

.page-detail-section .btn-page-nav.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-detail-section .page-indicator {
  font-size: 12px;
  color: var(--insight-text-secondary);
  min-width: 60px;
  text-align: center;
}

.page-detail-section .error-message {
  font-size: 12px;
  color: var(--page-detail-error-text);
  background: var(--page-detail-error-background);
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
}

.page-detail-section .page-detail-image {
  position: relative;
  display: block;
  width: 100%;
  margin-bottom: 12px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
  border-radius: 4px;
  overflow: hidden;
  text-align: left;
}

.page-detail-section .page-detail-image img {
  max-width: 100%;
  display: block;
  border-radius: 4px;
}

.page-detail-section .image-overlay {
  position: absolute;
  inset: 0;
  background: var(--page-detail-image-overlay-background);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.page-detail-section .page-detail-image:hover .image-overlay {
  background: var(--page-detail-image-overlay-hover-background);
}

.page-detail-section .zoom-hint {
  color: var(--color-text-inverse);
  font-size: 14px;
  opacity: 0;
  transition: opacity 0.2s;
}

.page-detail-section .page-detail-image:hover .zoom-hint {
  opacity: 1;
}

.page-detail-section .analysis-status-tag {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--insight-surface-secondary);
  color: var(--insight-text-secondary);
  margin-bottom: 12px;
}

.page-detail-section .analysis-status-tag.analyzed {
  background: var(--page-detail-analyzed-background);
  color: var(--page-detail-success-text);
}

.page-detail-section .page-summary {
  margin-bottom: 16px;
}

.page-detail-section .page-summary h5 {
  font-size: 14px;
  margin: 0 0 8px;
  color: var(--insight-text-primary);
}

.page-detail-section .page-summary p {
  font-size: 14px;
  line-height: 1.6;
  color: var(--insight-text-secondary);
  margin: 0;
}

.page-detail-section .page-summary.empty p {
  font-style: italic;
}

.page-detail-section .scene-mood-info {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
  padding: 10px;
  background: var(--insight-surface-secondary);
  border-radius: 6px;
}

.page-detail-section .info-item {
  font-size: 13px;
}

.page-detail-section .info-label {
  color: var(--insight-text-secondary);
}

.page-detail-section .info-value {
  color: var(--insight-text-primary);
}

.page-detail-section .dialogues-section {
  margin-bottom: 16px;
}

.page-detail-section .dialogues-section h5 {
  font-size: 14px;
  margin: 0 0 12px;
  color: var(--insight-text-primary);
}

.page-detail-section .dialogues-section.empty p {
  font-size: 13px;
  color: var(--insight-text-secondary);
  font-style: italic;
}

.page-detail-section .dialogue-item {
  padding: 10px 12px;
  margin: 8px 0;
  background: var(--insight-surface-secondary);
  border-radius: 8px;
  border-left: 3px solid var(--insight-action-primary);
}

.page-detail-section .dialogue-speaker {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  font-size: 12px;
  color: var(--insight-action-primary);
  margin-bottom: 6px;
}

.page-detail-section .speaker-icon {
  font-size: 14px;
}

.page-detail-section .dialogue-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--insight-text-primary);
}

.page-detail-section .dialogue-original {
  font-size: 12px;
  color: var(--insight-text-secondary);
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed var(--color-border-muted);
}

.page-detail-section .original-label {
  font-weight: 500;
}

.page-detail-section .page-detail-actions {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}

.page-detail-section .btn-spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid currentcolor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 6px;
}

.page-detail-section .image-preview-modal {
  background: var(--page-detail-preview-backdrop);
  display: flex;
  align-items: center;
  justify-content: center;
  outline: none;
}

.page-detail-section .image-preview-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.page-detail-section .image-preview-content img {
  max-width: 100%;
  max-height: calc(90vh - 60px);
  object-fit: contain;
}

.page-detail-section .preview-close {
  position: absolute;
  top: -45px;
  right: 0;
  background: none;
  border: none;
  color: var(--color-text-inverse);
  font-size: 36px;
  cursor: pointer;
  padding: 5px 10px;
  transition: transform 0.2s;
}

.page-detail-section .preview-close:hover {
  transform: scale(1.1);
}

.page-detail-section .preview-nav {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 16px;
}

.page-detail-section .preview-nav-btn {
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background: var(--page-detail-preview-control-background);
  color: var(--color-text-inverse);
  font-size: 18px;
  cursor: pointer;
  transition: all 0.2s;
}

.page-detail-section .preview-nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.page-detail-section .preview-nav-btn:hover:not(:disabled) {
  background: var(--page-detail-preview-control-hover-background);
}

.page-detail-section .preview-page-info {
  color: var(--color-text-inverse);
  font-size: 14px;
  min-width: 80px;
  text-align: center;
}
</style>
