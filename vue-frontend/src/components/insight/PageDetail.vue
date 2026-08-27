<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'

import { ref, computed, onUnmounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import * as insightApi from '@/api/insight'
import type { PageAnalysisData } from '@/api/insight'
import { useBodyScrollLock } from '@/composables/useBodyScrollLock'
import { useDialogLifecycle } from '@/composables/useDialogLifecycle'
import { triggerBlobDownload } from '@/utils/browserDownload'

const insightStore = useInsightStore()
const taskCenterStore = useTaskCenterStore()

const pageAnalysis = ref<PageAnalysisData | null>(null)
const loadedImageUrl = ref('')
const isLoading = ref(false)
const isReanalyzing = ref(false)
const pendingReanalyzePage = ref<number | null>(null)
const showImagePreview = ref(false)
const isPageImageUnavailable = ref(false)
const imagePreviewDialog = ref<HTMLElement | null>(null)
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
  return loadedImageUrl.value
})

const hasPageImage = computed(() => Boolean(pageImageUrl.value) && !isPageImageUnavailable.value)

const pageAnalysisState = computed(() => pageAnalysis.value?.analysisState ?? 'not_analyzed')

const isPageAnalyzed = computed(() => {
  return pageAnalysisState.value === 'ready' || pageAnalysisState.value === 'stale'
})

const analysisStatusLabel = computed(() => {
  switch (pageAnalysisState.value) {
    case 'ready':
      return '✓ 已分析'
    case 'stale':
      return '△ 结果已过期'
    case 'running':
      return '… 分析中'
    case 'failed':
      return '! 分析失败'
    default:
      return '○ 未分析'
  }
})

const emptySummaryMessage = computed(() => {
  switch (pageAnalysisState.value) {
    case 'ready':
      return '此页分析完成，但没有生成页面摘要'
    case 'stale':
      return '此页分析结果已过期，可重新分析以更新结果'
    case 'running':
      return '正在分析此页，完成后会自动更新'
    case 'failed':
      return '此页分析失败，可点击下方按钮重试'
    default:
      return '此页尚未分析，点击下方按钮开始分析'
  }
})

const isReanalyzeTaskRunning = computed(() => {
  if (pageAnalysisState.value === 'running') return true
  return (
    pendingReanalyzePage.value !== null &&
    pendingReanalyzePage.value === selectedPageNum.value &&
    insightStore.isAnalyzing
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
    const page = await insightApi.getPageData(bookId, pageNum)

    if (!isCurrentPageDetailRequest(requestId, bookId, pageNum)) return

    pageAnalysis.value = page.analysis
    loadedImageUrl.value = page.sourceUrl
  } catch (error) {
    if (!isCurrentPageDetailRequest(requestId, bookId, pageNum)) return
    pageAnalysis.value = null
    loadedImageUrl.value = ''
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
  const bookId = insightStore.currentBookId
  const pageNum = selectedPageNum.value
  if (!bookId || !pageNum || isReanalyzing.value) return

  isReanalyzing.value = true
  errorMessage.value = ''

  try {
    const submission = await insightApi.reanalyzePage(bookId, pageNum)
    if (insightStore.currentBookId !== bookId || selectedPageNum.value !== pageNum) return
    pendingReanalyzePage.value = pageNum
    taskCenterStore.trackJob(submission.jobId)
  } catch (error) {
    if (insightStore.currentBookId === bookId && selectedPageNum.value === pageNum) {
      const message = (error as { message?: string })?.message
      errorMessage.value = message || '重新分析失败'
    }
  } finally {
    isReanalyzing.value = false
  }
}

function openImagePreview(): void {
  if (!hasPageImage.value) return
  showImagePreview.value = true
}

function closeImagePreview(): void {
  showImagePreview.value = false
}

useDialogLifecycle({
  open: showImagePreview,
  container: imagePreviewDialog,
  close: closeImagePreview,
})
useBodyScrollLock(showImagePreview)

function handlePreviewKeydown(event: KeyboardEvent): void {
  if (!showImagePreview.value) return

  switch (event.key) {
    case 'ArrowLeft':
      if (hasPrevPage.value) {
        event.preventDefault()
        navigatePrev()
      }
      break
    case 'ArrowRight':
      if (hasNextPage.value) {
        event.preventDefault()
        navigateNext()
      }
      break
  }
}

function handlePageImageError(): void {
  isPageImageUnavailable.value = true
  closeImagePreview()
}

const isExporting = ref(false)

async function exportPageData(): Promise<void> {
  const bookId = insightStore.currentBookId
  const pageNum = selectedPageNum.value
  if (!bookId || !pageNum || !pageAnalysis.value || isExporting.value) {
    return
  }

  isExporting.value = true

  try {
    const blob = await insightApi.downloadPageAnalysis(
      bookId,
      pageNum,
    )
    triggerBlobDownload(blob, `${bookId}_page_${pageNum}.md`)

  } catch (error) {
    if (insightStore.currentBookId === bookId && selectedPageNum.value === pageNum) {
      errorMessage.value = error instanceof Error ? error.message : '导出失败'
    }
  } finally {
    isExporting.value = false
  }
}

watch(
  [() => insightStore.currentBookId, selectedPageNum],
  ([bookId, pageNum], [previousBookId]) => {
    if (bookId !== previousBookId) pendingReanalyzePage.value = null
    if (bookId !== previousBookId || pageNum === null) closeImagePreview()
    loadedImageUrl.value = ''
    void loadPageDetail()
  },
  { immediate: true },
)

watch(pageImageUrl, () => {
  isPageImageUnavailable.value = false
})

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
  <div class="page-detail-panel">
    <ProductSectionHeader
      class="page-detail-panel__section-header"
      title="页面详情"
      icon-name="file-text"
      size="sm"
    >
      <template #icon>📄</template>
    </ProductSectionHeader>

    <div class="page-detail-panel__body">
      <ProductEmptyState
        v-if="!selectedPageNum"
        class="page-detail-panel__empty-state"
        role="note"
        title="点击左侧导航树中的页面查看详情"
      >
        <template #icon>📄</template>
      </ProductEmptyState>

      <div v-else-if="isLoading" class="page-detail-panel__loading-state">
        <UiSpinner
          class="page-detail-panel__loading-indicator"
          label="加载页面详情"
          :decorative="false"
          :size="32"
        />
        <p>加载中...</p>
      </div>

      <div v-else class="page-detail-panel__content">
        <div class="page-detail-panel__header">
          <h4 class="page-detail-panel__title">
            <span aria-hidden="true">📄</span>
            <span>第 {{ selectedPageNum }} 页</span>
          </h4>
          <div class="page-detail-panel__nav-buttons">
            <UiButton
              variant="secondary"
              size="xs"
              :disabled="!hasPrevPage"
              title="上一页 (←)"
              @click="navigatePrev"
            >
              <UiIcon name="chevron-left" size="14" />
              <span>上一张</span>
            </UiButton>
            <span class="page-detail-panel__page-indicator">{{ selectedPageNum }} / {{ totalPages }}</span>
            <UiButton
              variant="secondary"
              size="xs"
              :disabled="!hasNextPage"
              title="下一页 (→)"
              @click="navigateNext"
            >
              <span>下一张</span>
              <UiIcon name="chevron-right" size="14" />
            </UiButton>
          </div>
        </div>

        <ProductStatusBanner
          v-if="errorMessage"
          class="page-detail-panel__error-feedback"
          icon-name="alert-triangle"
          role="alert"
          tone="danger"
        >
          {{ errorMessage }}
        </ProductStatusBanner>

        <UiButton
          variant="toolbar"
          class="page-detail-panel__image-trigger"
          :aria-label="isPageImageUnavailable
            ? `第 ${selectedPageNum} 页图片加载失败`
            : hasPageImage
              ? `预览第 ${selectedPageNum} 页图片`
              : `第 ${selectedPageNum} 页暂无图片`"
          :disabled="!hasPageImage"
          @click="openImagePreview"
        >
          <img
            v-if="hasPageImage"
            class="page-detail-panel__image"
            :src="pageImageUrl"
            :alt="`第${selectedPageNum}页`"
            @error="handlePageImageError"
          >
          <div
            v-else
            class="page-detail-panel__image-fallback"
            role="img"
            :aria-label="isPageImageUnavailable
              ? `第${selectedPageNum}页图片加载失败`
              : `第${selectedPageNum}页暂无图片`"
          >
            <UiIcon name="image" size="28" />
            <span>{{ isPageImageUnavailable ? '图片加载失败' : '暂无页面图片' }}</span>
          </div>
          <div v-if="hasPageImage" class="page-detail-panel__image-overlay">
            <span class="page-detail-panel__zoom-hint">
              <UiIcon name="search" />
              <span>点击放大</span>
            </span>
          </div>
        </UiButton>

        <div
          class="page-detail-panel__analysis-status"
          :class="`page-detail-panel__analysis-status--${pageAnalysisState}`"
          :data-state="pageAnalysisState"
        >
          {{ analysisStatusLabel }}
        </div>

        <div v-if="pageAnalysis?.page_summary" class="page-detail-panel__summary">
          <h5 class="page-detail-panel__summary-title">
            <UiIcon name="file-text" />
            <span>页面摘要</span>
          </h5>
          <p class="page-detail-panel__summary-text">{{ pageAnalysis.page_summary }}</p>
        </div>
        <ProductStatusBanner
          v-else
          class="page-detail-panel__summary-feedback"
          icon-name="file-text"
          role="note"
          tone="neutral"
        >
          {{ emptySummaryMessage }}
        </ProductStatusBanner>

        <div v-if="pageAnalysis?.key_events?.length" class="page-detail-panel__dialogues">
          <h5 class="page-detail-panel__dialogues-title">
            <UiIcon name="sparkles" />
            <span>关键事件 ({{ pageAnalysis.key_events.length }})</span>
          </h5>
          <div
            v-for="(event, index) in pageAnalysis.key_events"
            :key="index"
            class="page-detail-panel__dialogue-item"
          >
            <div class="page-detail-panel__dialogue-speaker">{{ event.importance }}</div>
            <div class="page-detail-panel__dialogue-text">{{ event.summary }}</div>
          </div>
        </div>
        <ProductStatusBanner
          v-else-if="isPageAnalyzed"
          class="page-detail-panel__dialogue-feedback"
          icon-name="sparkles"
          role="note"
          title="此页没有识别出关键事件"
          tone="neutral"
        >
          当前页面分析结果中没有关键事件。
        </ProductStatusBanner>

        <ProductStatusBanner
          v-if="pageAnalysis?.continuity_notes"
          class="page-detail-panel__dialogue-feedback"
          icon-name="link"
          role="note"
          title="连续性说明"
          tone="neutral"
        >
          {{ pageAnalysis.continuity_notes }}
        </ProductStatusBanner>

        <ProductStatusBanner
          v-for="warning in pageAnalysis?.warnings || []"
          :key="warning.code + warning.message"
          class="page-detail-panel__dialogue-feedback"
          icon-name="alert-triangle"
          role="note"
          :title="warning.code"
          tone="warning"
        >
          {{ warning.message }}
        </ProductStatusBanner>

        <div class="page-detail-panel__actions">
          <UiButton
            variant="secondary"
            size="sm"
            :disabled="isReanalyzing || isReanalyzeTaskRunning"
            :loading="isReanalyzing || isReanalyzeTaskRunning"
            @click="reanalyzePage"
          >
            <UiSpinner
              v-if="isReanalyzing || isReanalyzeTaskRunning"
              class="page-detail-panel__action-spinner"
            />
            <span v-if="isReanalyzing">启动中...</span>
            <span v-else-if="isReanalyzeTaskRunning">分析中...</span>
            <template v-else>
              <UiIcon name="refresh" />
              <span>重新分析</span>
            </template>
          </UiButton>
          <UiButton
            v-if="isPageAnalyzed"
            variant="secondary"
            size="sm"
            :disabled="isExporting"
            @click="exportPageData"
          >
            <span v-if="isExporting">导出中...</span>
            <template v-else>
              <UiIcon name="file-text" />
              <span>导出此页</span>
            </template>
          </UiButton>
        </div>
      </div>
    </div>

    <OverlayLayer
      v-if="showImagePreview"
      class="page-detail-panel__image-preview-layer"
      @backdrop="closeImagePreview"
    >
      <div
        ref="imagePreviewDialog"
        class="page-detail-panel__image-preview-modal page-detail-panel__image-preview-content"
        role="dialog"
        aria-modal="true"
        :aria-label="`第 ${selectedPageNum} 页图片预览`"
        tabindex="-1"
        @click.stop
        @keydown="handlePreviewKeydown"
      >
        <UiIconButton
          class="page-detail-panel__preview-close"
          label="关闭图片预览"
          title="关闭 (Esc)"
          variant="inverse"
          size="xl"
          shape="circle"
          @click="closeImagePreview"
        >
          <UiIcon name="x" size="24" />
        </UiIconButton>
        <UiSpinner
          v-if="isLoading"
          class="page-detail-panel__preview-loading"
          label="加载预览图片"
          :decorative="false"
          :size="32"
        />
        <img
          v-else-if="hasPageImage"
          class="page-detail-panel__preview-image"
          :src="pageImageUrl"
          :alt="`第${selectedPageNum}页`"
          @error="handlePageImageError"
        >
        <ProductStatusBanner
          v-else
          class="page-detail-panel__preview-unavailable"
          tone="neutral"
          icon-name="image"
          role="note"
        >
          当前页面图片不可用
        </ProductStatusBanner>
        <div class="page-detail-panel__preview-nav">
          <UiIconButton
            class="page-detail-panel__preview-nav-button page-detail-panel__preview-nav-button--prev"
            :disabled="!hasPrevPage"
            label="预览上一页"
            title="上一页 (←)"
            variant="inverse"
            size="lg"
            shape="circle"
            @click.stop="navigatePrev"
          >
            <UiIcon name="chevron-left" size="20" />
          </UiIconButton>
          <span class="page-detail-panel__preview-page-info">{{ selectedPageNum }} / {{ totalPages }}</span>
          <UiIconButton
            class="page-detail-panel__preview-nav-button page-detail-panel__preview-nav-button--next"
            :disabled="!hasNextPage"
            label="预览下一页"
            title="下一页 (→)"
            variant="inverse"
            size="lg"
            shape="circle"
            @click.stop="navigateNext"
          >
            <UiIcon name="chevron-right" size="20" />
          </UiIconButton>
        </div>
      </div>
    </OverlayLayer>
  </div>
</template>

<style scoped>
.page-detail-panel {
  --page-detail-image-fallback-background: var(--insight-surface-secondary);
  --page-detail-image-fallback-border: var(--color-border-muted);
  --page-detail-image-fallback-text: var(--insight-text-secondary);
  --page-detail-image-overlay-background: transparent;
  --page-detail-image-overlay-hover-background: var(--color-overlay-scrim-subtle);
  --page-detail-ready-background: color-mix(in srgb, var(--color-status-success) 12%, transparent);
  --page-detail-stale-background: var(--color-status-warning-surface-soft);
  --page-detail-running-background: color-mix(in srgb, var(--color-status-info) 12%, transparent);
  --page-detail-failed-background: var(--color-surface-danger-soft);
  --page-detail-preview-backdrop: color-mix(in srgb, var(--color-overlay-backdrop-solid) 95%, transparent);
  --page-detail-success-text: var(--color-status-success);

  padding: 20px 18px;
  border-bottom: 1px solid var(--color-border-muted);
}

.page-detail-panel__section-header {
  margin-top: 14px;
}

.page-detail-panel__loading-state {
  display: grid;
  justify-items: center;
  gap: 12px;
  text-align: center;
  padding: 24px;
  color: var(--insight-text-secondary);
}

.page-detail-panel__empty-state {
  --product-empty-state-min-height: 0;
  --product-empty-state-padding: 55px 24px 4px;
  --product-empty-state-icon-width: auto;
  --product-empty-state-icon-height: auto;
  --product-empty-state-icon-margin-bottom: 22px;
  --product-empty-state-icon-border: 0;
  --product-empty-state-icon-radius: 0;
  --product-empty-state-icon-background: transparent;
  --product-empty-state-icon-color: inherit;
  --product-empty-state-icon-font-size: 48px;
  --product-empty-state-title: var(--insight-text-secondary);
  --product-empty-state-title-font-size: 16px;
  --product-empty-state-title-font-weight: 400;
  --product-empty-state-title-margin: 16px 0;
}

.page-detail-panel__loading-indicator {
  color: var(--insight-action-primary);
}

.page-detail-panel__header {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px 12px;
  margin-bottom: 12px;
}

.page-detail-panel__title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  font-size: 16px;
}

.page-detail-panel__nav-buttons {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.page-detail-panel__page-indicator {
  font-size: 12px;
  color: var(--insight-text-secondary);
  min-width: 60px;
  text-align: center;
}

.page-detail-panel__error-feedback {
  margin-bottom: 12px;
}

.page-detail-panel__image-trigger {
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

.page-detail-panel__image-trigger:disabled {
  cursor: not-allowed;
}

.page-detail-panel__image {
  max-width: 100%;
  display: block;
  border-radius: 4px;
}

.page-detail-panel__image-fallback {
  display: flex;
  min-height: 180px;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 1px dashed var(--page-detail-image-fallback-border);
  border-radius: 4px;
  background: var(--page-detail-image-fallback-background);
  color: var(--page-detail-image-fallback-text);
  font-size: 13px;
}

.page-detail-panel__image-overlay {
  position: absolute;
  inset: 0;
  background: var(--page-detail-image-overlay-background);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.page-detail-panel__image-trigger:hover .page-detail-panel__image-overlay,
.page-detail-panel__image-trigger:focus-visible .page-detail-panel__image-overlay {
  background: var(--page-detail-image-overlay-hover-background);
}

.page-detail-panel__zoom-hint {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--color-text-inverse);
  font-size: 14px;
  opacity: 0;
  transition: opacity 0.2s;
}

.page-detail-panel__image-trigger:hover .page-detail-panel__zoom-hint,
.page-detail-panel__image-trigger:focus-visible .page-detail-panel__zoom-hint {
  opacity: 1;
}

.page-detail-panel__analysis-status {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--insight-surface-secondary);
  color: var(--insight-text-secondary);
  margin-bottom: 12px;
}

.page-detail-panel__analysis-status--ready {
  background: var(--page-detail-ready-background);
  color: var(--page-detail-success-text);
}

.page-detail-panel__analysis-status--stale {
  background: var(--page-detail-stale-background);
  color: var(--color-status-warning);
}

.page-detail-panel__analysis-status--running {
  background: var(--page-detail-running-background);
  color: var(--color-status-info);
}

.page-detail-panel__analysis-status--failed {
  background: var(--page-detail-failed-background);
  color: var(--color-text-danger);
}

.page-detail-panel__summary {
  margin-bottom: 16px;
}

.page-detail-panel__summary-title,
.page-detail-panel__dialogues-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--insight-text-primary);
}

.page-detail-panel__summary-title {
  font-size: 14px;
  margin: 0 0 8px;
}

.page-detail-panel__summary-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--insight-text-secondary);
  margin: 0;
}

.page-detail-panel__summary-feedback {
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-padding: 0;
  --product-status-banner-icon-display: none;
  --product-status-banner-body-color: var(--insight-text-secondary);
  --product-status-banner-body-font-size: 14px;

  margin-bottom: 16px;
  font-style: italic;
}

.page-detail-panel__dialogues {
  margin-bottom: 16px;
}

.page-detail-panel__dialogues-title {
  font-size: 14px;
  margin: 0 0 12px;
}

.page-detail-panel__dialogue-feedback {
  margin-bottom: 16px;
}

.page-detail-panel__dialogue-item {
  padding: 10px 12px;
  margin: 8px 0;
  background: var(--insight-surface-secondary);
  border-radius: 8px;
  border-left: 3px solid var(--insight-action-primary);
}

.page-detail-panel__dialogue-speaker {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  font-size: 12px;
  color: var(--insight-action-primary);
  margin-bottom: 6px;
}

.page-detail-panel__dialogue-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--insight-text-primary);
}

.page-detail-panel__actions {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}

.page-detail-panel__image-preview-layer {
  background: var(--page-detail-preview-backdrop);
  display: flex;
  align-items: center;
  justify-content: center;
}

.page-detail-panel__image-preview-modal {
  outline: none;
}

.page-detail-panel__image-preview-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.page-detail-panel__preview-image {
  max-width: 100%;
  max-height: calc(90vh - 60px);
  object-fit: contain;
}

.page-detail-panel__preview-loading {
  color: var(--color-text-inverse);
}

.page-detail-panel__preview-unavailable {
  min-width: min(320px, 80vw);
}

.page-detail-panel__preview-close {
  position: absolute;
  top: -45px;
  right: 0;
}

.page-detail-panel__preview-nav {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 16px;
}

.page-detail-panel__preview-nav-button {
  color: var(--color-text-inverse);
}

.page-detail-panel__preview-page-info {
  color: var(--color-text-inverse);
  font-size: 14px;
  min-width: 80px;
  text-align: center;
}
</style>
