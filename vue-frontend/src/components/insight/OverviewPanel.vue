<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import type { OverviewTemplateType } from '@/types/insight'
import * as insightApi from '@/api/insight'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { marked } from 'marked'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { triggerBlobDownload } from '@/utils/browserDownload'
import { showToast } from '@/utils/toast'

const insightStore = useInsightStore()

const currentTemplate = ref<OverviewTemplateType>('no_spoiler')
const overviewContent = ref('')
const queuedMessage = ref('')
const errorMessage = ref('')
const recentPagesError = ref('')
const isLoading = ref(false)
const isExporting = ref(false)
const generatedTemplates = ref<OverviewTemplateType[]>([])
let overviewContentRequestSequence = 0
let generatedTemplatesRequestSequence = 0
let recentPagesRequestSequence = 0
let isOverviewPanelMounted = true

const recentAnalyzedPages = ref<Array<{
  page_num: number
  summary?: string
}>>([])

interface OverviewTemplateOption {
  value: OverviewTemplateType
  label: string
  iconText: string
  description: string
}

const templateOptions: OverviewTemplateOption[] = [
  { value: 'no_spoiler', label: '无剧透简介', iconText: '🎁', description: '不含剧透的简短介绍，适合推荐给他人' },
  { value: 'story_summary', label: '故事概要', iconText: '📖', description: '完整的剧情回顾，包含所有剧透' },
  { value: 'recap', label: '前情回顾', iconText: '⏪', description: '之前发生的重要事件回顾' },
  { value: 'character_guide', label: '角色图鉴', iconText: '👥', description: '主要角色介绍和关系' },
  { value: 'world_setting', label: '世界观设定', iconText: '🌍', description: '故事背景和世界观设定' },
  { value: 'highlights', label: '名场面盘点', iconText: '✨', description: '精彩片段和经典场景回顾' },
  { value: 'reading_notes', label: '阅读笔记', iconText: '📝', description: '阅读过程中的重点笔记' }
]

const templateSelectOptions = templateOptions.map(t => ({
  label: `${t.iconText} ${t.label}`,
  value: t.value
}))

const currentTemplateIconText = computed(() => {
  const template = templateOptions.find(t => t.value === currentTemplate.value)
  return template?.iconText || '📊'
})

const currentTemplateDescription = computed(() => {
  const template = templateOptions.find(t => t.value === currentTemplate.value)
  return template?.description || ''
})

const templateStatus = computed(() => {
  if (overviewContent.value || generatedTemplates.value.includes(currentTemplate.value)) {
    return '已生成'
  }
  return ''
})

const renderedContent = computed(() => {
  if (!overviewContent.value) return ''
  return sanitizeHtml(marked.parse(overviewContent.value) as string)
})

async function onTemplateChange(): Promise<void> {
  await loadCachedOverview()
}

function isCurrentBookRequest(requestId: number, currentRequestId: number, bookId: string): boolean {
  return (
    isOverviewPanelMounted &&
    requestId === currentRequestId &&
    insightStore.currentBookId === bookId
  )
}

function isCurrentOverviewRequest(
  requestId: number,
  bookId: string,
  template: OverviewTemplateType
): boolean {
  return (
    isCurrentBookRequest(requestId, overviewContentRequestSequence, bookId) &&
    currentTemplate.value === template
  )
}

async function loadCachedOverview(
  bookId = insightStore.currentBookId,
  template = currentTemplate.value
): Promise<void> {
  const requestId = ++overviewContentRequestSequence
  if (!bookId) {
    isLoading.value = false
    overviewContent.value = ''
    queuedMessage.value = ''
    errorMessage.value = ''
    return
  }

  isLoading.value = true
  overviewContent.value = ''
  queuedMessage.value = ''
  errorMessage.value = ''

  try {
    const content = await insightApi.getOverview(
      bookId,
      template
    )

    if (!isCurrentOverviewRequest(requestId, bookId, template)) return

    if (content) {
      overviewContent.value = content
      if (!generatedTemplates.value.includes(template)) {
        generatedTemplates.value.push(template)
      }
    } else {
      overviewContent.value = ''
    }
  } catch (error) {
    if (!isCurrentOverviewRequest(requestId, bookId, template)) return
    errorMessage.value = error instanceof Error
      ? `加载失败：${error.message}`
      : '加载失败，请重试'
  } finally {
    if (isCurrentOverviewRequest(requestId, bookId, template)) {
      isLoading.value = false
    }
  }
}

async function generateOverview(regenerate: boolean): Promise<void> {
  const bookId = insightStore.currentBookId
  const template = currentTemplate.value
  if (!bookId || isLoading.value) return
  const requestId = ++overviewContentRequestSequence

  isLoading.value = true
  overviewContent.value = ''
  queuedMessage.value = ''
  errorMessage.value = ''

  try {
    const result = await insightApi.regenerateOverview(
      bookId,
      template,
      regenerate
    )

    if (!isCurrentOverviewRequest(requestId, bookId, template)) return

    if (result.kind === 'cached') {
      overviewContent.value = result.content
      if (!generatedTemplates.value.includes(template)) {
        generatedTemplates.value.push(template)
      }
    } else {
      queuedMessage.value = '概览生成已进入任务中心，完成后将自动加载。'
    }
  } catch (error) {
    if (!isCurrentOverviewRequest(requestId, bookId, template)) return
    errorMessage.value = error instanceof Error
      ? `生成失败：${error.message}`
      : '生成失败，请重试'
  } finally {
    if (isCurrentOverviewRequest(requestId, bookId, template)) {
      isLoading.value = false
    }
  }
}

async function loadGeneratedTemplates(bookId = insightStore.currentBookId): Promise<OverviewTemplateType[]> {
  const requestId = ++generatedTemplatesRequestSequence
  if (!bookId) return []

  try {
    const templates = await insightApi.getGeneratedTemplates(bookId)
    if (!isCurrentBookRequest(requestId, generatedTemplatesRequestSequence, bookId)) return []

    generatedTemplates.value = templates
    return templates
  } catch {
    if (!isCurrentBookRequest(requestId, generatedTemplatesRequestSequence, bookId)) return []
  }
  return []
}

async function exportAnalysisData(): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId) {
    showToast('请先选择书籍', 'warning')
    return
  }
  if (isExporting.value) return

  isExporting.value = true

  try {
    await insightApi.exportAnalysis(bookId)
    showToast('完整导出已进入任务中心，完成后可下载', 'success')
  } catch (error) {
    showToast(
      error instanceof Error ? error.message : '导出失败',
      'error',
    )
  } finally {
    isExporting.value = false
  }
}

async function exportCurrentOverview(): Promise<void> {
  if (!overviewContent.value) {
    showToast('暂无内容可导出', 'warning')
    return
  }

  const bookId = insightStore.currentBookId
  const template = currentTemplate.value
  if (!bookId || isExporting.value) return

  isExporting.value = true
  try {
    const blob = await insightApi.downloadCurrentOverview(
      bookId,
      template,
    )
    triggerBlobDownload(
      blob,
      `${bookId}_${template}.md`,
    )
    showToast('导出成功', 'success')
  } catch (error) {
    showToast(error instanceof Error ? error.message : '导出失败', 'error')
  } finally {
    isExporting.value = false
  }
}

async function loadRecentAnalyzedPages(bookId = insightStore.currentBookId): Promise<void> {
  const requestId = ++recentPagesRequestSequence
  if (!bookId) return
  recentPagesError.value = ''

  try {
    const recentPages = await insightApi.getRecentAnalyzedPages(bookId)
    if (!isCurrentBookRequest(requestId, recentPagesRequestSequence, bookId)) return
    recentAnalyzedPages.value = recentPages
  } catch (error) {
    if (!isCurrentBookRequest(requestId, recentPagesRequestSequence, bookId)) return
    recentAnalyzedPages.value = []
    recentPagesError.value = error instanceof Error ? error.message : '加载最近分析失败'
  }
}

function retryRecentAnalyzedPages(): void {
  void loadRecentAnalyzedPages()
}

function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

async function refreshOverviewForCurrentBook(): Promise<void> {
  const bookId = insightStore.currentBookId

  isLoading.value = false
  overviewContent.value = ''
  queuedMessage.value = ''
  errorMessage.value = ''
  generatedTemplates.value = []
  recentAnalyzedPages.value = []
  recentPagesError.value = ''
  if (!bookId) return

  await Promise.all([
    loadGeneratedTemplates(bookId),
    loadRecentAnalyzedPages(bookId),
    loadCachedOverview(bookId, currentTemplate.value),
  ])
}

onMounted(async () => {
  await refreshOverviewForCurrentBook()
})

watch(() => insightStore.currentBookId, async () => {
  await refreshOverviewForCurrentBook()
})

watch(() => insightStore.dataRefreshKey, async (newKey) => {
  if (newKey > 0 && insightStore.currentBookId) {
    await refreshOverviewForCurrentBook()
  }
})

onUnmounted(() => {
  isOverviewPanelMounted = false
  overviewContentRequestSequence += 1
  generatedTemplatesRequestSequence += 1
  recentPagesRequestSequence += 1
})
</script>

<template>
  <div class="overview-panel">
    <div class="overview-panel__card overview-panel__card--summary">
      <div class="overview-panel__card-header">
        <div class="overview-panel__template-heading">
          <span class="overview-panel__title-icon" aria-hidden="true">{{ currentTemplateIconText }}</span>
          <UiSelect
            v-model="currentTemplate"
            aria-label="选择概览模板"
            :options="templateSelectOptions"
            @change="onTemplateChange"
          />
        </div>
        <div class="overview-panel__card-actions">
          <span class="overview-panel__template-status">{{ templateStatus }}</span>
          <UiIconButton
            label="生成/加载"
            size="sm"
            :disabled="isLoading"
            @click="generateOverview(false)"
          >
            <span aria-hidden="true">📄</span>
          </UiIconButton>
          <UiIconButton
            label="重新生成"
            size="sm"
            :disabled="isLoading"
            @click="generateOverview(true)"
          >
            <span aria-hidden="true">🔄</span>
          </UiIconButton>
        </div>
      </div>
      <p class="overview-panel__template-description">{{ currentTemplateDescription }}</p>
      <div class="overview-panel__card-content overview-panel__markdown">
        <ProductStatusBanner
          v-if="isLoading"
          tone="neutral"
          icon-name="refresh"
          title="正在加载概览"
          aria-live="polite"
        >
          正在读取当前模板的概览内容。
        </ProductStatusBanner>
        <div v-else-if="overviewContent" v-html="renderedContent"></div>
        <ProductStatusBanner
          v-else-if="errorMessage"
          tone="danger"
          icon-name="alert-triangle"
          title="概览操作失败"
          aria-live="polite"
        >
          {{ errorMessage }}
        </ProductStatusBanner>
        <ProductStatusBanner
          v-else-if="queuedMessage"
          tone="neutral"
          icon-name="refresh"
          title="概览生成中"
          aria-live="polite"
        >
          {{ queuedMessage }}
        </ProductStatusBanner>
        <ProductStatusBanner
          v-else
          class="overview-panel__empty-feedback"
          tone="neutral"
          icon-name="file-text"
        >
          选择模板类型，点击生成按钮
        </ProductStatusBanner>
      </div>
    </div>

    <div class="overview-panel__card overview-panel__card--stats">
      <h3 class="overview-panel__card-title">
        <span aria-hidden="true">📊</span>
        <span>分析统计</span>
      </h3>
      <div class="overview-panel__stats-grid">
        <div class="overview-panel__stat-item">
          <span class="overview-panel__stat-value">{{ insightStore.analyzedPageCount }}</span>
          <span class="overview-panel__stat-label">已分析页面</span>
        </div>
        <div class="overview-panel__stat-item">
          <span class="overview-panel__stat-value">{{ insightStore.chapters.length }}</span>
          <span class="overview-panel__stat-label">章节数</span>
        </div>
      </div>

      <ProductActionRow
        aria-label="概览导出操作"
        class="overview-panel__export-actions"
        justify="start"
        variant="toolbar"
      >
        <UiButton
          variant="secondary"
          :disabled="isExporting || !overviewContent"
          title="导出当前概览"
          @click="exportCurrentOverview" size="sm"
        >
          <span aria-hidden="true">📄</span>
          <span>导出当前</span>
        </UiButton>
        <UiButton
          variant="primary"
          :disabled="isExporting"
          title="导出完整分析数据"
          @click="exportAnalysisData" size="sm"
        >
          <span v-if="!isExporting" aria-hidden="true">📤</span>
          <span>{{ isExporting ? '导出中...' : '导出全部' }}</span>
        </UiButton>
      </ProductActionRow>
    </div>

    <div class="overview-panel__card overview-panel__card--recent">
      <h3 class="overview-panel__card-title">
        <span aria-hidden="true">🕐</span>
        <span>最近分析</span>
      </h3>
      <div class="overview-panel__recent-pages">
        <ProductStatusBanner
          v-if="recentPagesError"
          class="overview-panel__recent-error"
          tone="danger"
          icon-name="alert-triangle"
          title="最近分析加载失败"
          role="alert"
        >
          {{ recentPagesError }}
          <template #actions>
            <UiButton variant="secondary" size="sm" @click="retryRecentAnalyzedPages">
              重新加载
            </UiButton>
          </template>
        </ProductStatusBanner>
        <ProductStatusBanner
          v-else-if="recentAnalyzedPages.length === 0"
          class="overview-panel__empty-feedback"
          tone="neutral"
          icon-name="clock"
        >
          暂无分析记录
        </ProductStatusBanner>
        <ProductRecordCard
          v-for="page in recentAnalyzedPages"
          :key="page.page_num"
          as="button"
          class="overview-panel__recent-page-card"
          :aria-label="`查看第 ${page.page_num} 页分析详情`"
          @click="goToPage(page.page_num)"
        >
          <span class="overview-panel__recent-page-content">
            <span class="overview-panel__page-number">第 {{ page.page_num }} 页</span>
            <span v-if="page.summary" class="overview-panel__page-summary">{{ page.summary }}</span>
          </span>
        </ProductRecordCard>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overview-panel {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  gap: 20px;
}

.overview-panel__card {
  padding: 20px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--insight-surface-secondary);
}

.overview-panel__card--summary {
  grid-column: 1 / -1;
}

.overview-panel__card-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px 12px;
  margin-bottom: 12px;
}

.overview-panel__template-heading {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

.overview-panel__title-icon {
  color: var(--insight-action-primary);
  font-size: 20px;
  line-height: 1;
}

.overview-panel__card-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.overview-panel__template-status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
}

.overview-panel__template-description {
  margin: 0 0 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--color-border-muted);
  color: var(--insight-text-muted);
  font-size: 12px;
}

.overview-panel__card-title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 16px;
}

.overview-panel__card-content {
  color: var(--insight-text-secondary);
  line-height: 1.6;
}

.overview-panel__markdown {
  font-size: 14px;
  line-height: 1.8;
}

.overview-panel__markdown h2 {
  margin: 16px 0 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--color-border-muted);
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 16px;
}

.overview-panel__markdown h2:first-child {
  margin-top: 0;
}

.overview-panel__markdown h3 {
  margin: 12px 0 6px;
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 14px;
}

.overview-panel__markdown p {
  margin: 8px 0;
  color: var(--insight-text-secondary);
}

.overview-panel__markdown ul,
.overview-panel__markdown ol {
  margin: 8px 0;
  padding-left: 20px;
}

.overview-panel__markdown li {
  margin: 4px 0;
  color: var(--insight-text-secondary);
}

.overview-panel__markdown strong {
  color: var(--insight-text-primary);
  font-weight: 600;
}

.overview-panel__markdown em {
  color: var(--insight-text-secondary);
  font-style: italic;
}

.overview-panel__markdown blockquote {
  margin: 12px 0;
  padding: 8px 12px;
  border-left: 3px solid var(--insight-action-primary);
  border-radius: 0 6px 6px 0;
  background: var(--insight-surface-tertiary);
}

.overview-panel__markdown blockquote p {
  margin: 0;
}

.overview-panel__markdown hr {
  margin: 16px 0;
  border: none;
  border-top: 1px solid var(--color-border-muted);
}

.overview-panel__stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.overview-panel__stat-item {
  padding: 12px;
  border-radius: 8px;
  background: var(--insight-surface-tertiary);
  text-align: center;
}

.overview-panel__stat-value {
  display: block;
  color: var(--insight-action-primary);
  font-weight: 700;
  font-size: 28px;
}

.overview-panel__stat-label {
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.overview-panel__export-actions {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}

.overview-panel__recent-pages {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.overview-panel__recent-error {
  --product-status-banner-flex-direction: column;
  --product-status-banner-actions-width: 100%;

  margin-bottom: 4px;
}

.overview-panel__empty-feedback {
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-padding: 0;
  --product-status-banner-icon-display: none;
  --product-status-banner-content-display: block;
  --product-status-banner-body-color: var(--insight-text-muted);
  --product-status-banner-body-font-size: 14px;
  --product-status-banner-text-align: center;
}

.overview-panel__recent-page-card {
  --product-record-card-background: var(--insight-surface-tertiary);
  --product-record-card-border: transparent;
  --product-record-card-padding: 10px 12px;
  --product-record-card-radius: 6px;
  --product-record-card-shadow-hover: none;
}

.overview-panel__recent-page-card:hover {
  --product-record-card-background: var(--color-focus-brand-soft);

  transform: translateX(4px);
}

.overview-panel__recent-page-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.overview-panel__page-number {
  flex: 0 0 auto;
  font-size: 13px;
  font-weight: 500;
  color: var(--insight-action-primary);
}

.overview-panel__page-summary {
  flex: 1 1 auto;
  min-width: 0;
  font-size: 12px;
  color: var(--insight-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
