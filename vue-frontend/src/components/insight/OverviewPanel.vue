<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useInsightStore, type OverviewTemplateType } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { marked } from 'marked'
import { sanitizeHtml } from '@/utils/sanitizeHtml'
import { showToast } from '@/utils/toast'

const insightStore = useInsightStore()

const currentTemplate = ref<OverviewTemplateType>('no_spoiler')
const overviewContent = ref('')
const isLoading = ref(false)
const generatedTemplates = ref<OverviewTemplateType[]>([])
let overviewContentRequestSequence = 0
let generatedTemplatesRequestSequence = 0
let recentPagesRequestSequence = 0
let overviewRefreshSequence = 0
let isOverviewPanelMounted = true

const recentAnalyzedPages = ref<Array<{
  page_num: number
  summary?: string
  analyzed_at?: string
}>>([])

const templateOptions: Array<{ value: OverviewTemplateType; label: string; icon: string; description: string }> = [
  { value: 'no_spoiler', label: '无剧透简介', icon: '🎁', description: '不含剧透的简短介绍，适合推荐给他人' },
  { value: 'story_summary', label: '故事概要', icon: '📖', description: '完整的剧情回顾，包含所有剧透' },
  { value: 'recap', label: '前情回顾', icon: '⏪', description: '之前发生的重要事件回顾' },
  { value: 'character_guide', label: '角色图鉴', icon: '👥', description: '主要角色介绍和关系' },
  { value: 'world_setting', label: '世界观设定', icon: '🌍', description: '故事背景和世界观设定' },
  { value: 'highlights', label: '名场面盘点', icon: '✨', description: '精彩片段和经典场景回顾' },
  { value: 'reading_notes', label: '阅读笔记', icon: '📝', description: '阅读过程中的重点笔记' }
]

const templateSelectOptions = templateOptions.map(t => ({
  label: `${t.icon} ${t.label}`,
  value: t.value
}))

const currentTemplateIcon = computed(() => {
  const template = templateOptions.find(t => t.value === currentTemplate.value)
  return template?.icon || '📊'
})

const currentTemplateDescription = computed(() => {
  const template = templateOptions.find(t => t.value === currentTemplate.value)
  return template?.description || ''
})

const templateStatus = computed(() => {
  if (generatedTemplates.value.includes(currentTemplate.value)) {
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
  if (!bookId) return

  isLoading.value = true
  overviewContent.value = ''

  try {
    const response = await insightApi.getOverview(
      bookId,
      template
    )

    if (!isCurrentOverviewRequest(requestId, bookId, template)) return

    if (response.success && response.content) {
      overviewContent.value = response.content
      if (!generatedTemplates.value.includes(template)) {
        generatedTemplates.value.push(template)
      }
    } else {
      overviewContent.value = ''
    }
  } catch {
    if (!isCurrentOverviewRequest(requestId, bookId, template)) return
    overviewContent.value = '加载失败，请重试'
  } finally {
    if (isCurrentOverviewRequest(requestId, bookId, template)) {
      isLoading.value = false
    }
  }
}

async function generateOverview(regenerate: boolean): Promise<void> {
  const bookId = insightStore.currentBookId
  const template = currentTemplate.value
  const requestId = ++overviewContentRequestSequence
  if (!bookId) return

  isLoading.value = true
  overviewContent.value = ''

  try {
    const response = await insightApi.regenerateOverview(
      bookId,
      template,
      regenerate
    )

    if (!isCurrentOverviewRequest(requestId, bookId, template)) return

    if (response.success) {
      if (response.content) {
        overviewContent.value = response.content
        if (!generatedTemplates.value.includes(template)) {
          generatedTemplates.value.push(template)
        }
        if (template === 'story_summary' && response.cached !== true) {
          insightStore.triggerDataRefresh()
        }
      }
    } else {
      overviewContent.value = `生成失败: ${response.error || '未知错误'}`
    }
  } catch {
    if (!isCurrentOverviewRequest(requestId, bookId, template)) return
    overviewContent.value = '生成失败，请重试'
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
    const response = await insightApi.getGeneratedTemplates(bookId)
    if (!isCurrentBookRequest(requestId, generatedTemplatesRequestSequence, bookId)) return []

    if (response.success) {
      let templates: OverviewTemplateType[] = []
      if (response.generated) {
        templates = response.generated as OverviewTemplateType[]
      } else if (response.templates && Array.isArray(response.templates)) {
        templates = response.templates as OverviewTemplateType[]
      }
      generatedTemplates.value = templates
      return templates
    }
  } catch {
    if (!isCurrentBookRequest(requestId, generatedTemplatesRequestSequence, bookId)) return []
    generatedTemplates.value = []
  }
  return []
}

const isExporting = ref(false)

async function exportAnalysisData(): Promise<void> {
  if (!insightStore.currentBookId) {
    showToast('请先选择书籍', 'warning')
    return
  }

  isExporting.value = true

  try {
    const response = await insightApi.exportAnalysis(insightStore.currentBookId)

    if (response.success && response.markdown) {
      const blob = new Blob([response.markdown], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      try {
        const a = document.createElement('a')
        a.href = url
        a.download = `${insightStore.currentBookId}_analysis.md`
        a.click()
      } finally {
        URL.revokeObjectURL(url)
      }

      showToast('导出成功', 'success')
    } else {
      showToast('导出失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch {
    showToast('导出失败', 'error')
  } finally {
    isExporting.value = false
  }
}

function exportCurrentOverview(): void {
  if (!overviewContent.value) {
    showToast('暂无内容可导出', 'warning')
    return
  }

  const template = templateOptions.find(t => t.value === currentTemplate.value)
  const fileName = `${insightStore.currentBookId}_${currentTemplate.value}.md`

  const content = `# ${template?.label || currentTemplate.value}\n\n${overviewContent.value}`

  const blob = new Blob([content], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = url
    a.download = fileName
    a.click()
  } finally {
    URL.revokeObjectURL(url)
  }
  showToast('导出成功', 'success')
}

async function loadRecentAnalyzedPages(bookId = insightStore.currentBookId): Promise<void> {
  const requestId = ++recentPagesRequestSequence
  if (!bookId) return

  try {
    const stats = await insightApi.getAnalysisStatus(bookId)
    if (!isCurrentBookRequest(requestId, recentPagesRequestSequence, bookId)) return

    if (stats.success && insightStore.analyzedPageCount > 0) {
      const totalPages = insightStore.totalPageCount
      const analyzedCount = insightStore.analyzedPageCount
      const recentPages: Array<{ page_num: number; summary?: string }> = []

      const startPage = Math.max(1, analyzedCount - 4)
      for (let i = 0; i < Math.min(5, analyzedCount); i++) {
        const pageNum = startPage + i
        if (pageNum <= totalPages) {
          recentPages.push({
            page_num: pageNum,
            summary: `第 ${pageNum} 页`
          })
        }
      }

      recentAnalyzedPages.value = recentPages.reverse()
    }
  } catch {
    if (!isCurrentBookRequest(requestId, recentPagesRequestSequence, bookId)) return
    recentAnalyzedPages.value = []
  }
}

function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

async function refreshOverviewForCurrentBook(): Promise<void> {
  const bookId = insightStore.currentBookId
  const refreshId = ++overviewRefreshSequence
  if (!bookId) return

  overviewContent.value = ''
  generatedTemplates.value = []
  recentAnalyzedPages.value = []

  const templates = await loadGeneratedTemplates(bookId)
  if (!isCurrentBookRequest(refreshId, overviewRefreshSequence, bookId)) return

  await loadRecentAnalyzedPages(bookId)
  if (!isCurrentBookRequest(refreshId, overviewRefreshSequence, bookId)) return

  if (templates.includes(currentTemplate.value)) {
    await loadCachedOverview(bookId, currentTemplate.value)
  }
}

onMounted(async () => {
  await refreshOverviewForCurrentBook()
})

watch(() => insightStore.currentBookId, async (newBookId) => {
  if (newBookId) {
    await refreshOverviewForCurrentBook()
  }
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
  overviewRefreshSequence += 1
})
</script>

<template>
  <div class="overview-grid">
    <div class="overview-card summary-card">
      <div class="card-header">
        <div class="card-title-with-selector">
          <span class="card-title-icon">{{ currentTemplateIcon }}</span>
          <CustomSelect
            v-model="currentTemplate"
            :options="templateSelectOptions"
            @change="onTemplateChange"
          />
        </div>
        <div class="card-header-actions">
          <span class="template-status">{{ templateStatus }}</span>
          <UiButton
            variant="toolbar"
            class="button-icon"
            title="生成/加载"
            @click="generateOverview(false)"
          >
            📄
          </UiButton>
          <UiButton
            variant="toolbar"
            class="button-icon"
            title="重新生成"
            @click="generateOverview(true)"
          >
            🔄
          </UiButton>
        </div>
      </div>
      <p class="template-description">{{ currentTemplateDescription }}</p>
      <div class="card-content markdown-content">
        <div v-if="isLoading" class="loading-text">加载中...</div>
        <div v-else-if="overviewContent" v-html="renderedContent"></div>
        <div v-else class="placeholder-text">选择模板类型，点击生成按钮</div>
      </div>
    </div>

    <div class="overview-card stats-card">
      <h3 class="card-title">📊 分析统计</h3>
      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-value">{{ insightStore.analyzedPageCount }}</span>
          <span class="stat-label">已分析页面</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">{{ insightStore.chapters.length }}</span>
          <span class="stat-label">章节数</span>
        </div>
      </div>

      <div class="export-actions">
        <UiButton
          variant="secondary"
          class="overview-action-button overview-action-button--secondary"
          :disabled="isExporting || !overviewContent"
          title="导出当前概览"
          @click="exportCurrentOverview" size="sm"
        >
          📄 导出当前
        </UiButton>
        <UiButton
          variant="primary"
          class="overview-action-button overview-action-button--primary"
          :disabled="isExporting"
          title="导出完整分析数据"
          @click="exportAnalysisData" size="sm"
        >
          {{ isExporting ? '导出中...' : '📤 导出全部' }}
        </UiButton>
      </div>
    </div>

    <div class="overview-card recent-card">
      <h3 class="card-title">🕐 最近分析</h3>
      <div class="recent-pages">
        <div v-if="recentAnalyzedPages.length === 0" class="placeholder-text">暂无分析记录</div>
        <UiButton
          v-for="page in recentAnalyzedPages"
          :key="page.page_num"
          variant="toolbar"
          class="recent-page-item"
          :aria-label="`查看第 ${page.page_num} 页分析详情`"
          @click="goToPage(page.page_num)"
        >
          <span class="page-number">第 {{ page.page_num }} 页</span>
          <span v-if="page.summary" class="page-summary">{{ page.summary }}</span>
        </UiButton>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overview-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}

.overview-grid .overview-card {
    background: var(--insight-surface-secondary);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--color-border-muted);
}

.overview-grid .overview-card.summary-card {
    grid-column: span 2;
}

.overview-grid .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.overview-grid .card-title-with-selector {
    display: flex;
    align-items: center;
    gap: 8px;
}

.overview-grid .card-title-icon {
    font-size: 20px;
    line-height: 1;
}

.overview-grid .card-header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
}

.overview-grid .template-status {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
}

.overview-grid .template-description {
    font-size: 12px;
    color: var(--insight-text-muted);
    margin: 0 0 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--color-border-muted);
}

.overview-grid .placeholder-text {
    padding: 0;
    text-align: left;
}

.overview-grid .card-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    color: var(--insight-text-primary);
}

.overview-grid .card-header .card-title {
    margin-bottom: 0;
}

.overview-grid .button-icon {
    width: 32px;
    height: 32px;
    border: none;
    background: var(--insight-surface-tertiary);
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}

.overview-grid .button-icon:hover {
    background: var(--insight-action-primary);
    color: var(--color-text-inverse);
}

.overview-grid .card-content {
    color: var(--insight-text-secondary);
    line-height: 1.6;
}

.overview-grid .markdown-content {
    font-size: 14px;
    line-height: 1.8;
}

.overview-grid .markdown-content h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--insight-text-primary);
    margin: 16px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--color-border-muted);
}

.overview-grid .markdown-content h2:first-child {
    margin-top: 0;
}

.overview-grid .markdown-content h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--insight-text-primary);
    margin: 12px 0 6px;
}

.overview-grid .markdown-content p {
    margin: 8px 0;
    color: var(--insight-text-secondary);
}

.overview-grid .markdown-content ul, .overview-grid .markdown-content ol {
    margin: 8px 0;
    padding-left: 20px;
}

.overview-grid .markdown-content li {
    margin: 4px 0;
    color: var(--insight-text-secondary);
}

.overview-grid .markdown-content strong {
    color: var(--insight-text-primary);
    font-weight: 600;
}

.overview-grid .markdown-content em {
    font-style: italic;
    color: var(--insight-text-secondary);
}

.overview-grid .markdown-content blockquote {
    margin: 12px 0;
    padding: 8px 12px;
    border-left: 3px solid var(--insight-action-primary);
    background: var(--insight-surface-tertiary);
    border-radius: 0 6px 6px 0;
}

.overview-grid .markdown-content blockquote p {
    margin: 0;
}

.overview-grid .markdown-content hr {
    border: none;
    border-top: 1px solid var(--color-border-muted);
    margin: 16px 0;
}

.overview-grid .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

.overview-grid .stat-item {
    text-align: center;
    padding: 12px;
    background: var(--insight-surface-tertiary);
    border-radius: 8px;
}

.overview-grid .stat-value {
    display: block;
    font-size: 28px;
    font-weight: 700;
    color: var(--insight-action-primary);
}

.overview-grid .stat-label {
    font-size: 12px;
    color: var(--insight-text-secondary);
}

.overview-grid .loading-text {
  color: var(--insight-text-secondary);
  text-align: center;
  padding: 40px;
}

.overview-grid .export-actions {
  display: flex;
  gap: 8px;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
}

.overview-grid .overview-action-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 6px 12px;
  border: none;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.overview-grid .overview-action-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.overview-grid .overview-card .overview-action-button--primary {
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
}

.overview-grid .overview-card .overview-action-button--primary:hover:not(:disabled) {
  background: var(--insight-action-primary-strong);
}

.overview-grid .overview-card .overview-action-button--secondary {
  background: var(--insight-surface-tertiary);
  color: var(--insight-text-primary);
  border: 1px solid var(--color-border-muted);
}

.overview-grid .overview-card .overview-action-button--secondary:hover:not(:disabled) {
  background: var(--color-border-muted);
}

.overview-grid .recent-pages {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.overview-grid .recent-page-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 10px 12px;
  border: 0;
  background: var(--insight-surface-tertiary);
  border-radius: 6px;
  color: inherit;
  cursor: pointer;
  font: inherit;
  transition: all 0.2s;
}

.overview-grid .recent-page-item:hover {
  background: var(--color-focus-brand-soft);
  transform: translateX(4px);
}

.overview-grid .recent-page-item .page-number {
  font-size: 13px;
  font-weight: 500;
  color: var(--insight-action-primary);
}

.overview-grid .recent-page-item .page-summary {
  font-size: 12px;
  color: var(--insight-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 150px;
}
</style>
