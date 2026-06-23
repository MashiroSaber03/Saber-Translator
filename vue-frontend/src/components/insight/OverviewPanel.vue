<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 概览面板组件
 * 显示漫画分析的概览统计、摘要和最近分析记录
 */

import { ref, computed, onMounted, watch } from 'vue'
import { useInsightStore, type OverviewTemplateType } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { marked } from 'marked'

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 当前选中的模板类型 */
const currentTemplate = ref<OverviewTemplateType>('no_spoiler')

/** 概览内容 */
const overviewContent = ref('')

/** 是否正在加载 */
const isLoading = ref(false)

/** 已生成的模板列表 */
const generatedTemplates = ref<OverviewTemplateType[]>([])

/** 最近分析的页面 */
const recentAnalyzedPages = ref<Array<{
  page_num: number
  summary?: string
  analyzed_at?: string
}>>([])

// ============================================================
// 模板配置
// ============================================================

/** 模板选项 */
const templateOptions: Array<{ value: OverviewTemplateType; label: string; icon: string; description: string }> = [
  { value: 'no_spoiler', label: '无剧透简介', icon: '🎁', description: '不含剧透的简短介绍，适合推荐给他人' },
  { value: 'story_summary', label: '故事概要', icon: '📖', description: '完整的剧情回顾，包含所有剧透' },
  { value: 'recap', label: '前情回顾', icon: '⏪', description: '之前发生的重要事件回顾' },
  { value: 'character_guide', label: '角色图鉴', icon: '👥', description: '主要角色介绍和关系' },
  { value: 'world_setting', label: '世界观设定', icon: '🌍', description: '故事背景和世界观设定' },
  { value: 'highlights', label: '名场面盘点', icon: '✨', description: '精彩片段和经典场景回顾' },
  { value: 'reading_notes', label: '阅读笔记', icon: '📝', description: '阅读过程中的重点笔记' }
]

/** 模板选项（用于CustomSelect） */
const templateSelectOptions = templateOptions.map(t => ({
  label: `${t.icon} ${t.label}`,
  value: t.value
}))

// ============================================================
// 计算属性
// ============================================================

/** 当前模板图标 */
const currentTemplateIcon = computed(() => {
  const template = templateOptions.find(t => t.value === currentTemplate.value)
  return template?.icon || '📊'
})

/** 当前模板描述 */
const currentTemplateDescription = computed(() => {
  const template = templateOptions.find(t => t.value === currentTemplate.value)
  return template?.description || ''
})

/** 模板状态文本 */
const templateStatus = computed(() => {
  if (generatedTemplates.value.includes(currentTemplate.value)) {
    return '已生成'
  }
  return ''
})

/** 渲染后的概览内容 */
const renderedContent = computed(() => {
  if (!overviewContent.value) return ''
  return marked.parse(overviewContent.value) as string
})

// ============================================================
// 方法
// ============================================================

/**
 * 模板变更处理 - 只读取缓存，不触发生成
 * 与接口流程 的 onOverviewTemplateChange 一致
 */
async function onTemplateChange(): Promise<void> {
  await loadCachedOverview()
}

/**
 * 加载缓存的概览内容（不触发生成）
 * 与接口流程 的 loadTemplateOverview 一致：GET /overview/{templateKey}
 */
async function loadCachedOverview(): Promise<void> {
  if (!insightStore.currentBookId) return

  isLoading.value = true
  overviewContent.value = ''

  try {
    // 使用 GET API 只读取缓存
    const response = await insightApi.getOverview(
      insightStore.currentBookId,
      currentTemplate.value
    ) as any

    if (response.success && response.content) {
      // 有缓存内容
      overviewContent.value = response.content
      // 更新已生成模板列表
      if (!generatedTemplates.value.includes(currentTemplate.value)) {
        generatedTemplates.value.push(currentTemplate.value)
      }
    } else {
      // 无缓存，显示提示
      overviewContent.value = ''
    }
  } catch (error) {
    console.error('加载概览失败:', error)
    overviewContent.value = '加载失败，请重试'
  } finally {
    isLoading.value = false
  }
}

/**
 * 生成概览（点击按钮时调用）
 * 与接口流程 的 generateOverviewWithTemplate 一致：POST /overview/generate
 * @param regenerate - 是否强制重新生成（🔄按钮为true，📄按钮为false）
 */
async function generateOverview(regenerate: boolean): Promise<void> {
  if (!insightStore.currentBookId) return

  isLoading.value = true
  overviewContent.value = ''

  try {
    // 使用 POST API 生成概览
    const response = await insightApi.regenerateOverview(
      insightStore.currentBookId,
      currentTemplate.value,
      regenerate  // force 参数
    ) as any

    if (response.success) {
      if (response.content) {
        overviewContent.value = response.content
        // 更新已生成模板列表
        if (!generatedTemplates.value.includes(currentTemplate.value)) {
          generatedTemplates.value.push(currentTemplate.value)
        }
        if (currentTemplate.value === 'story_summary' && response.cached !== true) {
          insightStore.triggerDataRefresh()
        }
      }
    } else {
      overviewContent.value = `生成失败: ${response.error || '未知错误'}`
    }
  } catch (error) {
    console.error('生成概览失败:', error)
    overviewContent.value = '生成失败，请重试'
  } finally {
    isLoading.value = false
  }
}

/**
 * 加载已生成的模板列表
 * 默认选中 no_spoiler，不因为后端自动生成 story_summary 切换用户当前选择。
 */
async function loadGeneratedTemplates(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    const response = await insightApi.getGeneratedTemplates(insightStore.currentBookId)
    if (response.success) {
      // API返回的是generated字段，不是templates
      let templates: OverviewTemplateType[] = []
      if (response.generated) {
        templates = response.generated as OverviewTemplateType[]
      } else if (response.templates && Array.isArray(response.templates)) {
        templates = response.templates as OverviewTemplateType[]
      }
      generatedTemplates.value = templates

      // 用户可以在下拉框中自行选择其他已生成的模板。
    }
  } catch (error) {
    console.error('加载模板列表失败:', error)
  }
}

/** 是否正在导出 */
const isExporting = ref(false)

/**
 * 导出完整分析数据
 */
async function exportAnalysisData(): Promise<void> {
  if (!insightStore.currentBookId) {
    alert('请先选择书籍')
    return
  }

  isExporting.value = true

  try {
    const response = await insightApi.exportAnalysis(insightStore.currentBookId) as any

    if (response.success && response.markdown) {
      // 下载 Markdown 文件
      const blob = new Blob([response.markdown], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${insightStore.currentBookId}_analysis.md`
      a.click()
      URL.revokeObjectURL(url)

      alert('导出成功')
    } else {
      alert('导出失败: ' + (response.error || '未知错误'))
    }
  } catch (error) {
    console.error('导出失败:', error)
    alert('导出失败')
  } finally {
    isExporting.value = false
  }
}

/**
 * 导出当前概览内容
 */
function exportCurrentOverview(): void {
  if (!overviewContent.value) {
    alert('暂无内容可导出')
    return
  }

  const template = templateOptions.find(t => t.value === currentTemplate.value)
  const fileName = `${insightStore.currentBookId}_${currentTemplate.value}.md`

  // 构建 Markdown 内容
  const content = `# ${template?.label || currentTemplate.value}\n\n${overviewContent.value}`

  const blob = new Blob([content], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = fileName
  a.click()
  URL.revokeObjectURL(url)
}

/**
 * 加载最近分析的页面
 */
async function loadRecentAnalyzedPages(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    // 获取最近分析的页面 (显示最多5个)
    const stats = await insightApi.getAnalysisStatus(insightStore.currentBookId)
    if (stats.success && insightStore.analyzedPageCount > 0) {
      // 从已分析页数倒推获取最近的几页
      const totalPages = insightStore.totalPageCount
      const analyzedCount = insightStore.analyzedPageCount
      const recentPages: Array<{ page_num: number; summary?: string }> = []

      // 简单实现：显示最后分析的5页
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

      recentAnalyzedPages.value = recentPages.reverse() // 最新的在前
    }
  } catch (error) {
    console.error('加载最近分析页面失败:', error)
  }
}

/**
 * 跳转到指定页面
 * @param pageNum - 页码
 */
function goToPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  await loadGeneratedTemplates()
  await loadRecentAnalyzedPages()
  // 如果当前模板已生成，自动加载
  if (generatedTemplates.value.includes(currentTemplate.value)) {
    await loadCachedOverview()
  }
})

// 监听书籍ID变化，重新加载概览数据
watch(() => insightStore.currentBookId, async (newBookId) => {
  if (newBookId) {
    overviewContent.value = ''
    generatedTemplates.value = []
    recentAnalyzedPages.value = []
    await loadGeneratedTemplates()
    await loadRecentAnalyzedPages()
    // 如果当前模板已生成，自动加载
    if (generatedTemplates.value.includes(currentTemplate.value)) {
      await loadCachedOverview()
    }
  }
})

// 监听数据刷新触发器（分析完成后自动刷新）
watch(() => insightStore.dataRefreshKey, async (newKey) => {
  if (newKey > 0 && insightStore.currentBookId) {
    console.log('OverviewPanel: 收到刷新信号，重新加载数据')
    await loadGeneratedTemplates()
    await loadRecentAnalyzedPages()
    // 自动加载已生成的模板内容
    if (generatedTemplates.value.includes(currentTemplate.value)) {
      await loadCachedOverview()
    }
  }
})
</script>

<template>
  <div class="overview-grid">
    <!-- 摘要卡片 -->
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

    <!-- 统计卡片 -->
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

      <!-- 导出按钮 -->
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

    <!-- 最近分析卡片 -->
    <div class="overview-card recent-card">
      <h3 class="card-title">🕐 最近分析</h3>
      <div class="recent-pages">
        <div v-if="recentAnalyzedPages.length === 0" class="placeholder-text">暂无分析记录</div>
        <div
          v-for="page in recentAnalyzedPages"
          :key="page.page_num"
          class="recent-page-item"
          @click="goToPage(page.page_num)"
        >
          <span class="page-number">第 {{ page.page_num }} 页</span>
          <span v-if="page.summary" class="page-summary">{{ page.summary }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overview-grid {
  --overview-panel-text-secondary: #ef4444;
}

/* ==================== 概览面板样式 - 当前样式 ==================== */

/* ==================== 组件样式 ==================== */

/* 概览网格 */
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

/* 模板选择器样式 */
.overview-grid .card-title-with-selector {
    display: flex;
    align-items: center;
    gap: 8px;
}

.overview-grid .card-title-icon {
    font-size: 20px;
    line-height: 1;
}

.overview-grid .template-select {
    padding: 6px 12px;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid var(--color-border-muted);
    border-radius: 6px;
    background: var(--insight-surface-secondary);
    color: var(--insight-text-primary);
    cursor: pointer;
    min-width: 140px;
    transition: all 0.2s;
}

.overview-grid .template-select:hover {
    border-color: var(--insight-action-primary);
}

.overview-grid .template-select:focus {
    outline: none;
    border-color: var(--insight-action-primary);
    box-shadow: 0 0 0 2px var(--overview-panel-shadow-default);
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

.overview-grid .template-status.status-cached {
    background: var(--overview-panel-surface-base);
    color: var(--overview-panel-text-primary);
}

.overview-grid .template-status.status-empty {
    background: var(--overview-panel-surface-raised);
    color: var(--text-tertiary);
}

.overview-grid .template-status.status-generating {
    background: var(--color-focus-brand-soft);
    color: var(--insight-action-primary);
    animation: pulse 1.5s infinite;
}

.overview-grid .template-status.status-error {
    background: var(--overview-panel-surface-muted);
    color: var(--overview-panel-text-secondary);
}

.overview-grid .template-description {
    font-size: 12px;
    color: var(--text-tertiary);
    margin: 0 0 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--color-border-muted);
}

.overview-grid .placeholder-text.generating {
    color: var(--insight-action-primary);
    animation: pulse 1.5s infinite;
}

.overview-grid .placeholder-text.error {
    color: var(--overview-panel-text-secondary);
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
    color: white;
}

.overview-grid .card-content {
    color: var(--insight-text-secondary);
    line-height: 1.6;
}

/* Markdown 渲染样式 */
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
  color: white;
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

/* 最近分析页面项 */
.overview-grid .recent-pages {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.overview-grid .recent-page-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  background: var(--insight-surface-tertiary);
  border-radius: 6px;
  cursor: pointer;
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
