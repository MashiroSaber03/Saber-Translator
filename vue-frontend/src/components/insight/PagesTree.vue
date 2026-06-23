<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 页面导航树组件
 * 显示章节和页面的树状结构，支持展开/折叠和页面选择
 */

import { ref, computed, onMounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 展开的章节ID集合 */
const expandedChapters = ref<Set<string>>(new Set())

/** 页面分析状态映射 */
const pageAnalyzedMap = ref<Map<number, boolean>>(new Map())

/** 已显示的页面数量（无章节模式下分页用） */
const displayedPageCount = ref(100)

// ============================================================
// 计算属性
// ============================================================

/** 章节列表 */
const chapters = computed(() => insightStore.chapters)

/** 总页数 */
const totalPages = computed(() => insightStore.totalPageCount)

// ============================================================
// 方法
// ============================================================

/**
 * 切换章节展开状态
 * @param chapterId - 章节ID
 */
function toggleChapter(chapterId: string): void {
  if (expandedChapters.value.has(chapterId)) {
    expandedChapters.value.delete(chapterId)
  } else {
    expandedChapters.value.add(chapterId)
  }
}

/**
 * 检查章节是否展开
 * @param chapterId - 章节ID
 */
function isChapterExpanded(chapterId: string): boolean {
  return expandedChapters.value.has(chapterId)
}

/**
 * 选择页面
 * @param pageNum - 页码
 */
function selectPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

/**
 * 检查页面是否已分析
 * @param pageNum - 页码
 */
function isPageAnalyzed(pageNum: number): boolean {
  return pageAnalyzedMap.value.get(pageNum) || false
}

/**
 * 检查页面是否被选中
 * @param pageNum - 页码
 */
function isPageSelected(pageNum: number): boolean {
  return insightStore.selectedPageNum === pageNum
}

/**
 * 获取章节的页面范围数组
 * @param startPage - 起始页
 * @param endPage - 结束页
 */
function getPageRange(startPage: number, endPage: number): number[] {
  const pages: number[] = []
  for (let i = startPage; i <= endPage; i++) {
    pages.push(i)
  }
  return pages
}

/**
 * 获取缩略图URL
 * @param pageNum - 页码
 */
function getThumbnailUrl(pageNum: number): string {
  if (!insightStore.currentBookId) return ''
  return insightApi.getThumbnailUrl(insightStore.currentBookId, pageNum)
}

/**
 * 加载更多页面（无章节模式下分页）
 */
function loadMorePages(): void {
  displayedPageCount.value = Math.min(
    displayedPageCount.value + 100,
    totalPages.value
  )
}

/**
 * 处理图片加载错误
 * @param event - 错误事件
 */
function handleImageError(event: Event): void {
  const img = event.target as HTMLImageElement
  img.style.opacity = '0'
}

/**
 * 检查章节是否已完全分析
 * @param chapter - 章节信息
 */
function isChapterAnalyzed(chapter: { startPage: number; endPage: number }): boolean {
  const pageCount = chapter.endPage - chapter.startPage + 1
  let analyzedCount = 0
  for (let p = chapter.startPage; p <= chapter.endPage; p++) {
    if (pageAnalyzedMap.value.get(p)) {
      analyzedCount++
    }
  }
  return analyzedCount === pageCount
}

/**
 * 重新分析章节
 * @param chapterId - 章节ID
 */
async function reanalyzeChapter(chapterId: string): Promise<void> {
  if (!insightStore.currentBookId) return
  if (!confirm('确定要重新分析此章节吗？')) return
  
  try {
    const response = await insightApi.reanalyzeChapter(insightStore.currentBookId, chapterId)
    if (response.success) {
      const taskId = (response as any).task_id
      if (taskId) {
        insightStore.setCurrentTaskId(taskId)
      }
      insightStore.setAnalysisStatus('running')
      alert('章节分析已启动')
    } else {
      alert('启动失败: ' + (response.error || '未知错误'))
    }
  } catch (error) {
    console.error('重新分析章节失败:', error)
    alert('重新分析失败')
  }
}

/**
 * 加载已分析页面列表
 */
async function loadAnalyzedPages(): Promise<void> {
  if (!insightStore.currentBookId) return
  
  try {
    const response = await fetch(`/api/manga-insight/${insightStore.currentBookId}/pages`)
    const data = await response.json()
    if (data.success && data.pages) {
      const analyzedPages = data.pages as number[]
      analyzedPages.forEach(p => {
        pageAnalyzedMap.value.set(p, true)
      })
    }
  } catch (error) {
    console.error('加载已分析页面失败:', error)
  }
}

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  // 加载已分析页面
  await loadAnalyzedPages()
  
  // 默认展开第一个章节
  if (chapters.value.length > 0 && chapters.value[0]) {
    expandedChapters.value.add(chapters.value[0].id)
  }
})

/**
 * 监听分析进度变化，自动刷新已分析页面标记
 * 通过监听 analyzedPageCount 变化刷新页面分析状态。
 */
watch(
  () => insightStore.analyzedPageCount,
  async (newCount, oldCount) => {
    // 当已分析页数变化时，重新加载页面分析状态
    if (newCount !== oldCount && newCount > 0) {
      console.log(`已分析页数变化: ${oldCount} -> ${newCount}，刷新页面标记`)
      // 清空现有标记并重新加载。
      pageAnalyzedMap.value.clear()
      await loadAnalyzedPages()
    }
  }
)
</script>

<template>
  <div class="sidebar-section pages-tree-section">
    <div class="section-header">
      <h3 class="section-title">内容导航</h3>
      <span class="page-count-badge">{{ totalPages }}页</span>
    </div>
    
    <div class="pages-tree">
      <!-- 无章节时显示提示或直接显示页面网格 -->
      <template v-if="chapters.length === 0">
        <div v-if="totalPages === 0" class="empty-hint">
          暂无页面
        </div>
        <!-- 无章节时直接显示页面网格 -->
        <div v-else class="tree-all-pages">
          <div 
            v-for="pageNum in getPageRange(1, Math.min(totalPages, displayedPageCount))"
            :key="pageNum"
            class="tree-page-item"
            :class="{ 
              selected: isPageSelected(pageNum),
              analyzed: isPageAnalyzed(pageNum)
            }"
            :data-page="pageNum"
            @click="selectPage(pageNum)"
          >
            <img 
              :src="getThumbnailUrl(pageNum)" 
              :alt="`第${pageNum}页`"
              class="tree-page-thumb"
              loading="lazy"
              @error="handleImageError($event)"
            >
            <span class="tree-page-num">{{ pageNum }}</span>
          </div>
        </div>
        <!-- 加载更多按钮 -->
        <div v-if="totalPages > displayedPageCount" class="tree-load-more">
          <UiButton variant="toolbar" class="btn-load-more" @click="loadMorePages">
            加载更多 (还有 {{ totalPages - displayedPageCount }} 页)
          </UiButton>
        </div>
      </template>
      
      <!-- 有章节时：按章节组织 -->
      <template v-else>
        <div 
          v-for="chapter in chapters" 
          :key="chapter.id"
          class="tree-chapter"
          :class="{ expanded: isChapterExpanded(chapter.id) }"
        >
          <!-- 章节标题 -->
          <div 
            class="tree-chapter-header"
            @click="toggleChapter(chapter.id)"
          >
            <span class="tree-expand-icon">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5l8 7-8 7z" /></svg>
            </span>
            <div class="tree-chapter-info">
              <span class="tree-chapter-title">{{ chapter.title }}</span>
              <span class="tree-chapter-meta">{{ chapter.endPage - chapter.startPage + 1 }}页</span>
            </div>
            <span 
              class="tree-chapter-status" 
              :class="{ analyzed: isChapterAnalyzed(chapter) }"
            ></span>
            <UiButton
              variant="toolbar" 
              class="btn-reanalyze-chapter" 
              title="重新分析此章节"
              @click.stop="reanalyzeChapter(chapter.id)"
            >
              🔄
            </UiButton>
          </div>
          
          <!-- 章节页面网格（4列） -->
          <div class="tree-pages-grid">
            <div 
              v-for="pageNum in getPageRange(chapter.startPage, chapter.endPage)"
              :key="pageNum"
              class="tree-page-item"
              :class="{ 
                selected: isPageSelected(pageNum),
                analyzed: isPageAnalyzed(pageNum)
              }"
              :data-page="pageNum"
              @click="selectPage(pageNum)"
            >
              <img 
                :src="getThumbnailUrl(pageNum)" 
                :alt="`第${pageNum}页`"
                class="tree-page-thumb"
                loading="lazy"
                @error="handleImageError($event)"
              >
              <span class="tree-page-num">{{ pageNum }}</span>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.pages-tree-section {
  --pages-tree-shadow-default: rgba(99, 102, 241, .2);
  --pages-tree-surface-base: rgba(0, 0, 0, .7);
}

/* ==================== PagesTree样式 ==================== */

/* ==================== 页面树样式 ==================== */

.pages-tree-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    padding: 12px 0;
}

.pages-tree-section .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px 12px;
    border-bottom: 1px solid var(--color-border-muted);
}

.pages-tree-section .section-title {
    margin: 0;
    font-size: 13px;
}

.page-count-badge {
    font-size: 11px;
    padding: 2px 8px;
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-secondary);
    border-radius: 10px;
}

.pages-tree {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
}

.tree-chapter {
    margin-bottom: 2px;
}

.tree-chapter-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    cursor: pointer;
    transition: background 0.15s;
    user-select: none;
}

.tree-chapter-header:hover {
    background: var(--insight-surface-tertiary);
}

.tree-chapter-header.active {
    background: var(--color-focus-brand-soft);
}

.tree-expand-icon {
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--insight-text-muted);
    transition: transform 0.2s;
}

.tree-chapter.expanded .tree-expand-icon {
    transform: rotate(90deg);
}

.tree-chapter-info {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}

.tree-chapter-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--insight-text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.tree-chapter-meta {
    font-size: 11px;
    color: var(--insight-text-muted);
    flex-shrink: 0;
}

.tree-chapter-status {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--insight-text-muted);
    flex-shrink: 0;
}

.tree-chapter-status.analyzed {
    background: var(--insight-status-success);
}

.btn-reanalyze-chapter {
    background: none;
    border: none;
    cursor: pointer;
    padding: 2px 6px;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
    flex-shrink: 0;
}

.tree-chapter-header:hover .btn-reanalyze-chapter {
    opacity: 0.6;
}

.tree-chapter-header:hover .btn-reanalyze-chapter:hover {
    opacity: 1;
}

.tree-pages-grid {
    display: none;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    padding: 8px 16px 8px 40px;
    background: var(--insight-surface-page);
}

.tree-chapter.expanded .tree-pages-grid {
    display: grid;
}

.tree-page-item {
    aspect-ratio: 3/4;
    background: var(--insight-surface-tertiary);
    border-radius: 4px;
    overflow: hidden;
    cursor: pointer;
    position: relative;
    border: 2px solid transparent;
    transition: all 0.15s;
}

.tree-page-item:hover {
    border-color: var(--insight-action-primary-soft);
    transform: scale(1.02);
}

.tree-page-item.selected {
    border-color: var(--insight-action-primary);
    box-shadow: 0 0 0 2px var(--pages-tree-shadow-default);
}

.tree-page-item.analyzed::after {
    content: '';
    position: absolute;
    top: 3px;
    right: 3px;
    width: 12px;
    height: 12px;
    background: var(--insight-status-success);
    border-radius: 50%;
    border: 1.5px solid var(--insight-surface-page);
}

.tree-page-thumb {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    background: var(--insight-surface-tertiary);
}

.tree-page-num {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 2px 4px;
    background: linear-gradient(transparent, var(--pages-tree-surface-base));
    color: white;
    font-size: 10px;
    text-align: center;
}

.tree-all-pages {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    padding: 8px 16px;
}

.tree-load-more {
    padding: 12px 16px;
    text-align: center;
}

.btn-load-more {
    padding: 6px 16px;
    font-size: 12px;
    background: var(--insight-surface-tertiary);
    border: 1px solid var(--color-border-muted);
    border-radius: 6px;
    color: var(--insight-text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.btn-load-more:hover {
    background: var(--insight-surface-secondary);
    color: var(--insight-text-primary);
}
</style>
