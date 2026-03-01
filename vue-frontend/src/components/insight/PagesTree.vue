<script setup lang="ts">
/**
 * 页面导航树组件
 * 显示章节和页面的树状结构，支持展开/折叠和页面选择
 */

import { ref, computed, onMounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import { getLocalWriteHeaders } from '@/api/localToken'

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
  if (!confirm('确定要重新分析此章节吗？')) return
  
  try {
    const headers = await getLocalWriteHeaders()
    const response = await fetch(
      `/api/manga-insight/${insightStore.currentBookId}/reanalyze/chapter/${chapterId}`,
      { method: 'POST', headers }
    )
    const data = await response.json()
    if (data.success) {
      alert('章节分析已启动')
    } else {
      alert('启动失败: ' + (data.error || '未知错误'))
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
 * 【修复】原版在分析完成后会调用 renderPagesTree 重新渲染
 * Vue 版通过监听 analyzedPageCount 变化自动刷新
 */
watch(
  () => insightStore.analyzedPageCount,
  async (newCount, oldCount) => {
    // 当已分析页数变化时，重新加载页面分析状态
    if (newCount !== oldCount && newCount > 0) {
      console.log(`已分析页数变化: ${oldCount} -> ${newCount}，刷新页面标记`)
      // 清空现有标记并重新加载
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
          <button class="btn-load-more" @click="loadMorePages">
            加载更多 (还有 {{ totalPages - displayedPageCount }} 页)
          </button>
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
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5l8 7-8 7z"/></svg>
            </span>
            <div class="tree-chapter-info">
              <span class="tree-chapter-title">{{ chapter.title }}</span>
              <span class="tree-chapter-meta">{{ chapter.endPage - chapter.startPage + 1 }}页</span>
            </div>
            <span 
              class="tree-chapter-status" 
              :class="{ analyzed: isChapterAnalyzed(chapter) }"
            ></span>
            <button 
              class="btn-reanalyze-chapter" 
              title="重新分析此章节"
              @click.stop="reanalyzeChapter(chapter.id)"
            >
              🔄
            </button>
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
/* ==================== PagesTree 完整样式 ==================== */

/* ==================== CSS变量 ==================== */
.pages-tree-section {
  --bg-primary: #f8fafc;
  --bg-secondary: #fff;
  --bg-tertiary: #f1f5f9;
  --bg-hover: rgb(99, 102, 241, 0.1);
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --color-primary: #6366f1;
  --primary-light: #818cf8;
  --primary-dark: #4f46e5;
  --success-color: #22c55e;
  --success: #22c55e;
  --warning-color: #f59e0b;
  --error-color: #ef4444;
}

/* ==================== 组件特定样式 ==================== */

.tree-chapter-header {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.tree-chapter-header:hover {
  background-color: var(--bg-hover);
}

.tree-expand-icon {
  width: 16px;
  font-size: 10px;
  color: var(--text-secondary);
}

.tree-chapter-title {
  flex: 1;
  font-weight: 500;
}

.tree-chapter-pages {
  font-size: 12px;
  color: var(--text-secondary);
}

.tree-pages-list {
  padding-left: 16px;
}

.tree-page-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.tree-page-item:hover {
  background-color: var(--bg-hover);
}

.tree-page-item.selected {
  background-color: var(--primary-light);
}

.tree-page-thumbnail {
  width: 32px;
  height: 40px;
  object-fit: cover;
  border-radius: 2px;
}

.tree-page-num {
  flex: 1;
  font-size: 13px;
}

.tree-page-status {
  color: var(--success);
  font-size: 12px;
}

/* ==================== 页面树完整样式 - 从 manga-insight.css 迁移 ==================== */

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
    border-bottom: 1px solid var(--border-color);
}

.pages-tree-section .section-title {
    margin: 0;
    font-size: 13px;
}

.page-count-badge {
    font-size: 11px;
    padding: 2px 8px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
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
    background: var(--bg-tertiary);
}

.tree-chapter-header.active {
    background: rgb(99, 102, 241, 0.1);
}

.tree-expand-icon {
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
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
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.tree-chapter-meta {
    font-size: 11px;
    color: var(--text-muted);
    flex-shrink: 0;
}

.tree-chapter-status {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
}

.tree-chapter-status.analyzed {
    background: var(--success-color);
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

.btn-reanalyze-chapter:hover {
    opacity: 1;
}

.tree-pages-grid {
    display: none;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    padding: 8px 16px 8px 40px;
    background: var(--bg-primary);
}

.tree-chapter.expanded .tree-pages-grid {
    display: grid;
}

.tree-page-item {
    aspect-ratio: 3/4;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
    cursor: pointer;
    position: relative;
    border: 2px solid transparent;
    transition: all 0.15s;
}

.tree-page-item:hover {
    border-color: var(--primary-light);
    transform: scale(1.02);
}

.tree-page-item.selected {
    border-color: var(--color-primary);
    box-shadow: 0 0 0 2px rgb(99, 102, 241, 0.2);
}

.tree-page-item.analyzed::after {
    content: '';
    position: absolute;
    top: 3px;
    right: 3px;
    width: 12px;
    height: 12px;
    background: var(--success-color);
    border-radius: 50%;
    border: 1.5px solid var(--bg-primary);
}

.tree-page-thumb {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    background: var(--bg-tertiary);
}

.tree-page-num {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 2px 4px;
    background: linear-gradient(transparent, rgb(0,0,0,0.7));
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
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.btn-load-more:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
}
</style>
