<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import { showToast } from '@/utils/toast'

const insightStore = useInsightStore()

const expandedChapters = ref<Set<string>>(new Set())
const pageAnalyzedMap = ref<Map<number, boolean>>(new Map())
const displayedPageCount = ref(100)
let analyzedPagesRequestSequence = 0
let isPagesTreeMounted = true

const chapters = computed(() => insightStore.chapters)
const totalPages = computed(() => insightStore.totalPageCount)

function toggleChapter(chapterId: string): void {
  if (expandedChapters.value.has(chapterId)) {
    expandedChapters.value.delete(chapterId)
  } else {
    expandedChapters.value.add(chapterId)
  }
}

function isChapterExpanded(chapterId: string): boolean {
  return expandedChapters.value.has(chapterId)
}

function selectPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

function isPageAnalyzed(pageNum: number): boolean {
  return pageAnalyzedMap.value.get(pageNum) || false
}

function isPageSelected(pageNum: number): boolean {
  return insightStore.selectedPageNum === pageNum
}

function getPageRange(startPage: number, endPage: number): number[] {
  const pages: number[] = []
  for (let i = startPage; i <= endPage; i++) {
    pages.push(i)
  }
  return pages
}

function getThumbnailUrl(pageNum: number): string {
  if (!insightStore.currentBookId) return ''
  return insightApi.getThumbnailUrl(insightStore.currentBookId, pageNum)
}

function loadMorePages(): void {
  displayedPageCount.value = Math.min(
    displayedPageCount.value + 100,
    totalPages.value
  )
}

function handleImageError(event: Event): void {
  const img = event.target as HTMLImageElement
  img.style.opacity = '0'
}

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

async function reanalyzeChapter(chapterId: string): Promise<void> {
  if (!insightStore.currentBookId) return
  if (!confirm('确定要重新分析此章节吗？')) return
  
  try {
    const response = await insightApi.reanalyzeChapter(insightStore.currentBookId, chapterId)
    if (response.success) {
      const taskId = response.task_id
      if (taskId) {
        insightStore.setCurrentTaskId(taskId)
      }
      insightStore.setAnalysisStatus('running')
      showToast('章节分析已启动', 'success')
    } else {
      showToast('启动失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch {
    showToast('重新分析失败', 'error')
  }
}

async function loadAnalyzedPages(bookId = insightStore.currentBookId): Promise<void> {
  const requestId = ++analyzedPagesRequestSequence
  if (!bookId) {
    pageAnalyzedMap.value = new Map()
    return
  }
  
  try {
    const response = await insightApi.getAnalyzedPages(bookId)
    if (!isCurrentAnalyzedPagesRequest(requestId, bookId)) return

    if (response.success && response.pages) {
      const nextMap = new Map<number, boolean>()
      const analyzedPages = response.pages
      analyzedPages.forEach(p => {
        nextMap.set(p, true)
      })
      pageAnalyzedMap.value = nextMap
    }
  } catch {
    if (!isCurrentAnalyzedPagesRequest(requestId, bookId)) return
    showToast('加载页面分析状态失败', 'error')
  }
}

function isCurrentAnalyzedPagesRequest(requestId: number, bookId: string): boolean {
  return (
    isPagesTreeMounted &&
    requestId === analyzedPagesRequestSequence &&
    insightStore.currentBookId === bookId
  )
}

onMounted(async () => {
  await loadAnalyzedPages()
  
  if (chapters.value.length > 0 && chapters.value[0]) {
    expandedChapters.value.add(chapters.value[0].id)
  }
})

watch(
  () => insightStore.analyzedPageCount,
  async (newCount, previousCount) => {
    if (newCount !== previousCount && newCount > 0) {
      pageAnalyzedMap.value.clear()
      await loadAnalyzedPages()
    }
  }
)

watch(
  () => insightStore.currentBookId,
  async (bookId) => {
    pageAnalyzedMap.value = new Map()
    await loadAnalyzedPages(bookId || '')
  }
)

onUnmounted(() => {
  isPagesTreeMounted = false
  analyzedPagesRequestSequence += 1
})
</script>

<template>
  <div class="sidebar-section pages-tree-section">
    <div class="section-header">
      <h3 class="section-title">内容导航</h3>
      <span class="page-count-badge">{{ totalPages }}页</span>
    </div>
    
    <div class="pages-tree">
      <template v-if="chapters.length === 0">
        <div v-if="totalPages === 0" class="empty-hint">
          暂无页面
        </div>
        <div v-else class="tree-all-pages">
          <UiButton
            v-for="pageNum in getPageRange(1, Math.min(totalPages, displayedPageCount))"
            :key="pageNum"
            variant="toolbar"
            class="tree-page-item"
            :class="{ 
              selected: isPageSelected(pageNum),
              analyzed: isPageAnalyzed(pageNum)
            }"
            :data-page="pageNum"
            :aria-label="`选择第 ${pageNum} 页`"
            :aria-pressed="isPageSelected(pageNum)"
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
          </UiButton>
        </div>
        <div v-if="totalPages > displayedPageCount" class="tree-load-more">
          <UiButton variant="toolbar" class="btn-load-more" @click="loadMorePages">
            加载更多 (还有 {{ totalPages - displayedPageCount }} 页)
          </UiButton>
        </div>
      </template>
      
      <template v-else>
        <div 
          v-for="chapter in chapters" 
          :key="chapter.id"
          class="tree-chapter"
          :class="{ expanded: isChapterExpanded(chapter.id) }"
        >
          <div class="tree-chapter-header">
            <UiButton
              variant="toolbar"
              class="tree-chapter-toggle"
              :aria-expanded="isChapterExpanded(chapter.id)"
              :aria-label="`${isChapterExpanded(chapter.id) ? '收起' : '展开'}${chapter.title}`"
              @click="toggleChapter(chapter.id)"
            >
              <span class="tree-expand-icon">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5l8 7-8 7z" /></svg>
              </span>
              <span class="tree-chapter-info">
                <span class="tree-chapter-title">{{ chapter.title }}</span>
                <span class="tree-chapter-meta">{{ chapter.endPage - chapter.startPage + 1 }}页</span>
              </span>
              <span
                class="tree-chapter-status"
                :class="{ analyzed: isChapterAnalyzed(chapter) }"
              ></span>
            </UiButton>
            <UiButton
              variant="toolbar" 
              class="btn-reanalyze-chapter" 
              title="重新分析此章节"
              @click.stop="reanalyzeChapter(chapter.id)"
            >
              🔄
            </UiButton>
          </div>
          
          <div class="tree-pages-grid">
            <UiButton
              v-for="pageNum in getPageRange(chapter.startPage, chapter.endPage)"
              :key="pageNum"
              variant="toolbar"
              class="tree-page-item"
              :class="{ 
                selected: isPageSelected(pageNum),
                analyzed: isPageAnalyzed(pageNum)
              }"
              :data-page="pageNum"
              :aria-label="`选择第 ${pageNum} 页`"
              :aria-pressed="isPageSelected(pageNum)"
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
            </UiButton>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.pages-tree-section {
  --pages-tree-selected-ring: rgba(99, 102, 241, .2);
  --pages-tree-page-number-gradient-end: rgba(0, 0, 0, .7);
}

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
    transition: background 0.15s;
    user-select: none;
}

.tree-chapter-header:hover {
    background: var(--insight-surface-tertiary);
}

.tree-chapter-toggle {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    border: 0;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font: inherit;
    text-align: left;
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

.tree-chapter-header:focus-within .btn-reanalyze-chapter {
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
    display: block;
    width: 100%;
    padding: 0;
    color: inherit;
    font: inherit;
    text-align: left;
    border: 2px solid transparent;
    transition: all 0.15s;
}

.tree-page-item:hover {
    border-color: var(--insight-action-primary-soft);
    transform: scale(1.02);
}

.tree-page-item.selected {
    border-color: var(--insight-action-primary);
    box-shadow: 0 0 0 2px var(--pages-tree-selected-ring);
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
    background: linear-gradient(transparent, var(--pages-tree-page-number-gradient-end));
    color: var(--color-text-inverse);
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
