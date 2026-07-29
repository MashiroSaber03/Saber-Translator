<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import { showToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'

const insightStore = useInsightStore()

const expandedChapters = ref<Set<string>>(new Set())
const pageAnalyzedMap = ref<Map<number, boolean>>(new Map())
let analyzedPagesRequestSequence = 0
let isPagesTreeMounted = true

const chapters = computed(() => insightStore.chapters)
const totalPages = computed(() => insightStore.totalPageCount)
const contentNavigationChips = computed<ProductChipItem[]>(() => [
  {
    id: 'total-pages',
    label: `${totalPages.value}页`,
    tone: 'neutral',
  },
])

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

function createPageThumbnailItems(startPage: number, endPage: number): ProductThumbnailGridItem[] {
  return getPageRange(startPage, endPage).map(pageNum => ({
    id: pageNum,
    src: getThumbnailUrl(pageNum),
    alt: `第${pageNum}页`,
    label: `第 ${pageNum} 页`,
    selected: isPageSelected(pageNum),
    marked: isPageAnalyzed(pageNum),
  }))
}

function selectThumbnailPage(pageId: string | number): void {
  selectPage(Number(pageId))
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

function chapterStateChips(chapter: { id: string; startPage: number; endPage: number }): ProductChipItem[] {
  const analyzed = isChapterAnalyzed(chapter)

  return [
    {
      id: `${chapter.id}-pages`,
      label: `${chapter.endPage - chapter.startPage + 1}页`,
      tone: 'neutral',
    },
    {
      id: `${chapter.id}-analysis`,
      label: analyzed ? '已分析' : '待分析',
      tone: analyzed ? 'success' : 'neutral',
    },
  ]
}

async function reanalyzeChapter(chapterId: string): Promise<void> {
  if (!insightStore.currentBookId) return
  const confirmed = await confirmProductAction({
    title: '重新分析章节',
    message: '确定要重新分析此章节吗？',
    confirmText: '重新分析',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

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
  <div class="pages-tree-panel">
    <ProductSectionHeader
      class="pages-tree-panel__header"
      title="内容导航"
      size="sm"
    >
      <template #actions>
        <ProductChipList
          class="pages-tree-panel__header-chips"
          aria-label="内容导航统计"
          :items="contentNavigationChips"
        />
      </template>
    </ProductSectionHeader>

    <div class="pages-tree-panel__body">
      <template v-if="chapters.length === 0">
        <ProductStatusBanner
          v-if="totalPages === 0"
          class="pages-tree-panel__empty-status"
          tone="neutral"
          role="note"
          icon-name="image"
          title="暂无页面"
        >
          导入或选择书籍后将在这里显示页面缩略图。
        </ProductStatusBanner>
        <VirtualThumbnailGrid
          v-else
          class="pages-tree-panel__all-pages"
          aria-label="所有页面导航"
          :columns="4"
          :active-id="insightStore.selectedPageNum"
          :items="createPageThumbnailItems(1, totalPages)"
          @select="selectThumbnailPage"
        />
      </template>

      <template v-else>
        <ProductRecordCard
          v-for="chapter in chapters"
          :key="chapter.id"
          class="pages-tree-panel__chapter"
          :class="{ 'pages-tree-panel__chapter--expanded': isChapterExpanded(chapter.id) }"
        >
          <template #meta>
            <div class="pages-tree-panel__chapter-main">
              <UiButton
                variant="toolbar"
                class="pages-tree-panel__chapter-toggle"
                :aria-expanded="isChapterExpanded(chapter.id)"
                :aria-label="`${isChapterExpanded(chapter.id) ? '收起' : '展开'}${chapter.title}`"
                @click="toggleChapter(chapter.id)"
              >
                <span class="pages-tree-panel__expand-icon">
                  <UiIcon name="chevron-right" size="12" stroke-width="2.5" />
                </span>
                <span class="pages-tree-panel__chapter-title">{{ chapter.title }}</span>
              </UiButton>
              <ProductChipList
                class="pages-tree-panel__chapter-chips"
                :aria-label="`${chapter.title}章节状态`"
                :items="chapterStateChips(chapter)"
              />
            </div>
          </template>

          <template #actions>
            <ProductActionRow
              :aria-label="`${chapter.title}章节操作`"
              justify="end"
            >
              <UiIconButton
                title="重新分析此章节"
                :label="`重新分析${chapter.title}`"
                variant="soft"
                size="xs"
                @click.stop="reanalyzeChapter(chapter.id)"
              >
                <UiIcon name="refresh" size="14" />
              </UiIconButton>
            </ProductActionRow>
          </template>

          <VirtualThumbnailGrid
            v-if="isChapterExpanded(chapter.id)"
            class="pages-tree-panel__pages-grid"
            :aria-label="`${chapter.title}页面导航`"
            :columns="4"
            :active-id="insightStore.selectedPageNum"
            :items="createPageThumbnailItems(chapter.startPage, chapter.endPage)"
            @select="selectThumbnailPage"
          />
        </ProductRecordCard>
      </template>
    </div>
  </div>
</template>

<style scoped>
.pages-tree-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 12px 0;
}

.pages-tree-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px 12px;
  border-bottom: 1px solid var(--color-border-muted);
  margin-bottom: 0;
}

.pages-tree-panel__body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.pages-tree-panel__chapter {
  --product-record-card-accent: var(--color-border-default);
  --product-record-card-background: var(--insight-surface-page);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-gap: 8px;
  --product-record-card-padding: 10px 12px;
  --product-record-card-radius: 8px;
  --product-record-card-shadow-hover: none;

  margin: 8px 16px;
}

.pages-tree-panel__chapter-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.pages-tree-panel__chapter-toggle {
  flex: 1 1 130px;
  display: flex;
  align-items: center;
  gap: 8px;
  width: auto;
  min-width: 0;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  user-select: none;
}

.pages-tree-panel__header-chips,
.pages-tree-panel__chapter-chips {
  flex: 0 0 auto;
}

.pages-tree-panel__expand-icon {
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--insight-text-muted);
  transition: transform 0.2s;
}

.pages-tree-panel__chapter--expanded .pages-tree-panel__expand-icon {
  transform: rotate(90deg);
}

.pages-tree-panel__chapter-title {
  min-width: 0;
  font-size: 13px;
  font-weight: 500;
  color: var(--insight-text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pages-tree-panel__pages-grid {
  padding: 6px 0 0 24px;
}

.pages-tree-panel__all-pages {
  padding: 8px 16px;
}

.pages-tree-panel__empty-status {
  margin: 8px 16px;
}

.pages-tree-panel__load-more {
  padding: 12px 16px;
}
</style>
