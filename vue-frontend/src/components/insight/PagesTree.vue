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
import type { V2InsightPageSummary } from '@/api/v2/insight'
import { showToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'

const insightStore = useInsightStore()

const expandedChapters = ref<Set<string>>(new Set())
interface PageListState {
  items: V2InsightPageSummary[]
  loading: boolean
  nextCursor: number | null
}
const GLOBAL_PAGE_LIST = '__all__'
const pageLists = ref<Map<string, PageListState>>(new Map())
const reanalyzingChapterId = ref<string | null>(null)
let pageListGeneration = 0
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
    void loadPageList(chapterId)
  }
}

function isChapterExpanded(chapterId: string): boolean {
  return expandedChapters.value.has(chapterId)
}

function selectPage(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

function isPageSelected(pageNum: number): boolean {
  return insightStore.selectedPageNum === pageNum
}

function pageListKey(chapterId?: string): string {
  return chapterId || GLOBAL_PAGE_LIST
}

function pageList(chapterId?: string): PageListState {
  return pageLists.value.get(pageListKey(chapterId)) ?? {
    items: [],
    loading: false,
    nextCursor: 0,
  }
}

function createPageThumbnailItems(chapterId?: string): ProductThumbnailGridItem[] {
  return pageList(chapterId).items.map(page => ({
    id: page.displayPageNumber,
    src: page.thumbnailUrl ?? '',
    alt: `第${page.displayPageNumber}页`,
    label: `第 ${page.displayPageNumber} 页`,
    selected: isPageSelected(page.displayPageNumber),
    marked: isPublishedAnalysisState(page.analysisState),
  }))
}

function isPublishedAnalysisState(state: V2InsightPageSummary['analysisState']): boolean {
  return state === 'ready' || state === 'stale'
}

async function loadPageList(chapterId?: string, reset = false): Promise<void> {
  const bookId = insightStore.currentBookId
  const key = pageListKey(chapterId)
  const current = pageList(chapterId)
  if (!bookId || current.loading || (!reset && current.nextCursor === null)) return
  const generation = pageListGeneration
  const nextState: PageListState = {
    items: reset ? [] : current.items,
    loading: true,
    nextCursor: reset ? 0 : current.nextCursor,
  }
  pageLists.value = new Map(pageLists.value).set(key, nextState)
  try {
    const response = await insightApi.getInsightPagesPage(bookId, {
      ...(chapterId ? { chapterId } : {}),
      cursor: nextState.nextCursor ?? 0,
      limit: 100,
    })
    if (
      !isPagesTreeMounted
      || generation !== pageListGeneration
      || insightStore.currentBookId !== bookId
    ) return
    const known = new Set(nextState.items.map(page => page.pageId))
    pageLists.value = new Map(pageLists.value).set(key, {
      items: [...nextState.items, ...response.items.filter(page => !known.has(page.pageId))],
      loading: false,
      nextCursor: response.nextCursor,
    })
  } catch {
    if (generation !== pageListGeneration || insightStore.currentBookId !== bookId) return
    pageLists.value = new Map(pageLists.value).set(key, { ...nextState, loading: false })
    showToast('加载页面列表失败', 'error')
  }
}

function selectThumbnailPage(pageId: string | number): void {
  selectPage(Number(pageId))
}

type ChapterAnalysisState = 'none' | 'partial' | 'completed'

function chapterAnalysisState(chapter: {
  id: string
  analyzed?: boolean
  analyzedCount?: number
}): ChapterAnalysisState {
  const state = pageLists.value.get(chapter.id)
  if (state && state.nextCursor === null && state.items.length > 0) {
    const analyzedCount = state.items.filter(page => isPublishedAnalysisState(page.analysisState)).length
    if (analyzedCount === state.items.length) return 'completed'
    return analyzedCount > 0 ? 'partial' : 'none'
  }
  if (chapter.analyzed) return 'completed'
  return Number(chapter.analyzedCount) > 0 ? 'partial' : 'none'
}

function chapterStateChips(chapter: {
  id: string
  startPage: number
  endPage: number
  analyzed?: boolean
  analyzedCount?: number
}): ProductChipItem[] {
  const analysisState = chapterAnalysisState(chapter)
  const pageCount = chapter.startPage > 0 && chapter.endPage >= chapter.startPage
    ? chapter.endPage - chapter.startPage + 1
    : 0

  return [
    {
      id: `${chapter.id}-pages`,
      label: `${pageCount}页`,
      tone: 'neutral',
    },
    {
      id: `${chapter.id}-analysis`,
      label: analysisState === 'completed'
        ? '已分析'
        : analysisState === 'partial'
          ? '部分分析'
          : '待分析',
      tone: analysisState === 'completed'
        ? 'success'
        : analysisState === 'partial'
          ? 'warning'
          : 'neutral',
    },
  ]
}

async function reanalyzeChapter(chapterId: string): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId || reanalyzingChapterId.value !== null) return
  reanalyzingChapterId.value = chapterId

  try {
    const confirmed = await confirmProductAction({
      title: '重新分析章节',
      message: '确定要重新分析此章节吗？',
      confirmText: '重新分析',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed || insightStore.currentBookId !== bookId) return
    const submission = await insightApi.reanalyzeChapter(bookId, chapterId)
    if (insightStore.currentBookId !== bookId) return
    insightStore.setCurrentTaskId(submission.jobId)
    insightStore.setAnalysisStatus('queued')
    showToast('章节分析已启动', 'success')
  } catch (error) {
    if (insightStore.currentBookId === bookId) {
      showToast(error instanceof Error ? error.message : '重新分析失败', 'error')
    }
  } finally {
    if (reanalyzingChapterId.value === chapterId) reanalyzingChapterId.value = null
  }
}

async function loadInitialPageList(): Promise<void> {
  if (!insightStore.currentBookId) return
  if (chapters.value.length > 0 && chapters.value[0]) {
    expandedChapters.value.add(chapters.value[0].id)
    await loadPageList(chapters.value[0].id)
  } else if (totalPages.value > 0) {
    await loadPageList()
  }
}

async function refreshLoadedPageLists(): Promise<void> {
  const loadedKeys = [...pageLists.value.keys()]
  if (loadedKeys.length === 0) return

  pageListGeneration += 1
  pageLists.value = new Map(
    [...pageLists.value].map(([key, state]) => [key, { ...state, loading: false }])
  )
  await Promise.all(
    loadedKeys.map(key => loadPageList(key === GLOBAL_PAGE_LIST ? undefined : key, true))
  )
}

onMounted(async () => {
  await loadInitialPageList()
})

watch(
  [() => insightStore.analyzedPageCount, () => insightStore.dataRefreshKey],
  async ([newCount, newRefreshKey], [previousCount, previousRefreshKey]) => {
    if (newCount !== previousCount || newRefreshKey !== previousRefreshKey) {
      await refreshLoadedPageLists()
    }
  }
)

watch(
  () => insightStore.currentBookId,
  async () => {
    pageListGeneration += 1
    pageLists.value = new Map()
    expandedChapters.value = new Set()
    await loadInitialPageList()
  }
)

watch(
  () => chapters.value.map(chapter => chapter.id).join('\u0000'),
  async () => {
    if (pageLists.value.size === 0) await loadInitialPageList()
  }
)

onUnmounted(() => {
  isPagesTreeMounted = false
  pageListGeneration += 1
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
          :items="createPageThumbnailItems()"
          @select="selectThumbnailPage"
        />
        <UiButton
          v-if="totalPages > 0 && pageList().nextCursor !== null"
          class="pages-tree-panel__load-more"
          variant="secondary"
          size="sm"
          :disabled="pageList().loading"
          @click="loadPageList()"
        >
          {{ pageList().loading ? '加载中...' : '加载更多页面' }}
        </UiButton>
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
                :disabled="reanalyzingChapterId !== null"
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
            :items="createPageThumbnailItems(chapter.id)"
            @select="selectThumbnailPage"
          />
          <UiButton
            v-if="isChapterExpanded(chapter.id) && pageList(chapter.id).nextCursor !== null"
            class="pages-tree-panel__load-more"
            variant="secondary"
            size="sm"
            :disabled="pageList(chapter.id).loading"
            @click="loadPageList(chapter.id)"
          >
            {{ pageList(chapter.id).loading ? '加载中...' : '加载更多页面' }}
          </UiButton>
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
