<script setup lang="ts">
import AppShell from '@/components/ui/AppShell.vue'
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'
import ProductThemeToggle from '@/components/product/ProductThemeToggle.vue'
import ProductTabbedWorkspace from '@/components/product/ProductTabbedWorkspace.vue'
import ProductThreePaneWorkspace from '@/components/product/ProductThreePaneWorkspace.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useInsightStore } from '@/stores/insightStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import BookSelector from '@/components/insight/BookSelector.vue'
import AnalysisProgress from '@/components/insight/AnalysisProgress.vue'
import OverviewPanel from '@/components/insight/OverviewPanel.vue'
import TimelinePanel from '@/components/insight/TimelinePanel.vue'
import QAPanel from '@/components/insight/QAPanel.vue'
import NotesPanel from '@/components/insight/NotesPanel.vue'
import PageDetail from '@/components/insight/PageDetail.vue'
import PagesTree from '@/components/insight/PagesTree.vue'
import InsightSettingsModal from '@/components/insight/InsightSettingsModal.vue'
import ChapterSelectModal from '@/components/insight/ChapterSelectModal.vue'
import ContinuationPanel from '@/components/insight/ContinuationPanel.vue'
import CharacterStudioEntryPanel from '@/components/insight/CharacterStudioEntryPanel.vue'
import * as insightApi from '@/api/insight'
import { getBookDetail } from '@/api/bookshelf'
import { resolveAnalysisStatus } from '@/utils/insightStatus'
import { projectInsightPageProgress } from '@/utils/insightJobProgress'
import type { BookData, ChapterData, ChapterInfo } from '@/types'

const route = useRoute()
const router = useRouter()
const insightStore = useInsightStore()
const bookshelfStore = useBookshelfStore()
const taskCenterStore = useTaskCenterStore()

type InsightTabId = 'overview' | 'qa' | 'timeline' | 'continuation' | 'character_studio'

const insightTabs: Array<{ id: InsightTabId; label: string; iconName: UiIconName }> = [
  { id: 'overview', label: '概览', iconName: 'bar-chart' },
  { id: 'qa', label: '智能问答', iconName: 'message' },
  { id: 'timeline', label: '时间线', iconName: 'clock' },
  { id: 'continuation', label: '续写', iconName: 'palette' },
  { id: 'character_studio', label: '角色工坊', iconName: 'book-marked' },
]

const activeTab = ref<InsightTabId>('overview')
const showSettingsModal = ref(false)
const showMobileSidebar = ref(false)
const showMobileWorkspace = ref(false)
let bookLoadSequence = 0
let isInsightViewMounted = false
let lastHandledTerminalEventId = 0

const loadedBookDetail = ref<{
  id: string
  title: string
  cover?: string
  totalPages: number
} | null>(null)

const showChapterSelectModal = ref(false)

const currentBook = computed(() => {
  if (loadedBookDetail.value) return loadedBookDetail.value
  if (!insightStore.currentBookId) return null
  return bookshelfStore.books.find(b => b.id === insightStore.currentBookId)
})

const hasSelectedBook = computed(() => !!insightStore.currentBookId)

const bookCoverUrl = computed(() => {
  if (!currentBook.value?.cover) return ''
  return currentBook.value.cover
})

function switchTab(tab: string): void {
  if (insightTabs.some(item => item.id === tab)) {
    activeTab.value = tab as InsightTabId
  }
}

function mapBookChaptersToInsightChapters(chapters: ChapterData[]): ChapterInfo[] {
  let pageOffset = 0
  return chapters.map((chapter, index) => {
    const pageCount = chapter.imageCount ?? 0
    const startPage = pageOffset + 1
    const endPage = pageOffset + pageCount
    pageOffset = endPage
    return {
      id: chapter.id,
      title: chapter.title || `第 ${index + 1} 章`,
      startPage,
      endPage,
      analyzed: false,
    }
  })
}

function setLoadedBookDetail(book: BookData): void {
  loadedBookDetail.value = {
    id: book.id,
    title: book.title,
    cover: book.cover,
    totalPages: book.totalPages ?? 0,
  }
  insightStore.setBookTotalPages(book.totalPages ?? 0)

  if (book.chapters?.length) {
    insightStore.setChapters(mapBookChaptersToInsightChapters(book.chapters))
  }
}

function isCurrentBookLoad(loadId: number, bookId: string): boolean {
  return isInsightViewMounted && loadId === bookLoadSequence && insightStore.currentBookId === bookId
}

async function loadBook(bookId: string): Promise<void> {
  if (!bookId) return

  const loadId = ++bookLoadSequence
  insightStore.setCurrentBook(bookId)
  insightStore.setLoading(true)

  try {
    const book = await getBookDetail(bookId)
    if (!isCurrentBookLoad(loadId, bookId)) return
    setLoadedBookDetail(book)

    await loadAnalysisStatus(bookId)
    if (!isCurrentBookLoad(loadId, bookId)) return

    if (insightStore.chapters.length === 0) {
      try {
        const chapters = await insightApi.getInsightChapters(bookId)
        if (!isCurrentBookLoad(loadId, bookId)) return
        if (chapters.length > 0) {
          insightStore.setChapters(chapters.map(chapter => ({
            id: chapter.id,
            title: chapter.title,
            startPage: chapter.start_page,
            endPage: chapter.end_page,
            analyzed: chapter.analyzed,
          })))
        }
      } catch {
        // 章节补充接口不可用时保留书籍详情中的章节状态。
      }
    }

    await insightStore.loadNotesFromAPI()
    if (!isCurrentBookLoad(loadId, bookId)) return

    router.replace({ query: { book: bookId } })

  } catch (error) {
    if (isCurrentBookLoad(loadId, bookId)) {
      insightStore.setError(error instanceof Error ? error.message : '加载书籍失败')
    }
  } finally {
    if (isCurrentBookLoad(loadId, bookId)) {
      insightStore.setLoading(false)
    }
  }
}

async function loadAnalysisStatus(bookId = insightStore.currentBookId): Promise<void> {
  if (!bookId) return

  try {
    const response = await insightApi.getAnalysisStatus(bookId)
    if (!isInsightViewMounted || insightStore.currentBookId !== bookId) return
    insightStore.setAnalyzedPagesCount(response.analyzedPagesCount)

    const resolvedStatus = resolveAnalysisStatus(response)
    insightStore.setAnalysisStatus(resolvedStatus)

    if (response.currentTask) {
      insightStore.setCurrentTaskId(response.currentTask.jobId)
      insightStore.updateProgress(
        response.currentTask.progress.analyzedPages,
        response.currentTask.progress.totalPages,
      )
    }
  } catch {
    // 保持最近一次后端快照；全局任务流仍可继续投影活动任务。
  }
}

function projectActiveInsightJob(): void {
  const bookId = insightStore.currentBookId
  if (!bookId) return
  const active = taskCenterStore.queue.find(job => (
    job.bookId === bookId
    && job.kind === 'insight_analysis'
    && job.status !== 'interrupted'
  ))
  if (!active) return
  insightStore.setCurrentTaskId(active.jobId)
  insightStore.setAnalysisStatus(
    active.status === 'paused' ? 'paused' : 'running',
  )
  const progress = projectInsightPageProgress(active.progress)
  insightStore.updateProgress(
    progress.current,
    progress.total,
    progress.currentStepKind,
  )
}

function openSettingsModal(): void {
  showSettingsModal.value = true
}

function closeSettingsModal(): void {
  showSettingsModal.value = false
}

function toggleMobileSidebar(): void {
  showMobileSidebar.value = !showMobileSidebar.value
  if (showMobileSidebar.value) {
    showMobileWorkspace.value = false
  }
}

function toggleMobileWorkspace(): void {
  showMobileWorkspace.value = !showMobileWorkspace.value
  if (showMobileWorkspace.value) {
    showMobileSidebar.value = false
  }
}

function goToTranslate(): void {
  if (!insightStore.currentBookId) {
    router.push('/translate')
    return
  }

  const chapters = insightStore.chapters

  if (!chapters || chapters.length === 0) {
    router.push({ path: '/translate', query: { book: insightStore.currentBookId } })
  } else if (chapters.length === 1) {
    router.push({
      path: '/translate',
      query: {
        book: insightStore.currentBookId,
        chapter: chapters[0]!.id
      }
    })
  } else {
    showChapterSelectModal.value = true
  }
}

function handleChapterSelect(chapterId: string): void {
  showChapterSelectModal.value = false
  router.push({
    path: '/translate',
    query: {
      book: insightStore.currentBookId!,
      chapter: chapterId
    }
  })
}

function closeChapterSelectModal(): void {
  showChapterSelectModal.value = false
}

onMounted(async () => {
  isInsightViewMounted = true
  await bookshelfStore.loadBooks()
  if (!isInsightViewMounted) return

  const bookId = route.query.book as string
  if (bookId) {
    await loadBook(bookId)
  }
})

onUnmounted(() => {
  isInsightViewMounted = false
  bookLoadSequence += 1
})

watch(
  [
    () => taskCenterStore.queue,
    () => insightStore.currentBookId,
  ],
  projectActiveInsightJob,
  { immediate: true },
)

watch(
  () => taskCenterStore.latestEvent,
  async event => {
    if (
      !event
      || event.eventId <= lastHandledTerminalEventId
      || !['job_finished', 'job_failed', 'job_cancelled'].includes(event.type)
    ) {
      return
    }
    const relatedJob = [...taskCenterStore.queue, ...taskCenterStore.history]
      .find(job => job.jobId === event.jobId)
    if (!relatedJob || relatedJob.bookId !== insightStore.currentBookId) return
    lastHandledTerminalEventId = event.eventId
    if (['derived_rebuild', 'vector_rebuild', 'continuation'].includes(relatedJob.kind)) {
      insightStore.triggerDataRefresh()
      return
    }
    if (
      relatedJob.kind !== 'insight_analysis'
      || event.jobId !== insightStore.currentTaskId
    ) {
      return
    }
    const terminalStatus = event.type === 'job_finished'
      ? 'completed'
      : event.type === 'job_cancelled'
        ? 'idle'
        : 'failed'
    await loadAnalysisStatus()
    if (!isInsightViewMounted) return
    insightStore.setAnalysisStatus(terminalStatus)
    insightStore.setCurrentTaskId(null)
    insightStore.triggerDataRefresh()
  },
)
</script>

<template>
  <AppShell class="insight-page" viewport-mode="locked">
    <ProductPageHeader
      variant="fixed"
      logo-title="书架首页"
      nav-label="漫画分析导航"
      actions-label="漫画分析操作"
    >
      <template #nav>
        <ProductHeaderAction
          as="router-link"
          to="/"
          class="insight-header__nav-link"
          icon-name="book-open"
          label="书架"
          collapse-label-on-mobile
        />
        <ProductHeaderAction
          class="insight-header__nav-link"
          icon-name="globe"
          label="翻译"
          collapse-label-on-mobile
          @click="goToTranslate"
        />
        <ProductHeaderAction
          as="span"
          class="insight-header__nav-link insight-header__nav-link--active"
          icon-name="search"
          label="分析"
          active
          collapse-label-on-mobile
        />
        <ProductHeaderAction
          as="a"
          href="https://www.mashirosaber.top/use/manga-insight.html"
          target="_blank"
          rel="noopener noreferrer"
          class="insight-header__nav-link"
          title="使用教程"
          icon-name="file-text"
          label="教程"
          collapse-label-on-mobile
        />
      </template>

      <template #actions>
        <ProductHeaderAction
          title="设置"
          aria-label="设置"
          icon-name="settings"
          icon-only
          @click="openSettingsModal"
        />
        <ProductThemeToggle class="insight-header__theme-toggle" />
      </template>
    </ProductPageHeader>

    <ProductThreePaneWorkspace
      as="main"
      class="insight-view__main"
      aria-label="漫画分析三栏工作区"
      left-width="280px"
      right-width="320px"
      mobile-mode="drawer"
      :left-mobile-visible="showMobileSidebar"
      :right-mobile-visible="showMobileWorkspace"
    >
      <template #left>
        <div class="insight-view__book-summary">
          <div class="insight-view__book-cover-frame">
            <img
              v-if="bookCoverUrl"
              :src="bookCoverUrl"
              :alt="`${currentBook?.title || '书籍'}封面`"
              class="insight-view__book-cover"
            >
            <div v-else class="insight-view__book-cover-placeholder">
              <UiIcon name="book-open" size="26" />
            </div>
          </div>
          <h2 class="insight-view__book-title" :title="currentBook?.title">{{ currentBook?.title || '选择书籍' }}</h2>
          <div class="insight-view__book-meta">
            <span class="insight-view__book-meta-item">
              <UiIcon name="file-text" class="insight-view__book-meta-icon" />
              <span>{{ currentBook?.totalPages || 0 }}</span> 页
            </span>
            <span class="insight-view__book-meta-item">
              <UiIcon name="bar-chart" class="insight-view__book-meta-icon" />
              <span>{{ insightStore.analyzedPageCount }}</span> 已分析
            </span>
          </div>
        </div>

        <AnalysisProgress v-if="hasSelectedBook" />

        <PagesTree v-if="hasSelectedBook" />
      </template>

      <div class="insight-view__content">
        <div v-if="!hasSelectedBook" class="insight-view__select-book-prompt">
          <UiIcon name="book-open" class="insight-view__select-book-icon" size="42" />
          <h2 class="insight-view__select-book-title">选择要分析的书籍</h2>
          <p class="insight-view__select-book-description">从下方列表中选择一本书籍开始智能分析</p>
          <BookSelector @select="loadBook" />
        </div>

        <ProductTabbedWorkspace
          v-else
          class="insight-view__tabbed-workspace"
          :tabs="insightTabs"
          :active-tab="activeTab"
          aria-label="漫画分析工作区"
          @select="switchTab"
        >
          <template #beforeTabs>
            <UiIconButton
              class="insight-view__mobile-nav-button"
              label="打开导航"
              :active="showMobileSidebar"
              :pressed="showMobileSidebar"
              @click="toggleMobileSidebar"
            >
              <UiIcon name="book-open" />
            </UiIconButton>
          </template>

          <div v-if="activeTab === 'overview' && hasSelectedBook" class="insight-view__tab-content">
            <OverviewPanel />
          </div>

          <div v-else-if="activeTab === 'qa' && hasSelectedBook" class="insight-view__tab-content">
            <QAPanel />
          </div>

          <div v-else-if="activeTab === 'timeline' && hasSelectedBook" class="insight-view__tab-content">
            <TimelinePanel />
          </div>

          <div v-else-if="activeTab === 'continuation' && hasSelectedBook" class="insight-view__tab-content">
            <ContinuationPanel />
          </div>

          <div v-else-if="activeTab === 'character_studio' && hasSelectedBook" class="insight-view__tab-content">
            <CharacterStudioEntryPanel />
          </div>

          <template #afterTabs>
            <UiIconButton
              class="insight-view__mobile-nav-button"
              label="打开笔记"
              :active="showMobileWorkspace"
              :pressed="showMobileWorkspace"
              @click="toggleMobileWorkspace"
            >
              <UiIcon name="file-text" />
            </UiIconButton>
          </template>
        </ProductTabbedWorkspace>
      </div>

      <template v-if="hasSelectedBook" #right>
        <PageDetail />

        <NotesPanel />
      </template>
    </ProductThreePaneWorkspace>

    <InsightSettingsModal
      v-if="showSettingsModal"
      @close="closeSettingsModal"
    />

    <ChapterSelectModal
      v-if="showChapterSelectModal && insightStore.currentBookId"
      :chapters="insightStore.chapters"
      @select="handleChapterSelect"
      @close="closeChapterSelectModal"
    />
  </AppShell>
</template>

<style scoped>
.insight-page {
  --insight-view-sidebar-divider: var(--color-border-muted);

  overflow: hidden;
  margin: 0;
  padding: 56px 20px 0 20px;
  display: flex;
  flex-direction: column;
}

.insight-view__main {
  flex: 1;
}

.insight-view__content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}

.insight-view__tabbed-workspace {
  --product-tabbed-workspace-bar-background: var(--insight-surface-secondary);
  --product-tabbed-workspace-border: var(--insight-view-sidebar-divider);
  --product-tabbed-workspace-tab-text: var(--insight-text-secondary);
  --product-tabbed-workspace-tab-background-hover: var(--insight-surface-tertiary);
  --product-tabbed-workspace-tab-background-active: var(--insight-action-primary);
}

.insight-view__mobile-nav-button {
  display: none;
  flex-shrink: 0;
}

.insight-view__tab-content {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  padding: 20px;
}

.insight-view__select-book-prompt {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  text-align: center;
}

.insight-view__select-book-icon {
  margin-bottom: 16px;
  font-size: 64px;
}

.insight-view__select-book-title {
  margin-bottom: 8px;
  color: var(--insight-text-primary);
}

.insight-view__select-book-description {
  margin-bottom: 24px;
  color: var(--insight-text-secondary);
}

@media (--breakpoint-lg-down) {
  .insight-view__mobile-nav-button {
    display: inline-flex;
  }
}

.insight-view__book-summary {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 0;
  text-align: center;
  border-bottom: 1px solid var(--insight-view-sidebar-divider);
}

.insight-view__book-cover-frame {
  width: 120px;
  height: 160px;
  margin: 0 auto 12px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--insight-surface-tertiary);
  position: relative;
}

.insight-view__book-cover {
  width: 100%;
  height: 100%;
  max-width: 120px;
  max-height: 160px;
  object-fit: cover;
  display: block;
}

.insight-view__book-cover-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  color: var(--insight-text-muted);
}

.insight-view__book-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--insight-text-primary);
  margin: 0 0 10px 0;
  text-align: center;
  max-width: 100%;
  word-break: break-word;
  line-height: 1.4;
}

.insight-view__book-meta {
  display: flex;
  justify-content: center;
  gap: 16px;
  font-size: 13px;
  color: var(--insight-text-secondary);
  flex-wrap: wrap;
}

.insight-view__book-meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.insight-view__book-meta-icon {
  font-size: 14px;
}

</style>
