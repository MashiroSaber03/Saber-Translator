<script setup lang="ts">
import AppShell from '@/components/ui/AppShell.vue'
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductThemeToggle from '@/components/product/ProductThemeToggle.vue'
import ProductTabbedWorkspace from '@/components/product/ProductTabbedWorkspace.vue'
import ProductThreePaneWorkspace from '@/components/product/ProductThreePaneWorkspace.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useInsightStore } from '@/stores/insightStore'
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
import {
  projectInsightPageProgress,
  projectTerminalInsightPageProgress,
  type InsightTerminalEventType,
} from '@/utils/insightJobProgress'
import { stepKindLabel } from '@/utils/taskDisplay'
import type { BookData } from '@/types'
import { showToast } from '@/utils/toast'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'

const route = useRoute()
const router = useRouter()
const insightStore = useInsightStore()
const taskCenterStore = useTaskCenterStore()
const publicAccess = usePublicUserAccess()

type InsightTabId = 'overview' | 'qa' | 'timeline' | 'continuation' | 'character_studio'

const allInsightTabs: Array<{ id: InsightTabId; label: string; glyph: string }> = [
  { id: 'overview', label: '概览', glyph: '📊' },
  { id: 'qa', label: '智能问答', glyph: '💬' },
  { id: 'timeline', label: '时间线', glyph: '📈' },
  { id: 'continuation', label: '续写', glyph: '🎨' },
  { id: 'character_studio', label: '角色工坊', glyph: '🃏' },
]
const canUseCharacterStudio = computed(() => publicAccess.featureAllowed('characterStudio'))
const insightTabs = computed(() => allInsightTabs.filter(tab => (
  tab.id !== 'character_studio' || canUseCharacterStudio.value
)))

function insightTabGlyph(tabId: string): string {
  return insightTabs.value.find(tab => tab.id === tabId)?.glyph ?? ''
}

const activeTab = ref<InsightTabId>('overview')
const showSettingsModal = ref(false)
const showMobileSidebar = ref(false)
const showMobileWorkspace = ref(false)
let bookLoadSequence = 0
let isInsightViewMounted = false
let terminalProjectionPrimed = false
const observedTerminalKeys = new Set<string>()

const loadedBookDetail = ref<{
  id: string
  title: string
  cover?: string
  totalPages: number
} | null>(null)

const showChapterSelectModal = ref(false)

const currentBook = computed(() => {
  if (loadedBookDetail.value?.id === insightStore.currentBookId) return loadedBookDetail.value
  return null
})

const hasLoadedBook = computed(() => currentBook.value !== null)

const bookCoverUrl = computed(() => {
  if (!currentBook.value?.cover) return ''
  return currentBook.value.cover
})

function switchTab(tab: string): void {
  const selectedTab = insightTabs.value.find(item => item.id === tab)
  if (selectedTab) activeTab.value = selectedTab.id
}

function applyLoadedBook(
  book: BookData,
  chapters: Awaited<ReturnType<typeof insightApi.getInsightChapters>>,
): void {
  loadedBookDetail.value = {
    id: book.id,
    title: book.title,
    cover: book.cover,
    totalPages: book.totalPages ?? 0,
  }
  insightStore.setBookTotalPages(book.totalPages ?? 0)
  insightStore.setChapters(chapters)
}

function applyAnalysisStatus(
  response: Awaited<ReturnType<typeof insightApi.getAnalysisStatus>>,
): void {
  insightStore.setAnalyzedPagesCount(response.analyzedPagesCount)
  insightStore.setAnalysisStatus(resolveAnalysisStatus(response))

  if (response.currentTask) {
    insightStore.setCurrentTaskId(response.currentTask.jobId)
    insightStore.updateProgress(
      response.currentTask.progress.analyzedPages,
      response.currentTask.progress.totalPages,
    )
    return
  }

  insightStore.setCurrentTaskId(null)
  insightStore.updateProgress(0, 0)
}

function isCurrentBookLoad(loadId: number, bookId: string): boolean {
  return isInsightViewMounted && loadId === bookLoadSequence && insightStore.currentBookId === bookId
}

async function loadBook(bookId: string): Promise<void> {
  if (!bookId) return

  const loadId = ++bookLoadSequence
  showChapterSelectModal.value = false
  showMobileSidebar.value = false
  showMobileWorkspace.value = false
  loadedBookDetail.value = null
  observedTerminalKeys.clear()
  terminalProjectionPrimed = false
  insightStore.setCurrentBook(bookId)
  insightStore.setLoading(true)

  try {
    const [book, analysisStatus, chapters] = await Promise.all([
      getBookDetail(bookId),
      insightApi.getAnalysisStatus(bookId),
      insightApi.getInsightChapters(bookId),
    ])
    if (!isCurrentBookLoad(loadId, bookId)) return

    applyLoadedBook(book, chapters)
    applyAnalysisStatus(analysisStatus)
    primeTerminalJobs(bookId)
    projectActiveInsightJob()
    void insightStore.loadNotesFromAPI()
    void router.replace({ query: { book: bookId } })

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

async function refreshAnalysisFacts(bookId: string): Promise<void> {
  try {
    const response = await insightApi.getAnalysisStatus(bookId)
    if (!isInsightViewMounted || insightStore.currentBookId !== bookId) return
    insightStore.setAnalyzedPagesCount(response.analyzedPagesCount)
  } catch {
    // The durable task projection remains authoritative for task state.
  }
}

function projectActiveInsightJob(): void {
  const bookId = insightStore.currentBookId
  if (
    !bookId
    || !isInsightViewMounted
  ) return
  const jobs = [...taskCenterStore.queue, ...taskCenterStore.history]
  let refreshDomainData = false
  let refreshAnalysisStatus = false
  for (const job of jobs) {
    if (
      job.bookId !== bookId
      || job.status === 'interrupted'
      || !['cancelled', 'completed', 'completed_with_errors', 'failed'].includes(job.status)
      || ![
        'insight_analysis',
        'derived_rebuild',
        'vector_rebuild',
        'continuation',
      ].includes(job.kind)
    ) continue
    const key = `${bookId}:${job.jobId}:${job.status}`
    if (observedTerminalKeys.has(key)) continue
    observedTerminalKeys.add(key)
    if (terminalProjectionPrimed) {
      refreshDomainData = true
      if (job.kind === 'insight_analysis') refreshAnalysisStatus = true
    }
  }
  if (refreshDomainData) insightStore.triggerDataRefresh()
  if (refreshAnalysisStatus) void refreshAnalysisFacts(bookId)
  const matchesBookAnalysis = (job: (typeof taskCenterStore.queue)[number]): boolean => (
    job.bookId === bookId && job.kind === 'insight_analysis'
  )
  const active = taskCenterStore.queue.find(matchesBookAnalysis)
    ?? taskCenterStore.history.find(job => (
      matchesBookAnalysis(job) && job.status === 'interrupted'
    ))
  if (active) {
    insightStore.setCurrentTaskId(active.jobId)
    insightStore.setAnalysisStatus(active.status)
    const progress = projectInsightPageProgress(active.progress)
    insightStore.updateProgress(
      progress.current,
      progress.total,
      progress.currentStepKind ? stepKindLabel(progress.currentStepKind) : undefined,
    )
    return
  }

  const terminal = taskCenterStore.history.find(job => (
    matchesBookAnalysis(job) && job.status !== 'interrupted'
  ))
  if (!terminal) return
  const shouldProject = (
    insightStore.currentTaskId === terminal.jobId
    || ['queued', 'running', 'paused', 'interrupted'].includes(
      insightStore.analysisStatus,
    )
  )
  if (!shouldProject) return
  const eventType: InsightTerminalEventType = terminal.status === 'cancelled'
    ? 'job_cancelled'
    : terminal.status === 'failed'
      ? 'job_failed'
      : 'job_finished'
  const progress = projectTerminalInsightPageProgress(
    terminal.progress,
    eventType,
  )
  insightStore.setAnalysisStatus(terminal.status)
  insightStore.setCurrentTaskId(null)
  insightStore.updateProgress(progress.current, progress.total)
}

function primeTerminalJobs(bookId: string): void {
  observedTerminalKeys.clear()
  for (const job of [...taskCenterStore.queue, ...taskCenterStore.history]) {
    if (
      job.bookId === bookId
      && job.status !== 'interrupted'
      && ['cancelled', 'completed', 'completed_with_errors', 'failed'].includes(job.status)
      && [
        'insight_analysis',
        'derived_rebuild',
        'vector_rebuild',
        'continuation',
      ].includes(job.kind)
    ) {
      observedTerminalKeys.add(`${bookId}:${job.jobId}:${job.status}`)
    }
  }
  terminalProjectionPrimed = true
}

function routeBookId(value: unknown): string {
  return typeof value === 'string' ? value : ''
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

  if (chapters.length === 0) {
    showToast('当前书籍还没有章节，请先在书架中创建章节', 'warning')
  } else if (chapters.length === 1) {
    const onlyChapter = chapters[0]
    if (!onlyChapter) return
    router.push({
      path: '/translate',
      query: {
        book: insightStore.currentBookId,
        chapter: onlyChapter.id
      }
    })
  } else {
    showChapterSelectModal.value = true
  }
}

function handleChapterSelect(chapterId: string): void {
  const bookId = insightStore.currentBookId
  showChapterSelectModal.value = false
  if (!bookId || !insightStore.chapters.some(chapter => chapter.id === chapterId)) return
  router.push({
    path: '/translate',
    query: {
      book: bookId,
      chapter: chapterId
    }
  })
}

function closeChapterSelectModal(): void {
  showChapterSelectModal.value = false
}

function retryCurrentBook(): void {
  if (insightStore.currentBookId) void loadBook(insightStore.currentBookId)
}

function resetBookSelection(): void {
  bookLoadSequence += 1
  loadedBookDetail.value = null
  showChapterSelectModal.value = false
  showMobileSidebar.value = false
  showMobileWorkspace.value = false
  observedTerminalKeys.clear()
  terminalProjectionPrimed = false
  insightStore.setCurrentBook(null)
  insightStore.setLoading(false)
}

function chooseAnotherBook(): void {
  resetBookSelection()
  void router.replace({ query: {} })
}

onMounted(() => {
  isInsightViewMounted = true
  const bookId = routeBookId(route.query.book)
  if (bookId) {
    void loadBook(bookId)
  } else {
    resetBookSelection()
  }
})

onUnmounted(() => {
  observedTerminalKeys.clear()
  terminalProjectionPrimed = false
  isInsightViewMounted = false
  bookLoadSequence += 1
  insightStore.setLoading(false)
})

watch(
  () => routeBookId(route.query.book),
  bookId => {
    if (!isInsightViewMounted) return
    if (!bookId) {
      resetBookSelection()
      return
    }
    if (bookId !== insightStore.currentBookId) void loadBook(bookId)
  },
)

watch(
  [
    () => taskCenterStore.queue,
    () => taskCenterStore.history,
    () => insightStore.currentBookId,
  ],
  projectActiveInsightJob,
  { immediate: true },
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
      :show-right="hasLoadedBook"
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
              <span aria-hidden="true">📖</span>
            </div>
          </div>
          <h2 class="insight-view__book-title" :title="currentBook?.title">{{ currentBook?.title || '选择书籍' }}</h2>
          <div class="insight-view__book-meta">
            <span class="insight-view__book-meta-item">
              <span class="insight-view__book-meta-icon" aria-hidden="true">📄</span>
              <span>{{ currentBook?.totalPages ?? 0 }}</span> 页
            </span>
            <span class="insight-view__book-meta-item">
              <span class="insight-view__book-meta-icon" aria-hidden="true">📊</span>
              <span>{{ insightStore.analyzedPageCount }}</span> 已分析
            </span>
          </div>
        </div>

        <AnalysisProgress v-if="hasLoadedBook" />

        <PagesTree v-if="hasLoadedBook" />
      </template>

      <div class="insight-view__content">
        <ProductStatusBanner
          v-if="insightStore.isLoading"
          class="insight-view__load-status"
          tone="neutral"
          title="正在加载书籍"
          aria-live="polite"
        >
          正在读取书籍详情与分析状态…
        </ProductStatusBanner>

        <ProductStatusBanner
          v-else-if="insightStore.error"
          class="insight-view__load-status"
          tone="danger"
          title="书籍加载失败"
          aria-live="assertive"
        >
          {{ insightStore.error }}
          <template #actions>
            <UiButton size="sm" variant="secondary" @click="retryCurrentBook">
              重试
            </UiButton>
            <UiButton size="sm" variant="ghost" @click="chooseAnotherBook">
              选择其他书籍
            </UiButton>
          </template>
        </ProductStatusBanner>

        <div v-else-if="!hasLoadedBook" class="insight-view__select-book-prompt">
          <span class="insight-view__select-book-icon" aria-hidden="true">📚</span>
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
          <template #tabIcon="{ tab }">{{ insightTabGlyph(tab.id) }}</template>
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

          <div
            v-show="activeTab === 'overview'"
            id="product-workspace-panel-overview"
            class="insight-view__tab-content"
            role="tabpanel"
            aria-labelledby="product-workspace-tab-overview"
          >
            <OverviewPanel />
          </div>

          <div
            v-show="activeTab === 'qa'"
            id="product-workspace-panel-qa"
            class="insight-view__tab-content"
            role="tabpanel"
            aria-labelledby="product-workspace-tab-qa"
          >
            <QAPanel />
          </div>

          <div
            v-show="activeTab === 'timeline'"
            id="product-workspace-panel-timeline"
            class="insight-view__tab-content"
            role="tabpanel"
            aria-labelledby="product-workspace-tab-timeline"
          >
            <TimelinePanel />
          </div>

          <div
            v-show="activeTab === 'continuation'"
            id="product-workspace-panel-continuation"
            class="insight-view__tab-content"
            role="tabpanel"
            aria-labelledby="product-workspace-tab-continuation"
          >
            <ContinuationPanel />
          </div>

          <div
            v-if="canUseCharacterStudio"
            v-show="activeTab === 'character_studio'"
            id="product-workspace-panel-character_studio"
            class="insight-view__tab-content"
            role="tabpanel"
            aria-labelledby="product-workspace-tab-character_studio"
          >
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

      <template #right>
        <PageDetail />

        <NotesPanel />
      </template>
    </ProductThreePaneWorkspace>

    <InsightSettingsModal
      v-if="showSettingsModal"
      @close="closeSettingsModal"
    />

    <ChapterSelectModal
      v-if="showChapterSelectModal && hasLoadedBook"
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
  --product-tabbed-workspace-tab-background-active: var(--color-action-primary);
}

.insight-view__load-status {
  margin: 16px 20px 0;
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
