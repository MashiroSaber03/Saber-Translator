<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import AppShell from '@/components/ui/AppShell.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useInsightStore } from '@/stores/insightStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'
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
import AppHeader from '@/components/common/AppHeader.vue'
import * as insightApi from '@/api/insight'
import { getBookDetail } from '@/api/bookshelf'
import { showToast } from '@/utils/toast'
import { resolveAnalysisStatus } from '@/utils/insightStatus'
import type { BookData, ChapterData, ChapterInfo } from '@/types'

const route = useRoute()
const router = useRouter()
const insightStore = useInsightStore()
const bookshelfStore = useBookshelfStore()

const activeTab = ref<'overview' | 'qa' | 'timeline' | 'continuation' | 'character_studio'>('overview')
const showSettingsModal = ref(false)
const showMobileSidebar = ref(false)
const showMobileWorkspace = ref(false)
let statusPollingTimer: ReturnType<typeof setInterval> | null = null
let refreshDataTimer: ReturnType<typeof setTimeout> | null = null
let bookLoadSequence = 0
let isInsightViewMounted = false

const loadedBookDetail = ref<{
  id: string
  title: string
  cover?: string
  total_pages: number
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

function switchTab(tab: 'overview' | 'qa' | 'timeline' | 'continuation' | 'character_studio'): void {
  activeTab.value = tab
}

function getChapterPageCount(chapter: ChapterData): number {
  return chapter.page_count ?? chapter.image_count ?? chapter.imageCount ?? 0
}

function mapBookChaptersToInsightChapters(chapters: ChapterData[]): ChapterInfo[] {
  let pageOffset = 0
  return chapters.map((chapter, index) => {
    const pageCount = getChapterPageCount(chapter)
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
    total_pages: book.total_pages || 0,
  }
  insightStore.setBookTotalPages(book.total_pages || 0)

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
    const bookData = await getBookDetail(bookId)
    if (!isCurrentBookLoad(loadId, bookId)) return

    if (!bookData.success) {
      throw new Error(bookData.error || '获取书籍信息失败')
    }

    if (bookData.book) {
      setLoadedBookDetail(bookData.book)
    }

    await loadAnalysisStatus(bookId)
    if (!isCurrentBookLoad(loadId, bookId)) return

    if (insightStore.chapters.length === 0) {
      try {
        const chaptersResponse = await insightApi.getInsightChapters(bookId)
        if (!isCurrentBookLoad(loadId, bookId)) return
        if (chaptersResponse.success && chaptersResponse.chapters && chaptersResponse.chapters.length > 0) {
          insightStore.setChapters(chaptersResponse.chapters.map(ch => ({
            id: ch.id,
            title: ch.title,
            startPage: ch.start_page,
            endPage: ch.end_page,
            analyzed: true
          })))
        }
      } catch {
        // 章节补充接口不可用时保留书籍详情中的章节状态。
      }
    }

    await insightStore.loadNotesFromAPI()
    if (!isCurrentBookLoad(loadId, bookId)) return

    router.replace({ query: { book: bookId } })

    if (insightStore.isAnalyzing) {
      startStatusPolling()
    }

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
    if (response.success) {
      if (response.analyzed_pages_count !== undefined) {
        insightStore.setAnalyzedPagesCount(response.analyzed_pages_count)
      }

      const resolvedStatus = resolveAnalysisStatus(response)
      insightStore.setAnalysisStatus(resolvedStatus)

      if (resolvedStatus === 'running' && response.current_task?.progress) {
        insightStore.updateProgress(
          response.current_task.progress.analyzed_pages || 0,
          response.current_task.progress.total_pages || 0
        )
      }
    }
  } catch {
    // 轮询失败时保持上一次可用状态，下一轮继续尝试。
  }
}

function startStatusPolling(): void {
  stopStatusPolling()
  statusPollingTimer = setInterval(async () => {
    const statusBeforePolling = insightStore.analysisStatus
    await loadAnalysisStatus()
    
    const status = insightStore.analysisStatus
    const wasActiveTask = statusBeforePolling === 'running' || statusBeforePolling === 'paused'
    if ((status === 'completed' || status === 'failed' || status === 'idle') && wasActiveTask) {
      stopStatusPolling()

      const refreshData = async () => {
        await loadAnalysisStatus()
        insightStore.triggerDataRefresh()
      }

      if (status === 'completed') {
        refreshDataTimer = setTimeout(() => {
          refreshDataTimer = null
          void refreshData()
        }, 1000)
      } else {
        await refreshData()
      }
    }
  }, 3000)
}

function stopStatusPolling(): void {
  if (statusPollingTimer) {
    clearInterval(statusPollingTimer)
    statusPollingTimer = null
  }
  if (refreshDataTimer) {
    clearTimeout(refreshDataTimer)
    refreshDataTimer = null
  }
}

function openSettingsModal(): void {
  showSettingsModal.value = true
}

function showFeatureNotice(): void {
  showToast('🌙 该功能正在开发中，敬请期待！', 'info')
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
  stopStatusPolling()
})

watch(() => insightStore.isAnalyzing, (isAnalyzing) => {
  if (isAnalyzing) {
    startStatusPolling()
  } else {
    stopStatusPolling()
  }
})
</script>

<template>
  <AppShell class="insight-page" viewport-mode="locked">
    <AppHeader variant="insight" logo-title="书架首页">
      <template #header-links>
        <router-link to="/" class="insight-header__nav-link">📚 书架</router-link>
        <UiButton variant="link" class="insight-header__nav-link" @click="goToTranslate">🌐 翻译</UiButton>
        <span class="insight-header__nav-link insight-header__nav-link--active">🔍 分析</span>
        <a href="https://www.mashirosaber.top/use/manga-insight.html" target="_blank" rel="noopener noreferrer" class="insight-header__nav-link" title="使用教程">📖 教程</a>
        <UiButton variant="toolbar" id="settingsBtn" class="insight-settings-action" title="设置" @click="openSettingsModal">⚙️</UiButton>
        <UiButton variant="toolbar" id="themeToggle" class="insight-header__theme-toggle" title="功能开发中" @click="showFeatureNotice">
          <span class="insight-header__theme-icon">☀️</span>
        </UiButton>
      </template>
    </AppHeader>

    <SidebarLayout as="main" class="insight-main">
      <aside class="insight-sidebar" :class="{ 'mobile-visible': showMobileSidebar }">
        <div class="sidebar-section book-info-section">
          <div class="book-cover-wrapper">
            <img
              v-if="bookCoverUrl"
              :src="bookCoverUrl"
              :alt="`${currentBook?.title || '书籍'}封面`"
              class="book-cover"
            >
            <div v-else class="book-cover-placeholder">
              <span>📖</span>
            </div>
          </div>
          <h2 class="insight-book-title" :title="currentBook?.title">{{ currentBook?.title || '选择书籍' }}</h2>
          <div class="book-meta">
            <span class="meta-item">
              <span class="meta-icon">📄</span> 
              <span id="totalPages">{{ currentBook?.total_pages || 0 }}</span> 页
            </span>
            <span class="meta-item">
              <span class="meta-icon">📊</span> 
              <span id="analyzedPages">{{ insightStore.analyzedPageCount }}</span> 已分析
            </span>
          </div>
        </div>

        <AnalysisProgress 
          v-if="hasSelectedBook"
          @start-polling="startStatusPolling"
          @stop-polling="stopStatusPolling"
        />

        <PagesTree v-if="hasSelectedBook" />
      </aside>

      <div class="insight-content">
        <div v-if="!hasSelectedBook" class="select-book-prompt">
          <div class="prompt-icon">📚</div>
          <h2>选择要分析的书籍</h2>
          <p>从下方列表中选择一本书籍开始智能分析</p>
          <BookSelector @select="loadBook" />
        </div>

        <div v-else class="content-tabs">
          <UiButton
            variant="toolbar" 
            class="mobile-nav-btn" 
            @click="toggleMobileSidebar" 
            aria-label="打开导航"
          >
            📚
          </UiButton>
          <div class="tabs-wrapper">
            <UiButton
              variant="toolbar" 
              class="tab-btn" 
              :class="{ active: activeTab === 'overview' }"
              @click="switchTab('overview')"
            >
              <span class="tab-icon">📊</span> 概览
            </UiButton>
            <UiButton
              variant="toolbar" 
              class="tab-btn" 
              :class="{ active: activeTab === 'qa' }"
              @click="switchTab('qa')"
            >
              <span class="tab-icon">💬</span> 智能问答
            </UiButton>
            <UiButton
              variant="toolbar" 
              class="tab-btn" 
              :class="{ active: activeTab === 'timeline' }"
              @click="switchTab('timeline')"
            >
              <span class="tab-icon">📈</span> 时间线
            </UiButton>
            <UiButton
              variant="toolbar" 
              class="tab-btn" 
              :class="{ active: activeTab === 'continuation' }"
              @click="switchTab('continuation')"
            >
              <span class="tab-icon">🎨</span> 续写
            </UiButton>
            <UiButton
              variant="toolbar"
              class="tab-btn"
              :class="{ active: activeTab === 'character_studio' }"
              @click="switchTab('character_studio')"
            >
              <span class="tab-icon">🃏</span> 角色工坊
            </UiButton>
          </div>
          <UiButton
            variant="toolbar" 
            class="mobile-nav-btn" 
            @click="toggleMobileWorkspace" 
            aria-label="打开笔记"
          >
            📝
          </UiButton>
        </div>

        <div v-show="activeTab === 'overview' && hasSelectedBook" class="tab-content">
          <OverviewPanel />
        </div>

        <div v-show="activeTab === 'qa' && hasSelectedBook" class="tab-content">
          <QAPanel />
        </div>

        <div v-show="activeTab === 'timeline' && hasSelectedBook" class="tab-content">
          <TimelinePanel />
        </div>

        <div v-show="activeTab === 'continuation' && hasSelectedBook" class="tab-content">
          <ContinuationPanel />
        </div>

        <div v-show="activeTab === 'character_studio' && hasSelectedBook" class="tab-content">
          <CharacterStudioEntryPanel />
        </div>
      </div>

      <aside 
        v-if="hasSelectedBook" 
        class="insight-workspace"
        :class="{ 'mobile-visible': showMobileWorkspace }"
      >
        <PageDetail />

        <NotesPanel />
      </aside>
    </SidebarLayout>

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
  --insight-border-color: var(--color-border-muted);

  overflow: hidden;
  margin: 0;
  padding: 56px 20px 0 20px;
  display: flex;
  flex-direction: column;
}

.insight-page .insight-header__nav-link {
    color: var(--insight-text-secondary);
    text-decoration: none;
    font-size: 14px;
    padding: 6px 12px;
    border-radius: 6px;
    transition: all 0.2s;
}

.insight-page .insight-header__nav-link:hover {
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-primary);
}

.insight-page .insight-header__nav-link--active {
    background: var(--insight-action-primary);
    color: var(--color-text-inverse);
}

.insight-page .insight-header__theme-toggle {
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 18px;
}

.insight-page .insight-main {
    display: flex;
    flex: 1;
    background: var(--insight-surface-page);
    overflow: hidden;
}

.insight-page .insight-sidebar {
    width: 280px;
    min-width: 280px;
    background: var(--insight-surface-secondary);
    border-right: 1px solid var(--insight-border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    max-height: 100%;
}

.insight-page .insight-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}

.insight-page .insight-workspace {
    width: 320px;
    min-width: 320px;
    background: var(--insight-surface-secondary);
    border-left: 1px solid var(--insight-border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    max-height: 100%;
}

.insight-page .content-tabs {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--insight-border-color);
    background: var(--insight-surface-secondary);
    align-items: center;
}

.insight-page .tabs-wrapper {
    display: flex;
    gap: 4px;
    flex: 1;
}

.insight-page .mobile-nav-btn {
    display: none;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-primary);
    border: 1px solid var(--insight-border-color);
    cursor: pointer;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: all 0.2s;
    flex-shrink: 0;
}

.insight-page .mobile-nav-btn:hover {
    background: var(--insight-action-primary);
    color: var(--color-text-inverse);
    border-color: var(--insight-action-primary);
}

.insight-page .mobile-nav-btn.active {
    background: var(--insight-action-primary);
    color: var(--color-text-inverse);
    border-color: var(--insight-action-primary);
}

.insight-page .tab-btn {
    padding: 8px 16px;
    border: none;
    background: transparent;
    color: var(--insight-text-secondary);
    font-size: 14px;
    cursor: pointer;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s;
}

.insight-page .tab-btn:hover {
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-primary);
}

.insight-page .tab-btn.active {
    background: var(--insight-action-primary);
    color: var(--color-text-inverse);
}

.insight-page .tab-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.insight-page .select-book-prompt {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    text-align: center;
}

.insight-page .prompt-icon {
    font-size: 64px;
    margin-bottom: 16px;
}

.insight-page .select-book-prompt h2 {
    margin-bottom: 8px;
    color: var(--insight-text-primary);
}

.insight-page .select-book-prompt p {
    color: var(--insight-text-secondary);
    margin-bottom: 24px;
}

.insight-page .book-selector {
    width: 300px;
}

.insight-page .insight-settings-action {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 500;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-primary);
}

.insight-page .insight-settings-action:hover {
    background: var(--insight-border-color);
}

.insight-page .placeholder-text {
    color: var(--insight-text-muted);
    text-align: center;
    padding: 20px;
    font-size: 14px;
}

.insight-page .empty-hint {
    color: var(--insight-text-muted);
    text-align: center;
    padding: 16px;
    font-size: 13px;
}

.loading-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid var(--insight-border-color);
    border-top-color: var(--insight-action-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.insight-sidebar.mobile-visible,
.insight-workspace.mobile-visible {
  display: block;
}

@media (--breakpoint-md-up) {
  .mobile-nav-btn {
    display: none;
  }
}

.book-info-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 16px;
  text-align: center;
  border-bottom: 1px solid var(--insight-border-color);
}

.book-cover-wrapper {
  width: 120px;
  height: 160px;
  margin: 0 auto 12px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--insight-surface-tertiary);
  position: relative;
}

.book-cover {
  width: 100%;
  height: 100%;
  max-width: 120px;
  max-height: 160px;
  object-fit: cover;
  display: block;
}

.book-cover-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  color: var(--insight-text-muted);
}

.insight-book-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--insight-text-primary);
  margin: 0 0 10px 0;
  text-align: center;
  max-width: 100%;
  word-break: break-word;
  line-height: 1.4;
}

.book-meta {
  display: flex;
  justify-content: center;
  gap: 16px;
  font-size: 13px;
  color: var(--insight-text-secondary);
  flex-wrap: wrap;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.meta-icon {
  font-size: 14px;
}

.sidebar-section {
  padding: 12px 0;
  border-bottom: 1px solid var(--insight-border-color);
}

.sidebar-section:last-child {
  border-bottom: none;
}

</style>
