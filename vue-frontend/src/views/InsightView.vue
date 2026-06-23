<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import AppShell from '@/components/ui/AppShell.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'
/**
 * 漫画分析页面视图组件
 * 提供AI驱动的漫画内容分析，包括概览、时间线、问答和笔记功能
 */

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

// ============================================================
// 路由和状态
// ============================================================

const route = useRoute()
const router = useRouter()
const insightStore = useInsightStore()
const bookshelfStore = useBookshelfStore()

// ============================================================
// 响应式状态
// ============================================================

/** 当前激活的选项卡 */
const activeTab = ref<'overview' | 'qa' | 'timeline' | 'continuation' | 'character_studio'>('overview')

/** 是否显示设置模态框 */
const showSettingsModal = ref(false)

/** 是否显示移动端侧边栏 */
const showMobileSidebar = ref(false)

/** 是否显示移动端工作区 */
const showMobileWorkspace = ref(false)

/** 分析状态轮询定时器 */
let statusPollingTimer: ReturnType<typeof setInterval> | null = null

/** 分析完成后的面板刷新定时器 */
let refreshDataTimer: ReturnType<typeof setTimeout> | null = null

/** 当前加载的书籍详情 */
const loadedBookDetail = ref<{
  id: string
  title: string
  cover?: string
  total_pages: number
} | null>(null)

/** 是否显示章节选择弹窗 */
const showChapterSelectModal = ref(false)

// ============================================================
// 计算属性
// ============================================================

/** 当前书籍信息 - 优先使用加载的详情数据 */
const currentBook = computed(() => {
  if (loadedBookDetail.value) return loadedBookDetail.value
  if (!insightStore.currentBookId) return null
  return bookshelfStore.books.find(b => b.id === insightStore.currentBookId)
})

/** 是否已选择书籍 */
const hasSelectedBook = computed(() => !!insightStore.currentBookId)

/** 书籍封面URL */
const bookCoverUrl = computed(() => {
  if (!currentBook.value?.cover) return ''
  return currentBook.value.cover
})

// ============================================================
// 方法
// ============================================================

/**
 * 切换选项卡
 * @param tab - 选项卡名称
 */
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

/**
 * 加载书籍
 * @param bookId - 书籍ID
 */
async function loadBook(bookId: string): Promise<void> {
  if (!bookId) return

  insightStore.setCurrentBook(bookId)
  insightStore.setLoading(true)

  try {
    const bookData = await getBookDetail(bookId)

    if (!bookData.success) {
      throw new Error(bookData.error || '获取书籍信息失败')
    }

    if (bookData.book) {
      setLoadedBookDetail(bookData.book)
    }

    // 获取分析状态
    await loadAnalysisStatus()

    // 如果书籍信息中没有章节，尝试从章节API获取
    if (insightStore.chapters.length === 0) {
      try {
        const chaptersResponse = await insightApi.getInsightChapters(bookId)
        if (chaptersResponse.success && chaptersResponse.chapters && chaptersResponse.chapters.length > 0) {
          insightStore.setChapters(chaptersResponse.chapters.map(ch => ({
            id: ch.id,
            title: ch.title,
            startPage: ch.start_page,
            endPage: ch.end_page,
            analyzed: true
          })))
        }
      } catch (e) {
        console.warn('获取章节列表失败:', e)
      }
    }

    // 加载笔记（通过API）
    await insightStore.loadNotesFromAPI()

    // 注：概览和时间线数据由 OverviewPanel 和 TimelinePanel 组件在 onMounted 时自行加载
    // triggerDataRefresh 仅在分析完成后由轮询逻辑调用

    // 更新URL参数
    router.replace({ query: { book: bookId } })

    // 如果正在分析，启动轮询
    if (insightStore.isAnalyzing) {
      startStatusPolling()
    }

  } catch (error) {
    console.error('加载书籍失败:', error)
    insightStore.setError(error instanceof Error ? error.message : '加载书籍失败')
  } finally {
    insightStore.setLoading(false)
  }
}

/**
 * 加载分析状态
 */
async function loadAnalysisStatus(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    const response = await insightApi.getAnalysisStatus(insightStore.currentBookId)
    if (response.success) {
      // 更新已分析页数
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
  } catch (error) {
    console.error('获取分析状态失败:', error)
  }
}

/**
 * 启动状态轮询
 * 与接口流程 的 startProgressPolling 保持一致：
 * 分析完成后自动刷新概览数据和目录树
 */
function startStatusPolling(): void {
  stopStatusPolling()
  statusPollingTimer = setInterval(async () => {
    const statusBeforePolling = insightStore.analysisStatus
    await loadAnalysisStatus()
    
    // 检查分析状态变化
    const status = insightStore.analysisStatus
    const wasActiveTask = statusBeforePolling === 'running' || statusBeforePolling === 'paused'
    if ((status === 'completed' || status === 'failed' || status === 'idle') && wasActiveTask) {
      // 停止轮询
      stopStatusPolling()

      const refreshData = async () => {
        await loadAnalysisStatus()
        // 触发面板组件刷新（通过 Store 的 dataRefreshKey）
        insightStore.triggerDataRefresh()
      }

      if (status === 'completed') {
        // completed 时保留延迟，等待后端汇总任务完成
        refreshDataTimer = setTimeout(() => {
          refreshDataTimer = null
          void refreshData()
        }, 1000)
      } else {
        // idle/failed 终态也要刷新，确保章节/单页任务结束后界面闭环
        await refreshData()
      }
    }
  }, 3000)
}

/**
 * 停止状态轮询
 */
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

/**
 * 打开设置模态框
 */
function openSettingsModal(): void {
  showSettingsModal.value = true
}

/**
 * 显示功能开发中提示
 */
function showFeatureNotice(): void {
  showToast('🌙 该功能正在开发中，敬请期待！', 'info')
}

/**
 * 关闭设置模态框
 */
function closeSettingsModal(): void {
  showSettingsModal.value = false
}

/**
 * 切换移动端侧边栏
 */
function toggleMobileSidebar(): void {
  showMobileSidebar.value = !showMobileSidebar.value
  if (showMobileSidebar.value) {
    showMobileWorkspace.value = false
  }
}

/**
 * 切换移动端工作区
 */
function toggleMobileWorkspace(): void {
  showMobileWorkspace.value = !showMobileWorkspace.value
  if (showMobileWorkspace.value) {
    showMobileSidebar.value = false
  }
}

/**
 * 跳转到翻译页面
 * 业务逻辑：根据章节情况决定是否弹窗选择
 */
function goToTranslate(): void {
  if (!insightStore.currentBookId) {
    // 未选书：直接跳转
    router.push('/translate')
    return
  }

  // 获取书籍的章节信息
  const chapters = insightStore.chapters
  
  if (!chapters || chapters.length === 0) {
    // 无章节：只带 book 参数跳转
    router.push({ path: '/translate', query: { book: insightStore.currentBookId } })
  } else if (chapters.length === 1) {
    // 只有 1 章：直接跳转，带上章节参数
    router.push({ 
      path: '/translate', 
      query: { 
        book: insightStore.currentBookId,
        chapter: chapters[0]!.id
      } 
    })
  } else {
    // 多章：弹窗让用户选择
    showChapterSelectModal.value = true
  }
}

/**
 * 处理章节选择
 * @param chapterId - 选中的章节ID
 */
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

/**
 * 关闭章节选择弹窗
 */
function closeChapterSelectModal(): void {
  showChapterSelectModal.value = false
}

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  // 加载书籍列表
  await bookshelfStore.fetchBooks()

  // 检查URL参数
  const bookId = route.query.book as string
  if (bookId) {
    await loadBook(bookId)
  }
})

onUnmounted(() => {
  stopStatusPolling()
})

// 监听分析状态变化
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
    <!-- 页面头部 -->
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

    <!-- 主内容区 -->
    <SidebarLayout as="main" class="insight-main">
      <!-- 左侧边栏 -->
      <aside class="insight-sidebar" :class="{ 'mobile-visible': showMobileSidebar }">
        <!-- 书籍信息 -->
        <div class="sidebar-section book-info-section">
          <div class="book-cover-wrapper">
            <img 
              v-if="bookCoverUrl" 
              :src="bookCoverUrl" 
              alt="封面" 
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

        <!-- 分析控制 -->
        <AnalysisProgress 
          v-if="hasSelectedBook"
          @start-polling="startStatusPolling"
          @stop-polling="stopStatusPolling"
        />

        <!-- 章节与页面导航树 -->
        <PagesTree v-if="hasSelectedBook" />
      </aside>

      <!-- 主内容区 -->
      <div class="insight-content">
        <!-- 选择书籍提示 -->
        <div v-if="!hasSelectedBook" class="select-book-prompt">
          <div class="prompt-icon">📚</div>
          <h2>选择要分析的书籍</h2>
          <p>从下方列表中选择一本书籍开始智能分析</p>
          <BookSelector @select="loadBook" />
        </div>

        <!-- 标签页导航 -->
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

        <!-- 概览标签页 -->
        <div v-show="activeTab === 'overview' && hasSelectedBook" class="tab-content">
          <OverviewPanel />
        </div>

        <!-- 智能问答标签页 -->
        <div v-show="activeTab === 'qa' && hasSelectedBook" class="tab-content">
          <QAPanel />
        </div>

        <!-- 时间线标签页 -->
        <div v-show="activeTab === 'timeline' && hasSelectedBook" class="tab-content">
          <TimelinePanel />
        </div>

        <!-- 续写标签页 -->
        <div v-show="activeTab === 'continuation' && hasSelectedBook" class="tab-content">
          <ContinuationPanel />
        </div>

        <!-- 角色工坊标签页 -->
        <div v-show="activeTab === 'character_studio' && hasSelectedBook" class="tab-content">
          <CharacterStudioEntryPanel />
        </div>
      </div>

      <!-- 右侧工作区 -->
      <aside 
        v-if="hasSelectedBook" 
        class="insight-workspace"
        :class="{ 'mobile-visible': showMobileWorkspace }"
      >
        <!-- 页面详情 -->
        <PageDetail />

        <!-- 笔记 -->
        <NotesPanel />
      </aside>
    </SidebarLayout>

    <!-- 设置模态框 -->
    <InsightSettingsModal 
      v-if="showSettingsModal"
      @close="closeSettingsModal"
    />
    
    <!-- 章节选择弹窗 -->
    <ChapterSelectModal
      v-if="showChapterSelectModal && insightStore.currentBookId"
      :chapters="insightStore.chapters"
      @select="handleChapterSelect"
      @close="closeChapterSelectModal"
    />
  </AppShell>
</template>

<style scoped>
/* ==================== 漫画分析页面样式 ==================== */

/* ==================== 页面根容器固定布局 ==================== */
.insight-page {
  --insight-border-color: var(--color-border-muted);

  overflow: hidden;
  margin: 0;
  padding: 56px 20px 0 20px;
  display: flex;
  flex-direction: column;
}

/* Header 内 slot 元素样式 */
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
    color: white;
}

.insight-page .insight-header__theme-toggle {
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 18px;
}

/* 布局 */
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
    /* 高度填满父容器，内容溢出时滚动 */
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
    /* 高度填满父容器，内容溢出时滚动 */
    max-height: 100%;
}

/* 标签页 */
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
    color: white;
    border-color: var(--insight-action-primary);
}

.insight-page .mobile-nav-btn.active {
    background: var(--insight-action-primary);
    color: white;
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
    color: white;
}

.insight-page .tab-content {
    /* 注意：display 由 v-show 控制，不在 CSS 中设置 */
    flex: 1;
    overflow-y: auto;
    padding: 0;
}

/* 选择书籍提示 */
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

/* 通用样式 */
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

/* 移动端侧边栏显示控制 */
.insight-sidebar.mobile-visible,
.insight-workspace.mobile-visible {
  display: block;
}

/* 移动端导航按钮 */
@media (--breakpoint-md-up) {
  .mobile-nav-btn {
    display: none;
  }
}

/* ==================== 书籍信息区域样式 - 按业务契约的垂直居中布局 ==================== */
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

/* ==================== 侧边栏区域通用样式 ==================== */
.sidebar-section {
  padding: 12px 0;
  border-bottom: 1px solid var(--insight-border-color);
}

.sidebar-section:last-child {
  border-bottom: none;
}

/* ==================== 标签页显示状态 ==================== */
.tab-content[style*="display: none"] {
  display: none;
}

.tab-content:not([style*="display: none"]) {
  display: block;
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
</style>
