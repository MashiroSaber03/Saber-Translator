<script setup lang="ts">
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
import * as insightApi from '@/api/insight'
import { showToast } from '@/utils/toast'

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
const activeTab = ref<'overview' | 'qa' | 'timeline'>('overview')

/** 是否显示设置模态框 */
const showSettingsModal = ref(false)

/** 是否显示移动端侧边栏 */
const showMobileSidebar = ref(false)

/** 是否显示移动端工作区 */
const showMobileWorkspace = ref(false)

/** 分析状态轮询定时器 */
let statusPollingTimer: ReturnType<typeof setInterval> | null = null

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
function switchTab(tab: 'overview' | 'qa' | 'timeline'): void {
  activeTab.value = tab
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
    // 获取书籍详情
    const bookResponse = await fetch(`/api/bookshelf/books/${bookId}`)
    const bookData = await bookResponse.json()

    if (!bookData.success) {
      throw new Error(bookData.error || '获取书籍信息失败')
    }

    // 存储书籍详情数据
    if (bookData.book) {
      loadedBookDetail.value = {
        id: bookData.book.id,
        title: bookData.book.title,
        cover: bookData.book.cover,
        total_pages: bookData.book.total_pages || 0
      }
      // 设置书籍总页数到store
      insightStore.setBookTotalPages(bookData.book.total_pages || 0)
      
      // 从书籍信息中获取章节数据（与原版JS一致）
      if (bookData.book.chapters && bookData.book.chapters.length > 0) {
        let pageOffset = 0
        const chaptersFromBook = bookData.book.chapters.map((ch: any, idx: number) => {
          const chapterId = ch.id || ch.chapter_id || `ch_${idx + 1}`
          const pageCount = ch.page_count || ch.pages?.length || 0
          const startPage = pageOffset + 1
          const endPage = pageOffset + pageCount
          pageOffset = endPage
          return {
            id: chapterId,
            title: ch.title || `第 ${idx + 1} 章`,
            startPage,
            endPage
          }
        })
        insightStore.setChapters(chaptersFromBook)
      }
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
            endPage: ch.end_page
          })))
        }
      } catch (e) {
        console.warn('获取章节列表失败:', e)
      }
    }

    // 加载笔记（通过API）
    await insightStore.loadNotesFromAPI()

    // 加载概览数据（如果已分析）
    if (insightStore.analyzedPageCount > 0) {
      await loadOverviewData()
    }

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
      
      // 根据current_task或analyzed字段判断状态
      if (response.current_task) {
        const taskStatus = response.current_task.status
        if (taskStatus === 'running') {
          insightStore.setAnalysisStatus('running')
          if (response.current_task.progress) {
            insightStore.updateProgress(
              response.current_task.progress.analyzed_pages || 0,
              response.current_task.progress.total_pages || 0
            )
          }
        } else if (taskStatus === 'paused') {
          insightStore.setAnalysisStatus('paused')
        } else if (taskStatus === 'completed') {
          insightStore.setAnalysisStatus('completed')
        } else if (taskStatus === 'failed') {
          insightStore.setAnalysisStatus('failed')
        } else {
          insightStore.setAnalysisStatus('idle')
        }
      } else if (response.analyzed) {
        insightStore.setAnalysisStatus('completed')
      } else {
        insightStore.setAnalysisStatus('idle')
      }
    }
  } catch (error) {
    console.error('获取分析状态失败:', error)
  }
}

/**
 * 启动状态轮询
 * 与原版 JS 的 startProgressPolling 保持一致：
 * 分析完成后自动刷新概览数据和目录树
 */
function startStatusPolling(): void {
  stopStatusPolling()
  statusPollingTimer = setInterval(async () => {
    await loadAnalysisStatus()
    
    // 检查分析状态变化
    const status = insightStore.analysisStatus
    if (status === 'completed' || status === 'failed' || status === 'idle') {
      // 停止轮询
      stopStatusPolling()
      
      // 分析完成后，刷新概览数据（与原版 JS 一致）
      if (status === 'completed') {
        console.log('分析完成，自动刷新概览数据')
        await loadOverviewData()
        // 触发目录树刷新（通过更新 store 状态让 PagesTree 组件响应）
        // 重新加载分析状态会更新 analyzedPagesCount，PagesTree 会自动响应
        await loadAnalysisStatus()
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
}

/**
 * 加载概览数据（概览、时间线等）
 * 仿照原版JS的loadOverviewData实现
 */
async function loadOverviewData(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    // 加载已生成的模板列表，获取第一个已生成的模板作为默认显示
    const templatesResponse = await insightApi.getGeneratedTemplates(insightStore.currentBookId) as any
    let templateToLoad = 'story_summary' // 默认使用story_summary
    
    if (templatesResponse.success && templatesResponse.generated && templatesResponse.generated.length > 0) {
      // 使用第一个已生成的模板
      templateToLoad = templatesResponse.generated[0]
      insightStore.setGeneratedTemplates(templatesResponse.generated as any)
    }

    // 加载概览内容（使用已生成的模板）
    try {
      const overviewResponse = await insightApi.getOverview(insightStore.currentBookId, templateToLoad) as any
      if (overviewResponse.success && overviewResponse.content) {
        insightStore.setOverview({
          type: templateToLoad,
          content: overviewResponse.content
        } as any)
      }
    } catch (overviewError) {
      // 概览加载失败不影响其他数据加载
      console.warn('加载概览失败:', overviewError)
    }

    // 加载时间线
    const timelineResponse = await insightApi.getTimeline(insightStore.currentBookId) as any
    if (timelineResponse.success) {
      // 时间线数据结构较复杂，直接传递给store处理
      insightStore.setTimeline(timelineResponse)
    }

  } catch (error) {
    console.error('加载概览数据失败:', error)
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
 * 复刻原版逻辑：根据章节情况决定是否弹窗选择
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

// 保存 body 原始样式（用于页面卸载时恢复）
let originalBodyPadding = ''
let originalBodyMargin = ''
let originalBodyOverflow = ''

onMounted(async () => {
  // 【关键修复4】移除 global.css 中 body 的 20px 左右内边距
  // 保存原始样式以便离开页面时恢复
  const bodyStyle = document.body.style
  originalBodyPadding = bodyStyle.padding
  originalBodyMargin = bodyStyle.margin
  originalBodyOverflow = bodyStyle.overflow
  
  // 强制移除 body 的内外边距，并禁止整体滚动
  bodyStyle.padding = '0'
  bodyStyle.margin = '0'
  bodyStyle.overflow = 'hidden'
  
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
  
  // 【关键修复4】恢复 body 的原始样式
  const bodyStyle = document.body.style
  bodyStyle.padding = originalBodyPadding
  bodyStyle.margin = originalBodyMargin
  bodyStyle.overflow = originalBodyOverflow
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
  <div class="insight-page">
    <!-- 页面头部 -->
    <header class="app-header">
      <div class="header-content">
        <div class="logo-container">
          <router-link to="/" title="书架首页">
            <img :src="'/pic/logo.png'" alt="Saber-Translator Logo" class="app-logo">
            <span class="app-name">Saber-Translator</span>
          </router-link>
        </div>
        <div class="header-links">
          <router-link to="/" class="nav-link">📚 书架</router-link>
          <a href="javascript:void(0)" class="nav-link" @click="goToTranslate">🌐 翻译</a>
          <span class="nav-link active">🔍 分析</span>
          <a href="https://www.mashirosaber.top/use/manga-insight.html" target="_blank" class="nav-link" title="使用教程">📖 教程</a>
          <button id="settingsBtn" class="btn btn-icon" title="设置" @click="openSettingsModal">⚙️</button>
          <button id="themeToggle" class="theme-toggle" title="功能开发中" @click="showFeatureNotice">
            <span class="theme-icon">☀️</span>
          </button>
        </div>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="insight-main">
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
          <h2 class="book-title" :title="currentBook?.title">{{ currentBook?.title || '选择书籍' }}</h2>
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
          <button 
            class="mobile-nav-btn" 
            @click="toggleMobileSidebar" 
            aria-label="打开导航"
          >
            📚
          </button>
          <div class="tabs-wrapper">
            <button 
              class="tab-btn" 
              :class="{ active: activeTab === 'overview' }"
              @click="switchTab('overview')"
            >
              <span class="tab-icon">📊</span> 概览
            </button>
            <button 
              class="tab-btn" 
              :class="{ active: activeTab === 'qa' }"
              @click="switchTab('qa')"
            >
              <span class="tab-icon">💬</span> 智能问答
            </button>
            <button 
              class="tab-btn" 
              :class="{ active: activeTab === 'timeline' }"
              @click="switchTab('timeline')"
            >
              <span class="tab-icon">📈</span> 时间线
            </button>
          </div>
          <button 
            class="mobile-nav-btn" 
            @click="toggleMobileWorkspace" 
            aria-label="打开笔记"
          >
            📝
          </button>
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
    </main>

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
  </div>
</template>

<style scoped>
/* ==================== 漫画分析页面完整样式 - 完整迁移自 manga-insight.css ==================== */

/* ==================== 页面根容器固定布局 - 复刻原版 ==================== */

/* 
 * 【关键修复1】建立 BFC 防止外边距折叠，强制固定高度
 * 原版行为：整个页面框架固定在视口内，所有滚动发生在内部容器
 * 
 * 【优化】使用 padding-top 而不是子元素的 margin-top，避免亚像素渲染问题
 */
.insight-page {
  /* 固定高度为视口高度，防止内容撑开 */
  height: 100vh;
  /* 隐藏溢出，确保不出现整体滚动条 */
  overflow: hidden;
  /* 清除外边距，防止折叠到父元素 */
  margin: 0;
  /* 【修复3 + 优化】覆盖 global.css，并为 fixed header 预留空间 */
  /* 合并 padding 声明：top 56px（为 header 预留空间），left/right/bottom 0（覆盖 global.css） */
  padding: 56px 0 0 0 !important;
  /* 使用 Flex 布局以支持子元素的高度计算 */
  display: flex;
  flex-direction: column;
  
  /* CSS变量定义 */
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f1f5f9;
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --text-tertiary: #94a3b8;
  --border-color: #e2e8f0;
  --primary-color: #6366f1;
  --primary-light: #818cf8;
  --primary-dark: #4f46e5;
  --success-color: #22c55e;
  --warning-color: #f59e0b;
  --error-color: #ef4444;
  --bg-hover: rgba(99, 102, 241, 0.1);
}

/* Header样式 */
.insight-page :deep(.app-header) {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    z-index: 100;
    display: flex;
    align-items: center;
    padding: 0 20px;
    max-width: none;
    width: auto;
    margin: 0;
}

.insight-page :deep(.header-content) {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    max-width: 100%;
    padding: 0;
    background: transparent;
    border-radius: 0;
    box-shadow: none;
}

.insight-page :deep(.logo-container a) {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
    color: var(--text-primary);
}

.insight-page :deep(.app-logo) {
    height: 32px;
    width: auto;
    max-height: 32px;
}

.insight-page :deep(.app-name) {
    font-weight: 600;
    font-size: 18px;
}

.insight-page :deep(.header-links) {
    display: flex;
    align-items: center;
    gap: 16px;
}

.insight-page :deep(.nav-link) {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 14px;
    padding: 6px 12px;
    border-radius: 6px;
    transition: all 0.2s;
}

.insight-page :deep(.nav-link:hover) {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.insight-page :deep(.nav-link.active) {
    background: var(--primary-color);
    color: white;
}

.insight-page :deep(.theme-toggle) {
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 18px;
}

/* 布局 */
/* 
 * 【关键修复2】主内容区使用固定高度，用 margin-top 为 fixed header 预留空间
 * 原版行为：主内容区严格占据 "100vh - header高度" 的空间，不会随内容撑开
 * 高度计算：margin-top (56px) + height (calc(100vh - 56px)) = 100vh（正好填满）
 */
.insight-main {
    display: flex;
    /* 使用 flex: 1 自动填充父容器剩余空间（100vh - 56px padding-top） */
    flex: 1;
    background: var(--bg-primary);
    /* 确保内部溢出不影响外层 */
    overflow: hidden;
}

.insight-sidebar {
    width: 280px;
    min-width: 280px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    /* 高度填满父容器，内容溢出时滚动 */
    max-height: 100%;
}

.insight-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}

.insight-workspace {
    width: 320px;
    min-width: 320px;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    /* 高度填满父容器，内容溢出时滚动 */
    max-height: 100%;
}

/* 标签页 */
.content-tabs {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
    align-items: center;
}

.tabs-wrapper {
    display: flex;
    gap: 4px;
    flex: 1;
}

.mobile-nav-btn {
    display: none;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    cursor: pointer;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: all 0.2s;
    flex-shrink: 0;
}

.mobile-nav-btn:hover {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.mobile-nav-btn.active {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.tab-btn {
    padding: 8px 16px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 14px;
    cursor: pointer;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s;
}

.tab-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.tab-btn.active {
    background: var(--primary-color);
    color: white;
}

.tab-content {
    /* 注意：display 由 v-show 控制，不在 CSS 中设置 */
    flex: 1;
    overflow-y: auto;
    /* 【关键修复5】移除内边距，让内容完全填满可用空间 */
    padding: 0;
}

/* 原版兼容：如果不使用 v-show，可通过 active 类控制显示
.tab-content.active {
    display: block;
}
*/

/* 表单元素 */
.form-select,
.form-input {
    width: 100%;
    padding: 8px 12px;
    font-size: 14px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    transition: border-color 0.2s;
}

.form-select:focus,
.form-input:focus {
    outline: none;
    border-color: var(--primary-color);
}

.form-label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 6px;
}

/* 选择书籍提示 */
.select-book-prompt {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    text-align: center;
}

.prompt-icon {
    font-size: 64px;
    margin-bottom: 16px;
}

.select-book-prompt h2 {
    margin-bottom: 8px;
    color: var(--text-primary);
}

.select-book-prompt p {
    color: var(--text-secondary);
    margin-bottom: 24px;
}

.book-selector {
    width: 300px;
}

/* 按钮样式 */
.btn {
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
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background: var(--primary-dark);
}

.btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: var(--border-color);
}

.btn-danger {
    background: var(--error-color);
    color: white;
}

.btn-danger:hover {
    opacity: 0.9;
}

.btn-block {
    width: 100%;
}

/* 通用样式 */
.placeholder-text {
    color: var(--text-muted);
    text-align: center;
    padding: 20px;
    font-size: 14px;
}

.empty-hint {
    color: var(--text-muted);
    text-align: center;
    padding: 16px;
    font-size: 13px;
}

.loading-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 2000;
}

.loading-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid var(--border-color);
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-text {
    margin-top: 16px;
    color: white;
    font-size: 14px;
}

/* 移动端侧边栏显示控制 */
.insight-sidebar.mobile-visible,
.insight-workspace.mobile-visible {
  display: block;
}

/* 移动端导航按钮 */
@media (min-width: 769px) {
  .mobile-nav-btn {
    display: none;
  }
}

/* ==================== 书籍信息区域样式 - 与原版一致的垂直居中布局 ==================== */
.book-info-section {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  padding: 20px 16px !important;
  text-align: center !important;
  border-bottom: 1px solid var(--border-color) !important;
}

.book-cover-wrapper {
  width: 120px;
  height: 160px;
  margin: 0 auto 12px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg-tertiary);
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
  color: var(--text-muted);
}

.book-title {
  font-size: 16px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  margin: 0 0 10px 0 !important;
  text-align: center !important;
  max-width: 100% !important;
  word-break: break-word !important;
  line-height: 1.4 !important;
}

.book-meta {
  display: flex !important;
  justify-content: center !important;
  gap: 16px !important;
  font-size: 13px !important;
  color: var(--text-secondary) !important;
  flex-wrap: wrap !important;
}

.meta-item {
  display: flex !important;
  align-items: center !important;
  gap: 4px !important;
}

.meta-icon {
  font-size: 14px !important;
}

/* ==================== 侧边栏区域通用样式 ==================== */
.sidebar-section {
  padding: 12px 0;
  border-bottom: 1px solid var(--border-color);
}

.sidebar-section:last-child {
  border-bottom: none;
}

/* ==================== v-show修复：标签页内容显示 ==================== */
.tab-content[style*="display: none"] {
  display: none !important;
}

.tab-content:not([style*="display: none"]) {
  display: block;
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
</style>
