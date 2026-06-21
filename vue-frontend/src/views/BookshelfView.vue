<script setup lang="ts">
/**
 * 书架页面视图组件
 * 显示用户的书籍收藏，支持搜索和标签筛选
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { getServerInfo } from '@/api'
import { getBookDetail } from '@/api/bookshelf'
import BookCard from '@/components/bookshelf/BookCard.vue'
import BookSearch from '@/components/bookshelf/BookSearch.vue'
import BookModal from '@/components/bookshelf/BookModal.vue'
import BookDetailModal from '@/components/bookshelf/BookDetailModal.vue'
import TagManageModal from '@/components/bookshelf/TagManageModal.vue'
import ConfirmModal from '@/components/common/ConfirmModal.vue'
import AppHeader from '@/components/common/AppHeader.vue'
import AppShell from '@/components/ui/AppShell.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiEmptyState from '@/components/ui/UiEmptyState.vue'
import { showToast } from '@/utils/toast'

const router = useRouter()
const bookshelfStore = useBookshelfStore()

// 局域网地址
const lanUrl = ref<string>('获取中...')

// 模态框状态
const showBookModal = ref(false)
const showDetailModal = ref(false)
const showTagManageModal = ref(false)
const showConfirmModal = ref(false)
const editingBookId = ref<string | null>(null)
const confirmMessage = ref('')
const confirmCallback = ref<(() => void) | null>(null)

// 计算属性
const filteredBooks = computed(() => bookshelfStore.filteredBooks)
const allTags = computed(() => bookshelfStore.tags)
const isEmpty = computed(() => filteredBooks.value.length === 0 && !bookshelfStore.searchQuery)

// pageshow 事件处理函数
// 当从翻译页面返回时（通过浏览器后退按钮），如果页面被 BFCache 缓存，自动刷新数据
function handlePageShow(event: PageTransitionEvent) {
  if (event.persisted) {
    console.log('[BookshelfView] 页面从 BFCache 恢复，刷新数据')
    bookshelfStore.loadBooks()
    bookshelfStore.loadTags()
    // 如果详情模态框已打开，刷新当前书籍详情
    if (showDetailModal.value && bookshelfStore.currentBook) {
      openBookDetail(bookshelfStore.currentBook.id)
    }
  }
}

// 初始化
onMounted(async () => {
  // 加载书籍和标签
  await Promise.all([
    bookshelfStore.loadBooks(),
    bookshelfStore.loadTags(),
  ])
  
  // 获取局域网地址
  try {
    const response = await getServerInfo()
    if (response.success && response.lan_url) {
      lanUrl.value = response.lan_url
    }
  } catch (error) {
    console.error('获取服务器信息失败:', error)
    lanUrl.value = '获取失败'
  }
  
  // 【当前行为】添加 pageshow 事件监听，处理浏览器 BFCache
  window.addEventListener('pageshow', handlePageShow)
})

// 清理事件监听器
onUnmounted(() => {
  window.removeEventListener('pageshow', handlePageShow)
})

// 复制局域网地址
async function copyLanUrl() {
  try {
    await navigator.clipboard.writeText(lanUrl.value)
    showToast('局域网地址已复制！', 'success')
  } catch {
    // 降级方案
    const textArea = document.createElement('textarea')
    textArea.value = lanUrl.value
    document.body.appendChild(textArea)
    textArea.select()
    document.execCommand('copy')
    document.body.removeChild(textArea)
    showToast('局域网地址已复制！', 'success')
  }
}

// 打开新建书籍模态框
function openCreateBookModal() {
  editingBookId.value = null
  showBookModal.value = true
}

// 打开编辑书籍模态框
function openEditBookModal(bookId: string) {
  editingBookId.value = bookId
  showBookModal.value = true
}

// 打开书籍详情模态框 - 调用API获取完整数据（包括章节）
// 失败时显示 toast，不打开不完整的书籍详情模态框
async function openBookDetail(bookId: string) {
  try {
    const response = await getBookDetail(bookId)
    
    if (!response.success) {
      throw new Error(response.error || '加载失败')
    }
    
    if (response.book) {
      // 更新store中的书籍数据
      bookshelfStore.updateBook(bookId, response.book)
    }
    
    // 只有成功时才设置当前书籍并打开模态框
    bookshelfStore.setCurrentBook(bookId)
    showDetailModal.value = true
    
  } catch (error) {
    // 【当前行为】失败时显示 toast 提示
    const errorMsg = error instanceof Error ? error.message : '未知错误'
    console.error('加载书籍详情失败:', error)
    showToast(`加载书籍详情失败: ${errorMsg}`, 'error')
  }
}

// 打开标签管理模态框
function openTagManageModal() {
  showTagManageModal.value = true
}

// 跳转到快速翻译
function goToTranslate() {
  router.push('/translate')
}

// 显示功能开发中提示
function showFeatureNotice() {
  showToast('🌙 该功能正在开发中，敬请期待！', 'info')
}
</script>

<template>
  <AppShell class="bookshelf-page">
    <!-- 页面头部 -->
    <AppHeader variant="bookshelf" logo-title="书架首页">
      <template #header-links>
        <span class="bookshelf-header__lan-access" title="其他设备可通过此地址访问">
          <span class="bookshelf-header__lan-icon">🌐局域网设备可通过该网址访问</span>
          <span id="lanUrl">{{ lanUrl }}</span>
          <UiButton variant="toolbar" class="bookshelf-header__copy-button" title="复制地址" @click="copyLanUrl">📋</UiButton>
        </span>
        <a href="http://www.mashirosaber.top" target="_blank" class="bookshelf-header__tutorial-link">使用教程</a>
        <a href="https://github.com/MashiroSaber03/Saber-Translator" target="_blank" class="bookshelf-header__github-link">
          <img src="/pic/github.jpg" alt="GitHub" class="bookshelf-header__github-icon">
        </a>
        <UiButton variant="toolbar" class="bookshelf-header__theme-toggle" title="功能开发中" @click="showFeatureNotice">
          <span class="bookshelf-header__theme-icon">☀️</span>
        </UiButton>
      </template>
    </AppHeader>

    <!-- 主内容区 -->
    <main class="bookshelf-main">
      <!-- 工具栏 -->
      <div class="bookshelf-toolbar">
        <h1 class="page-title">我的书架</h1>
        <div class="toolbar-actions">
          <UiButton variant="primary" @click="openCreateBookModal">
            <span class="bookshelf-button-icon">+</span>
            <span>新建书籍</span>
          </UiButton>
          <UiButton variant="secondary" @click="openTagManageModal">
            <span>🏷️ 管理标签</span>
          </UiButton>
          <UiButton variant="secondary" @click="goToTranslate">
            <span>快速翻译</span>
          </UiButton>
        </div>
      </div>

      <!-- 搜索和筛选栏 -->
      <BookSearch
        :tags="allTags"
        @search="bookshelfStore.setSearchQuery"
        @filter-tag="bookshelfStore.toggleTagFilter"
      />

      <!-- 书籍网格 -->
      <div class="books-container">
        <div v-if="filteredBooks.length > 0" class="books-grid">
          <BookCard
            v-for="book in filteredBooks"
            :key="book.id"
            :book="book"
            @click="openBookDetail(book.id)"
          />
        </div>
        
        <!-- 空状态提示 -->
        <UiEmptyState
          v-else-if="isEmpty"
          icon="📚"
          title="书架空空如也"
          description="点击&quot;新建书籍&quot;开始你的翻译之旅"
        >
          <UiButton variant="primary" @click="openCreateBookModal">
            <span class="bookshelf-button-icon">+</span>
            <span>新建第一本书</span>
          </UiButton>
        </UiEmptyState>
        
        <!-- 搜索无结果 -->
        <UiEmptyState
          v-else
          icon="🔍"
          title="未找到匹配的书籍"
          description="尝试调整搜索条件或标签筛选"
        />
      </div>
    </main>

    <!-- 模态框 -->
    <BookModal
      v-if="showBookModal"
      :book-id="editingBookId"
      @close="showBookModal = false"
      @saved="showBookModal = false"
    />

    <BookDetailModal
      v-if="showDetailModal"
      @close="showDetailModal = false"
      @edit="openEditBookModal"
    />

    <TagManageModal
      v-if="showTagManageModal"
      @close="showTagManageModal = false"
    />

    <ConfirmModal
      v-if="showConfirmModal"
      :message="confirmMessage"
      @confirm="confirmCallback?.(); showConfirmModal = false"
      @cancel="showConfirmModal = false"
    />
  </AppShell>
</template>

<style scoped>
/* ==================== 书架页面样式 ==================== */

/* header 内 slot 元素样式（需要 :deep 因为元素在 AppHeader 子组件 slot 中渲染） */
.bookshelf-header__lan-access {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--bookshelf-view-text-primary);
    font-size: 0.85rem;
    background: var(--color-surface-overlay-light-raised);
    padding: 6px 12px;
    border-radius: 20px;
    backdrop-filter: blur(4px);
    font-family: var(--font-mono, 'Consolas', 'Monaco', monospace);
}

.bookshelf-header__tutorial-link {
    color: var(--bookshelf-view-text-secondary);
    text-decoration: none;
    padding: 6px 12px;
    border-radius: 20px;
    background: var(--color-surface-overlay-light-soft);
    transition: all 0.2s ease;
}

.bookshelf-header__tutorial-link:hover {
    background: var(--color-surface-overlay-medium-strong);
}

.bookshelf-header__github-link {
    display: flex;
    align-items: center;
    padding: 6px;
    border-radius: 50%;
    background: var(--color-surface-overlay-light-soft);
    transition: all 0.2s ease;
}

.bookshelf-header__github-link:hover {
    background: var(--color-surface-overlay-medium-strong);
}

.bookshelf-header__github-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
}

.bookshelf-header__theme-toggle {
    background: var(--color-surface-overlay-medium);
    border: none;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
}

.bookshelf-header__theme-toggle:hover {
    background: var(--bookshelf-view-surface-base);
    transform: rotate(15deg);
}

/* 主内容区 */
.bookshelf-main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    min-height: 0;
}

.bookshelf-page {
    padding-inline: 20px;
}

.bookshelf-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 32px;
    flex-wrap: wrap;
    gap: 16px;
}

.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--color-text-default);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.page-title::before {
    content: '📚';
    font-size: 1.5rem;
}

.toolbar-actions {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.bookshelf-button-icon {
    font-size: 1.1rem;
    font-weight: 600;
}

/* 书籍网格容器 */
.books-container {
    min-height: 400px;
}

.books-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 24px;
}

/* 模态框、表单和 Toast 样式由对应组件 owner 管理 */
</style>
