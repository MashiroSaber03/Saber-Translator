<script setup lang="ts">
/**
 * 书架页面视图组件
 * 显示用户的书籍收藏，支持搜索和标签筛选
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { getServerInfo } from '@/api'
import { getBookDetail } from '@/api/bookshelf'  // 【修复 P2】导入 bookshelf API
import BookCard from '@/components/bookshelf/BookCard.vue'
import BookSearch from '@/components/bookshelf/BookSearch.vue'
import BookModal from '@/components/bookshelf/BookModal.vue'
import BookDetailModal from '@/components/bookshelf/BookDetailModal.vue'
import TagManageModal from '@/components/bookshelf/TagManageModal.vue'
import ConfirmModal from '@/components/common/ConfirmModal.vue'
import AppHeader from '@/components/common/AppHeader.vue'
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

// 【复刻原版 bookshelf.js】pageshow 事件处理函数
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
  
  // 【复刻原版】添加 pageshow 事件监听，处理浏览器 BFCache
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
// 【复刻原版 bookshelf.js openBookDetail】失败时显示 toast，不打开不完整的模态框
async function openBookDetail(bookId: string) {
  try {
    // 【修复 P2】使用统一的 API 调用方式
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
    // 【复刻原版】失败时显示 toast 提示
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
  <div class="bookshelf-page">
    <!-- 页面头部 -->
    <AppHeader variant="bookshelf" logo-title="书架首页">
      <template #header-links>
        <span class="lan-access-info" title="其他设备可通过此地址访问">
          <span class="lan-icon">🌐局域网设备可通过该网址访问</span>
          <span id="lanUrl">{{ lanUrl }}</span>
          <button type="button" class="copy-btn" title="复制地址" aria-label="复制局域网地址" @click="copyLanUrl">📋</button>
        </span>
        <a href="http://www.mashirosaber.top" target="_blank" rel="noopener noreferrer" class="tutorial-link">使用教程</a>
        <a href="https://github.com/MashiroSaber03/Saber-Translator" target="_blank" rel="noopener noreferrer" class="github-link">
          <img src="/pic/github.jpg" alt="GitHub" class="github-icon">
        </a>
        <button type="button" class="theme-toggle" title="功能开发中" aria-label="主题切换（开发中）" @click="showFeatureNotice">
          <span class="theme-icon">☀️</span>
        </button>
      </template>
    </AppHeader>

    <!-- 主内容区 -->
    <main class="bookshelf-main">
      <!-- 工具栏 -->
      <div class="bookshelf-toolbar">
        <h1 class="page-title">我的书架</h1>
        <div class="toolbar-actions">
          <button class="btn btn-primary" @click="openCreateBookModal">
            <span class="btn-icon">+</span>
            <span>新建书籍</span>
          </button>
          <button class="btn btn-secondary" @click="openTagManageModal">
            <span>🏷️ 管理标签</span>
          </button>
          <button class="btn btn-secondary" @click="goToTranslate">
            <span>快速翻译</span>
          </button>
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
        <div v-else-if="isEmpty" class="empty-state">
          <div class="empty-icon">📚</div>
          <h2>书架空空如也</h2>
          <p>点击"新建书籍"开始你的翻译之旅</p>
          <button class="btn btn-primary" @click="openCreateBookModal">
            <span class="btn-icon">+</span>
            <span>新建第一本书</span>
          </button>
        </div>
        
        <!-- 搜索无结果 -->
        <div v-else class="empty-state">
          <div class="empty-icon">🔍</div>
          <h2>未找到匹配的书籍</h2>
          <p>尝试调整搜索条件或标签筛选</p>
        </div>
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
  </div>
</template>

<style scoped>
/* ==================== 书架页面完整样式 - 完整迁移自 bookshelf.css ==================== */

/* header 内 slot 元素样式（需要 :deep 因为元素在 AppHeader 子组件 slot 中渲染） */
.lan-access-info {
    display: flex;
    align-items: center;
    gap: 6px;
    color: rgba(255, 255, 255, 0.95);
    font-size: 0.85rem;
    background: rgba(255, 255, 255, 0.18);
    padding: 6px 12px;
    border-radius: 20px;
    backdrop-filter: blur(4px);
    font-family: var(--font-mono, 'Consolas', 'Monaco', monospace);
}

.tutorial-link {
    color: rgba(255, 255, 255, 0.9);
    text-decoration: none;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.15);
    transition: all 0.2s ease;
}

.tutorial-link:hover {
    background: rgba(255, 255, 255, 0.25);
}

.github-link {
    display: flex;
    align-items: center;
    padding: 6px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.15);
    transition: all 0.2s ease;
}

.github-link:hover {
    background: rgba(255, 255, 255, 0.25);
}

.github-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
}

.theme-toggle {
    background: rgba(255, 255, 255, 0.2);
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

.theme-toggle:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: rotate(15deg);
}

/* 主内容区 */
.bookshelf-main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    min-height: calc(100vh - 64px);
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
    color: var(--text-primary);
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

/* 按钮样式 */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    white-space: nowrap;
    user-select: none;
}

.btn:active {
    transform: scale(0.97);
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.btn-secondary {
    background: var(--card-bg);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: var(--hover-bg);
    border-color: var(--text-secondary);
}

.btn-danger {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
}

.btn-danger:hover {
    background: linear-gradient(135deg, #e04555 0%, #d63343 100%);
    box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4);
}

.btn-sm {
    padding: 6px 14px;
    font-size: 0.85rem;
}

.btn-icon {
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

/* 空状态 */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px;
    text-align: center;
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: 16px;
}

.empty-state h2 {
    font-size: 1.5rem;
    color: var(--text-primary);
    margin: 0 0 8px 0;
}

.empty-state p {
    color: var(--text-secondary);
    margin: 0 0 24px 0;
}

/* 模态框/表单/Toast 样式已迁移到 BaseModal + global.css */
</style>
