<script setup lang="ts">
/**
 * 书籍选择器组件
 * 用于在漫画分析页面选择要分析的书籍
 */

import { ref, computed } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import CustomSelect from '@/components/common/CustomSelect.vue'

// ============================================================
// 事件定义
// ============================================================

const emit = defineEmits<{
  /** 选择书籍事件 */
  (e: 'select', bookId: string): void
}>()

// ============================================================
// 状态
// ============================================================

const bookshelfStore = useBookshelfStore()

// ============================================================
// 计算属性
// ============================================================

/** 书籍列表 */
const books = computed(() => bookshelfStore.books)

/** 书籍选项（用于CustomSelect） */
const bookOptions = computed(() => {
  const options = [{ label: '-- 选择书籍 --', value: '' }]
  books.value.forEach(book => {
    options.push({
      label: book.title || book.id,
      value: book.id
    })
  })
  return options
})

/** 当前选中的书籍ID */
const selectedBookId = ref('')

// ============================================================
// 方法
// ============================================================

/**
 * 处理书籍选择
 * @param bookId - 选中的书籍ID
 */
function handleSelect(bookId: string): void {
  selectedBookId.value = bookId
  if (bookId) {
    emit('select', bookId)
  }
}
</script>

<template>
  <div class="book-selector">
    <CustomSelect
      v-model="selectedBookId"
      :options="bookOptions"
      @change="handleSelect"
    />
  </div>
</template>

<style scoped>
/* ==================== 书籍选择器完整样式 - 从 manga-insight.css 迁移 ==================== */

/* ==================== CSS变量 ==================== */
.sidebar-section {
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f1f5f9;
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --primary-color: #6366f1;
}

:global(body.dark-theme) .sidebar-section,
.sidebar-section.dark-theme {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border-color: #334155;
}

/* ==================== 组件样式 ==================== */
.sidebar-section {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
}

.section-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.book-info-section {
    text-align: center;
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

.book-cover-wrapper .book-cover {
    width: 100%;
    height: 100%;
    max-width: 120px;
    max-height: 160px;
    object-fit: cover;
    display: block;
}

.book-cover-wrapper .book-cover-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
    color: var(--text-muted);
}

.book-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text-primary);
}

.book-meta {
    display: flex;
    justify-content: center;
    gap: 16px;
    font-size: 12px;
    color: var(--text-secondary);
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 4px;
}

.book-selector {
    width: 300px;
}

.book-select {
    width: 100%;
    padding: 12px 16px;
    font-size: 14px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-primary);
    color: var(--text-primary);
}

.chapter-item {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s;
    gap: 8px;
}

.chapter-item:hover {
    background: var(--bg-tertiary);
}

.chapter-item.selected {
    background: var(--primary-color);
    color: white;
}

.chapter-item.analyzed .chapter-status {
    color: var(--success-color);
}

.chapter-status {
    font-size: 12px;
    width: 16px;
}

.chapter-title {
    flex: 1;
    font-size: 13px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.chapter-pages {
    font-size: 11px;
    color: var(--text-secondary);
}

.chapter-item:hover .btn-reanalyze-chapter {
    opacity: 0.6;
}

.context-menu {
    position: fixed;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    min-width: 160px;
    z-index: 10000;
    padding: 4px 0;
}

.context-menu-item {
    padding: 8px 16px;
    cursor: pointer;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.context-menu-item:hover {
    background: var(--bg-tertiary);
}

.context-menu-divider {
    height: 1px;
    background: var(--border-color);
    margin: 4px 0;
}
</style>
