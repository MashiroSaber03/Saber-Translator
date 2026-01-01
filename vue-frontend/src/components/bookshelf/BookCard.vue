<script setup lang="ts">
/**
 * 书籍卡片组件
 * 使用与原版bookshelf.js完全相同的HTML结构和CSS类名
 */

import type { BookData } from '@/types'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { computed } from 'vue'

interface Props {
  book: BookData
  selected?: boolean
  batchMode?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  selected: false,
  batchMode: false,
})

const emit = defineEmits<{
  click: []
  edit: []
  select: []
}>()

const bookshelfStore = useBookshelfStore()
const allTags = computed(() => bookshelfStore.tags)

// 处理点击事件
function handleClick() {
  if (props.batchMode) {
    emit('select')
  } else {
    emit('click')
  }
}

// 获取标签颜色
function getTagColor(tagName: string): string {
  const tagInfo = allTags.value.find(t => t.name === tagName)
  return tagInfo?.color || '#667eea'
}

// 检查是否有有效封面
function hasCover(): boolean {
  return !!props.book.cover && props.book.cover.length > 0
}

// 处理图片加载错误
function handleImageError(event: Event) {
  const img = event.target as HTMLImageElement
  if (img.parentElement) {
    img.style.display = 'none'
    // 创建占位符
    const placeholder = document.createElement('div')
    placeholder.className = 'book-cover-placeholder'
    placeholder.textContent = '📖'
    img.parentElement.appendChild(placeholder)
  }
}
</script>

<template>
  <!-- 书籍卡片 - 使用与原版相同的HTML结构 -->
  <div
    class="book-card"
    :class="{ selected: selected, 'batch-mode': batchMode }"
    @click="handleClick"
  >
    <!-- 选择框（批量模式） -->
    <div v-if="batchMode" class="book-checkbox" @click.stop="emit('select')">
      <input type="checkbox" :checked="selected" @click.stop @change="emit('select')">
    </div>

    <!-- 封面图片 -->
    <div class="book-cover">
      <img
        v-if="hasCover()"
        :src="book.cover"
        :alt="book.title"
        @error="handleImageError"
      >
      <div v-else class="book-cover-placeholder">📖</div>
    </div>

    <!-- 书籍信息 - 垂直布局：书名、章节数、标签各占一行 -->
    <div class="book-info">
      <h3 class="book-title" :title="book.title">{{ book.title }}</h3>
      <p class="book-chapter-count">{{ book.chapter_count || book.chapters?.length || 0 }} 章节</p>
      <div v-if="book.tags && book.tags.length > 0" class="book-tags">
        <span
          v-for="tag in book.tags"
          :key="tag"
          class="book-tag"
          :style="{ background: getTagColor(tag) }"
        >
          {{ tag }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==================== 书籍卡片样式 - 完整迁移自 bookshelf.css ==================== */

/* 书籍卡片 */
.book-card {
    background: var(--card-bg);
    border-radius: var(--border-radius-md, 12px);
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
}

.book-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--border-radius-md, 12px);
    border: 2px solid transparent;
    transition: border-color 0.2s ease;
    pointer-events: none;
}

.book-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
}

.book-card:hover::after {
    border-color: rgba(102, 126, 234, 0.5);
}

.book-card:active {
    transform: translateY(-2px) scale(1.01);
}

/* 批量模式 */
.book-card.batch-mode {
    cursor: default;
}

.book-card.selected {
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.5);
}

/* 书籍封面 */
.book-cover {
    aspect-ratio: 3 / 4;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: block;
    overflow: hidden;
    position: relative;
    border-radius: var(--border-radius-md, 12px) var(--border-radius-md, 12px) 0 0;
}

.book-cover img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    transition: transform 0.3s ease;
}

.book-card:hover .book-cover img {
    transform: scale(1.05);
}

/* 书籍封面悬停遮罩 */
.book-cover::before {
    content: '查看详情';
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.9rem;
    font-weight: 500;
    opacity: 0;
    transition: opacity 0.2s ease;
    z-index: 1;
}

.book-card:hover .book-cover::before {
    opacity: 1;
}

.book-cover-placeholder {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 3rem;
    color: rgba(255, 255, 255, 0.8);
}

/* 书籍信息 */
.book-info {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.book-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
}

.book-chapter-count {
    font-size: 0.8rem;
    margin: 0;
    color: var(--text-secondary);
    margin: 4px 0;
}

/* 书籍标签 */
.book-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
}

.book-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    color: white;
    background: #667eea;
}

/* 批量选择复选框 */
.book-checkbox {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 2;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.book-checkbox input[type="checkbox"] {
    width: 16px;
    height: 16px;
    cursor: pointer;
    accent-color: #667eea;
}
</style>
