<script setup lang="ts">
/**
 * 书籍卡片组件
 * 展示书籍封面、标题、标签和基础统计信息。
 */

import type { BookData, TagData } from '@/types'
import UiButton from '@/components/ui/UiButton.vue'
import { computed, ref, watch } from 'vue'

interface Props {
  book: BookData
  tags?: TagData[]
}

const props = withDefaults(defineProps<Props>(), {
  tags: () => [],
})

const emit = defineEmits<{
  click: []
}>()

const coverFailed = ref(false)

const hasVisibleCover = computed(() => {
  return Boolean(props.book.cover && props.book.cover.length > 0 && !coverFailed.value)
})

watch(() => props.book.cover, () => {
  coverFailed.value = false
})

// 处理点击事件
function handleClick() {
  emit('click')
}

// 获取标签颜色
function getTagColor(tagName: string): string {
  const tagInfo = props.tags.find(tag => tag.name === tagName)
  return tagInfo?.color || '#667eea'
}

// 处理图片加载错误
function handleImageError() {
  coverFailed.value = true
}
</script>

<template>
  <UiButton
    variant="toolbar"
    type="button"
    class="book-card"
    :aria-label="`打开书籍：${book.title}`"
    @click="handleClick"
  >
    <!-- 封面图片 -->
    <div class="book-cover">
      <img
        v-if="hasVisibleCover"
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
  </UiButton>
</template>

<style scoped>
.book-card {
  --book-card-hover-border: rgba(102, 126, 234, .5);
  --book-card-shadow: rgba(0, 0, 0, .08);
  --book-card-hover-shadow: rgba(0, 0, 0, .15);
  --book-card-cover-overlay: rgba(0, 0, 0, .6);
  --book-card-cover-placeholder: rgba(255, 255, 255, .8);
}

/* 书籍卡片 */
.book-card {
    display: block;
    width: 100%;
    padding: 0;
    border: 0;
    text-align: left;
    background: var(--color-surface-card);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: 0 4px 12px var(--book-card-shadow);
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
}

.book-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--radius-lg);
    border: 2px solid transparent;
    transition: border-color 0.2s ease;
    pointer-events: none;
}

.book-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 32px var(--book-card-hover-shadow);
}

.book-card:hover::after {
    border-color: var(--book-card-hover-border);
}

.book-card:active {
    transform: translateY(-2px) scale(1.01);
}

/* 书籍封面 */
.book-cover {
    aspect-ratio: 3 / 4;
    background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
    display: block;
    overflow: hidden;
    position: relative;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
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
    background: var(--book-card-cover-overlay);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.9rem;
    font-weight: 500;
    opacity: 0;
    transition: opacity 0.2s ease;
    z-index: var(--z-local);
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
    color: var(--book-card-cover-placeholder);
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
    color: var(--color-text-default);
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
}

.book-chapter-count {
    font-size: 0.8rem;
    color: var(--color-text-supporting);
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
    background: var(--color-action-brand);
}
</style>
