<script setup lang="ts">
/**
 * 书籍搜索和标签筛选组件
 */

import { ref, computed } from 'vue'
import type { TagData } from '@/types'
import { useBookshelfStore } from '@/stores/bookshelfStore'

interface Props {
  tags: TagData[]
}

const props = defineProps<Props>()

const emit = defineEmits<{
  search: [query: string]
  filterTag: [tagId: string]
}>()

const bookshelfStore = useBookshelfStore()
const searchQuery = ref('')
const showClearBtn = computed(() => searchQuery.value.length > 0)

// 处理搜索
function handleSearch() {
  emit('search', searchQuery.value)
}

// 清除搜索
function clearSearch() {
  searchQuery.value = ''
  emit('search', '')
}

// 处理输入（防抖）
let searchTimeout: ReturnType<typeof setTimeout>
function handleInput() {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(() => {
    handleSearch()
  }, 300)
}

// 处理标签点击 - 使用标签名称进行筛选
function handleTagClick(tagName: string) {
  emit('filterTag', tagName)
}

// 检查标签是否被选中 - 使用标签名称
function isTagSelected(tagName: string): boolean {
  return bookshelfStore.selectedTagIds.includes(tagName)
}
</script>

<template>
  <div class="filter-bar">
    <!-- 搜索框 -->
    <div class="search-box">
      <input
        v-model="searchQuery"
        type="text"
        placeholder="搜索书籍名称或标签..."
        autocomplete="off"
        @input="handleInput"
        @keypress.enter="handleSearch"
      >
      <button class="search-btn" @click="handleSearch">🔍</button>
      <button
        v-if="showClearBtn"
        class="clear-search-btn"
        @click="clearSearch"
      >
        ✕
      </button>
    </div>

    <!-- 标签筛选 -->
    <div v-if="tags.length > 0" class="tag-filter">
      <span class="filter-label">标签筛选:</span>
      <div class="tag-chips">
        <!-- 【复刻原版】使用 tag.name 作为唯一标识 -->
        <span
          v-for="tag in tags"
          :key="tag.name"
          class="tag-chip"
          :class="{ active: isTagSelected(tag.name) }"
          :style="tag.color ? { '--tag-color': tag.color, backgroundColor: isTagSelected(tag.name) ? tag.color : '' } : {}"
          @click="handleTagClick(tag.name)"
        >
          {{ tag.name }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ==================== 搜索和筛选栏样式 - 完整迁移自 bookshelf.css ==================== */

.filter-bar {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 24px;
    padding: 16px;
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 4px 12px rgb(0, 0, 0, 0.08);
}

/* 搜索框 */
.search-box {
    display: flex;
    align-items: center;
    gap: 8px;
    position: relative;
}

.search-box input {
    flex: 1;
    padding: 10px 16px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.95rem;
    background: var(--input-bg);
    color: var(--text-primary);
    transition: border-color 0.2s, box-shadow 0.2s;
}

.search-box input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgb(102, 126, 234, 0.1);
}

.search-btn, .clear-search-btn {
    padding: 10px 14px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 1rem;
}

.search-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.search-btn:hover {
    transform: scale(1.05);
}

.clear-search-btn {
    background: var(--hover-bg);
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
}

.clear-search-btn:hover {
    background: var(--border-color);
    color: var(--text-primary);
}

/* 标签筛选 */
.tag-filter {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

.filter-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    white-space: nowrap;
}

.tag-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    flex: 1;
}

.tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--hover-bg);
    color: var(--text-primary);
    border: 2px solid transparent;
}

.tag-chip:hover {
    background: var(--tag-color, #667eea);
    color: white;
}

.tag-chip.active {
    background: var(--tag-color, #667eea);
    color: white;
    border-color: var(--tag-color, #667eea);
    box-shadow: 0 2px 8px rgb(102, 126, 234, 0.3);
}

.tag-count {
    background: rgb(255, 255, 255, 0.3);
    padding: 1px 6px;
    border-radius: 10px;
    font-size: 0.7rem;
}

.no-tags {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-style: italic;
}
</style>
