<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 书籍搜索和标签筛选组件
 */

import { ref, computed } from 'vue'
import type { TagData } from '@/types'
import { useBookshelfStore } from '@/stores/bookshelfStore'

interface Props {
  tags: TagData[]
}

defineProps<Props>()

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
      <UiInput
        v-model="searchQuery"
        type="text"
        placeholder="搜索书籍名称或标签..."
        autocomplete="off"
        @input="handleInput"
        @keypress.enter="handleSearch"
      />
      <UiButton variant="toolbar" class="search-btn" @click="handleSearch">🔍</UiButton>
      <UiButton
        variant="toolbar"
        v-if="showClearBtn"
        class="clear-search-btn"
        @click="clearSearch"
      >
        ✕
      </UiButton>
    </div>

    <!-- 标签筛选 -->
    <div v-if="tags.length > 0" class="tag-filter">
      <span class="filter-label">标签筛选:</span>
      <div class="tag-chips">
        <!-- 使用 tag.name 作为唯一标识 -->
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
.filter-bar {
  --book-search-shadow-default: rgba(0, 0, 0, .08);
  --book-search-shadow-raised: rgba(102, 126, 234, .1);
  --book-search-shadow-floating: rgba(102, 126, 234, .3);
  --book-search-surface-base: rgba(255, 255, 255, .3);
}

/* ==================== 搜索和筛选栏样式 - 当前样式 ==================== */

.filter-bar {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 24px;
    padding: 16px;
    background: var(--color-surface-card);
    border-radius: 12px;
    box-shadow: 0 4px 12px var(--book-search-shadow-default);
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
    border: 1px solid var(--color-border-muted);
    border-radius: 8px;
    font-size: 0.95rem;
    background: var(--color-surface-input);
    color: var(--color-text-default);
    transition: border-color 0.2s, box-shadow 0.2s;
}

.search-box input:focus {
    outline: none;
    border-color: var(--color-border-brand-gradient);
    box-shadow: 0 0 0 3px var(--book-search-shadow-raised);
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
    background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
    color: white;
}

.search-btn:hover {
    transform: scale(1.05);
}

.clear-search-btn {
    background: var(--color-surface-interactive-hover);
    color: var(--color-text-supporting);
    display: flex;
    align-items: center;
    justify-content: center;
}

.clear-search-btn:hover {
    background: var(--color-border-muted);
    color: var(--color-text-default);
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
    color: var(--color-text-supporting);
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
    background: var(--color-surface-interactive-hover);
    color: var(--color-text-default);
    border: 2px solid transparent;
}

.tag-chip:hover {
    background: var(--tag-color, var(--color-surface-brand-gradient-start));
    color: white;
}

.tag-chip.active {
    background: var(--tag-color, var(--color-surface-brand-gradient-start));
    color: white;
    border-color: var(--tag-color, var(--color-border-brand-gradient));
    box-shadow: 0 2px 8px var(--book-search-shadow-floating);
}

.tag-count {
    background: var(--book-search-surface-base);
    padding: 1px 6px;
    border-radius: 10px;
    font-size: 0.7rem;
}

.no-tags {
    color: var(--color-text-supporting);
    font-size: 0.85rem;
    font-style: italic;
}
</style>
