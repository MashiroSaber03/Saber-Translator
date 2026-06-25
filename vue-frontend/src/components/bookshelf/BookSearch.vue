<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 书籍搜索和标签筛选组件
 */

import { ref, computed, onUnmounted } from 'vue'
import type { TagData } from '@/types'

interface Props {
  tags: TagData[]
  selectedTagNames?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  selectedTagNames: () => [],
})

const emit = defineEmits<{
  search: [query: string]
  filterTag: [tagName: string]
}>()

const searchQuery = ref('')
const showClearBtn = computed(() => searchQuery.value.length > 0)

function handleSearch() {
  emit('search', searchQuery.value)
}

function clearSearch() {
  clearPendingSearch()
  searchQuery.value = ''
  emit('search', '')
}

let searchTimeout: ReturnType<typeof setTimeout> | null = null

function clearPendingSearch() {
  if (!searchTimeout) return
  clearTimeout(searchTimeout)
  searchTimeout = null
}

function handleInput() {
  clearPendingSearch()
  searchTimeout = setTimeout(() => {
    searchTimeout = null
    handleSearch()
  }, 300)
}

function handleTagClick(tagName: string) {
  emit('filterTag', tagName)
}

function isTagSelected(tagName: string): boolean {
  return props.selectedTagNames.includes(tagName)
}

onUnmounted(clearPendingSearch)
</script>

<template>
  <div class="filter-bar">
    <!-- 搜索框 -->
    <div class="search-box">
      <UiInput
        v-model="searchQuery"
        class="book-search-input"
        type="text"
        placeholder="搜索书籍名称或标签..."
        autocomplete="off"
        aria-label="搜索书籍"
        @input="handleInput"
        @keydown.enter="handleSearch"
      />
      <UiButton
        variant="toolbar"
        class="search-btn"
        aria-label="搜索"
        title="搜索"
        @click="handleSearch"
      >
        🔍
      </UiButton>
      <UiButton
        v-if="showClearBtn"
        variant="toolbar"
        class="clear-search-btn"
        aria-label="清除搜索"
        title="清除搜索"
        @click="clearSearch"
      >
        ✕
      </UiButton>
    </div>

    <!-- 标签筛选 -->
    <div v-if="tags.length > 0" class="tag-filter">
      <span class="filter-label">标签筛选:</span>
      <div class="tag-chips">
        <UiButton
          v-for="tag in tags"
          :key="tag.name"
          variant="toolbar"
          type="button"
          class="tag-chip"
          :class="{ active: isTagSelected(tag.name) }"
          :aria-pressed="isTagSelected(tag.name) ? 'true' : 'false'"
          :style="tag.color ? { '--tag-color': tag.color, backgroundColor: isTagSelected(tag.name) ? tag.color : '' } : {}"
          @click="handleTagClick(tag.name)"
        >
          {{ tag.name }}
        </UiButton>
      </div>
    </div>
  </div>
</template>

<style scoped>
.filter-bar {
  --book-search-panel-shadow: rgba(0, 0, 0, .08);
  --book-search-focus-shadow: rgba(102, 126, 234, .1);
  --book-search-active-chip-shadow: rgba(102, 126, 234, .3);
}

.filter-bar {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 24px;
    padding: 16px;
    background: var(--color-surface-card);
    border-radius: 12px;
    box-shadow: 0 4px 12px var(--book-search-panel-shadow);
}

/* 搜索框 */
.search-box {
    display: flex;
    align-items: center;
    gap: 8px;
    position: relative;
}

.book-search-input {
    flex: 1;
    padding: 10px 16px;
    border: 1px solid var(--color-border-muted);
    border-radius: 8px;
    font-size: 0.95rem;
    background: var(--color-surface-input);
    color: var(--color-text-default);
    transition: border-color 0.2s, box-shadow 0.2s;
}

.book-search-input:focus {
    outline: none;
    border-color: var(--color-border-brand-gradient);
    box-shadow: 0 0 0 3px var(--book-search-focus-shadow);
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
    background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
    color: var(--color-text-inverse);
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
    appearance: none;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.2s;
    background: var(--color-surface-interactive-hover);
    color: var(--color-text-default);
    border: 2px solid transparent;
}

.tag-chip:hover {
    background: var(--tag-color, var(--color-action-brand));
    color: var(--color-text-inverse);
}

.tag-chip.active {
    background: var(--tag-color, var(--color-action-brand));
    color: var(--color-text-inverse);
    border-color: var(--tag-color, var(--color-border-brand-gradient));
    box-shadow: 0 2px 8px var(--book-search-active-chip-shadow);
}
</style>
