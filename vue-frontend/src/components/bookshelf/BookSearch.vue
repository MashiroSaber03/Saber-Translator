<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import ProductSearchToolbar from '@/components/product/ProductSearchToolbar.vue'

import { ref, computed, onUnmounted, watch } from 'vue'
import type { TagData } from '@/types'

interface Props {
  tags: TagData[]
  query?: string
  selectedTagNames?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  query: '',
  selectedTagNames: () => [],
})

const emit = defineEmits<{
  search: [query: string]
  filterTag: [tagName: string]
}>()

const searchQuery = ref(props.query)
const tagItems = computed<ProductChipItem[]>(() => props.tags.map(tag => {
  const selected = isTagSelected(tag.name)

  if (!selected) {
    return {
      id: tag.name,
      label: tag.name,
      ariaLabel: `筛选标签 ${tag.name}`,
      interactive: true,
      selected: false,
      tone: 'neutral',
    }
  }

  return {
    id: tag.name,
    label: tag.name,
    ariaLabel: `取消筛选标签 ${tag.name}`,
    interactive: true,
    selected: true,
    tone: 'custom',
    backgroundColor: tag.color,
    borderColor: tag.color,
    textColor: 'var(--color-text-inverse)',
  }
}))

function handleSearch() {
  clearPendingSearch()
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

function handleSearchQueryUpdate(value: string) {
  searchQuery.value = value
  clearPendingSearch()
  searchTimeout = setTimeout(() => {
    searchTimeout = null
    handleSearch()
  }, 300)
}

function handleTagSelect(tagId: string | number) {
  emit('filterTag', String(tagId))
}

function isTagSelected(tagName: string): boolean {
  return props.selectedTagNames.includes(tagName)
}

watch(() => props.query, (query) => {
  if (query === searchQuery.value) return
  clearPendingSearch()
  searchQuery.value = query
})

onUnmounted(clearPendingSearch)
</script>

<template>
  <ProductSearchToolbar class="book-search" aria-label="书籍搜索和标签筛选">
    <template #search>
      <ProductSearchField
        :model-value="searchQuery"
        class="book-search__field"
        placeholder="搜索书籍名称或标签..."
        aria-label="搜索书籍"
        :show-icon="false"
        @update:model-value="handleSearchQueryUpdate"
        @search="handleSearch"
        @clear="clearSearch"
      />
      <UiButton
        variant="primary"
        class="book-search__submit-action"
        aria-label="搜索"
        title="搜索"
        @click="handleSearch"
      >
        🔍
      </UiButton>
    </template>

    <template v-if="tagItems.length > 0" #filters>
      <ProductChipList
        class="book-search__tags"
        aria-label="标签筛选"
        label="标签筛选:"
        :items="tagItems"
        @select="handleTagSelect"
      />
    </template>
  </ProductSearchToolbar>
</template>

<style scoped>
.book-search__field {
  --product-search-field-input-padding: 10px 16px;
  --product-search-field-radius: 8px;

  flex: 1;
  min-width: 0;
}

.book-search__submit-action {
  flex: 0 0 auto;
  min-width: 46px;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 1rem;
}

.book-search__tags {
  --product-chip-list-text: var(--color-text-supporting);
  --product-chip-list-label-text: var(--color-text-supporting);
}
</style>
