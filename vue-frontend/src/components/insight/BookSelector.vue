<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { getBooks } from '@/api/bookshelf'
import type { BookData } from '@/types'
import ProductBookSelector from '@/components/product/ProductBookSelector.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'

const emit = defineEmits<{
  (e: 'select', bookId: string): void
}>()

const books = ref<BookData[]>([])
const selectedBookId = ref('')
const isLoading = ref(true)
const error = ref<string | null>(null)
let loadSequence = 0
let isMounted = false

function handleSelect(bookId: string): void {
  emit('select', bookId)
}

async function loadBooks(): Promise<void> {
  const loadId = ++loadSequence
  isLoading.value = true
  error.value = null
  try {
    const loadedBooks = await getBooks()
    if (!isMounted || loadId !== loadSequence) return
    books.value = loadedBooks
  } catch (loadError) {
    if (!isMounted || loadId !== loadSequence) return
    error.value = loadError instanceof Error ? loadError.message : '加载书籍失败'
  } finally {
    if (isMounted && loadId === loadSequence) isLoading.value = false
  }
}

onMounted(() => {
  isMounted = true
  void loadBooks()
})

onUnmounted(() => {
  isMounted = false
  loadSequence += 1
})
</script>

<template>
  <ProductStatusBanner
    v-if="error"
    class="insight-book-selector__status"
    tone="danger"
    title="书籍列表加载失败"
    aria-live="assertive"
  >
    {{ error }}
    <template #actions>
      <UiButton size="sm" variant="secondary" @click="loadBooks">重试</UiButton>
    </template>
  </ProductStatusBanner>

  <ProductStatusBanner
    v-else-if="isLoading"
    class="insight-book-selector__status"
    tone="neutral"
    title="正在加载书籍"
    aria-live="polite"
  />

  <ProductStatusBanner
    v-else-if="books.length === 0"
    class="insight-book-selector__status"
    tone="neutral"
    title="书架中暂无书籍"
  >
    请先返回书架添加书籍。
  </ProductStatusBanner>

  <ProductBookSelector
    v-else
    v-model="selectedBookId"
    class="insight-book-selector"
    :books="books"
    placeholder="-- 选择书籍 --"
    @select="handleSelect"
  />
</template>

<style scoped>
.insight-book-selector,
.insight-book-selector__status {
  width: min(100%, 300px);
}
</style>
