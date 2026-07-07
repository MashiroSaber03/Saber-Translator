<script setup lang="ts">
import { ref, computed } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import ProductBookSelector from '@/components/product/ProductBookSelector.vue'

const emit = defineEmits<{
  (e: 'select', bookId: string): void
}>()

const bookshelfStore = useBookshelfStore()

const books = computed(() => bookshelfStore.books)

const selectedBookId = ref('')

function handleSelect(bookId: string): void {
  emit('select', bookId)
}
</script>

<template>
  <ProductBookSelector
    v-model="selectedBookId"
    class="insight-book-selector"
    :books="books"
    @select="handleSelect"
  />
</template>
