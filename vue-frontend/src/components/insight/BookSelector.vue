<script setup lang="ts">
import { ref, computed } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import CustomSelect from '@/components/common/CustomSelect.vue'

const emit = defineEmits<{
  (e: 'select', bookId: string): void
}>()

const bookshelfStore = useBookshelfStore()

const books = computed(() => bookshelfStore.books)

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

const selectedBookId = ref('')

function handleSelect(value: string | number): void {
  const bookId = String(value)
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
      fit
      @change="handleSelect"
    />
  </div>
</template>

<style scoped>
.book-selector {
    width: 300px;
}

</style>
