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
/* ==================== 书籍选择器样式 ==================== */

.book-selector {
    width: 300px;
}

.book-selector :deep(.custom-select) {
    width: 100%;
}
</style>
