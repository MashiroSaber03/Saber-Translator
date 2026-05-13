import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { updateBook } from '@/api/bookshelf'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import {
  createEmptyBookTranslationConstraints,
  normalizeBookTranslationConstraints,
} from '@/utils/bookTranslationConstraints'

export const useBookTranslationConstraintsStore = defineStore('bookTranslationConstraints', () => {
  const currentBookId = ref<string | null>(null)
  const constraints = ref<BookTranslationConstraints>(createEmptyBookTranslationConstraints())
  const isSaving = ref(false)

  const isAvailable = computed(() => Boolean(currentBookId.value))
  const glossary = computed(() => constraints.value.glossary)
  const nonTranslate = computed(() => constraints.value.non_translate)

  function loadBookConstraints(bookId: string, payload?: Partial<BookTranslationConstraints> | null): void {
    currentBookId.value = bookId
    constraints.value = normalizeBookTranslationConstraints(payload)
  }

  function resetBookConstraints(): void {
    currentBookId.value = null
    constraints.value = createEmptyBookTranslationConstraints()
  }

  async function saveBookConstraints(nextConstraints?: BookTranslationConstraints): Promise<boolean> {
    if (!currentBookId.value) {
      return false
    }

    const payload = normalizeBookTranslationConstraints(nextConstraints ?? constraints.value)
    isSaving.value = true
    try {
      const response = await updateBook(currentBookId.value, {
        translation_constraints: payload,
      })
      if (!response.success || !response.book) {
        return false
      }
      constraints.value = normalizeBookTranslationConstraints(response.book.translation_constraints)
      return true
    } catch (error) {
      console.error('保存书籍级翻译约束失败:', error)
      return false
    } finally {
      isSaving.value = false
    }
  }

  return {
    bookId: currentBookId,
    constraints,
    glossary,
    nonTranslate,
    isAvailable,
    isSaving,
    loadBookConstraints,
    resetBookConstraints,
    saveBookConstraints,
  }
})
