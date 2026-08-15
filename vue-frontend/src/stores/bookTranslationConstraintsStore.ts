import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { updateBookTranslationConstraints } from '@/api/bookshelf'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'

export const useBookTranslationConstraintsStore = defineStore('bookTranslationConstraints', () => {
  const currentBookId = ref<string | null>(null)
  const currentRevision = ref<number | null>(null)
  const constraints = ref<BookTranslationConstraints>(createEmptyBookTranslationConstraints())
  const isSaving = ref(false)

  const isAvailable = computed(() => Boolean(currentBookId.value))
  const glossary = computed(() => constraints.value.glossary)
  const nonTranslate = computed(() => constraints.value.nonTranslate)

  function loadBookConstraints(
    bookId: string,
    payload: BookTranslationConstraints,
    revision: number,
  ): void {
    currentBookId.value = bookId
    currentRevision.value = revision
    constraints.value = payload
  }

  function resetBookConstraints(): void {
    currentBookId.value = null
    currentRevision.value = null
    constraints.value = createEmptyBookTranslationConstraints()
  }

  async function saveBookConstraints(nextConstraints: BookTranslationConstraints): Promise<void> {
    if (!currentBookId.value || currentRevision.value === null) {
      throw new Error('书籍翻译约束尚未加载')
    }
    if (isSaving.value) {
      throw new Error('书籍翻译约束正在保存')
    }

    const bookId = currentBookId.value
    const baseRevision = currentRevision.value
    isSaving.value = true
    try {
      const result = await updateBookTranslationConstraints(
        bookId,
        nextConstraints,
        baseRevision,
      )
      if (currentBookId.value === bookId && currentRevision.value === baseRevision) {
        constraints.value = result.constraints
        currentRevision.value = result.revision
      }
    } finally {
      isSaving.value = false
    }
  }

  return {
    bookId: currentBookId,
    revision: currentRevision,
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
