import { ref, computed } from 'vue'
import type { Ref } from 'vue'
import type { NoteData, NoteType } from '@/types/insight'
import { mapInsightApiNote } from '@/stores/insight/insightNotesModels'
import * as insightApi from '@/api/insight'

export interface UseInsightNotesOptions {
  currentBookId: Ref<string | null>
}

export type NewInsightNoteInput = Omit<
  NoteData,
  'id' | 'createdAt' | 'updatedAt'
>

export function useInsightNotes(options: UseInsightNotesOptions) {
  const { currentBookId } = options
  let notesLoadRequestId = 0
  let noteDetailRequestId = 0

  const notes = ref<NoteData[]>([])

  const noteTypeFilter = ref<NoteType | 'all'>('all')

  const isLoading = ref(false)
  const isLoadingMore = ref(false)
  const nextCursor = ref<string | null>(null)

  const error = ref<string | null>(null)

  const filteredNotes = computed(() => {
    if (noteTypeFilter.value === 'all') {
      return notes.value
    }
    return notes.value.filter(note => note.type === noteTypeFilter.value)
  })

  function isActiveNotesLoad(requestId: number, requestedBookId: string): boolean {
    return requestId === notesLoadRequestId && currentBookId.value === requestedBookId
  }

  async function loadNotes(): Promise<void> {
    const requestedBookId = currentBookId.value
    const requestId = ++notesLoadRequestId
    if (!requestedBookId) {
      notes.value = []
      return
    }

    isLoading.value = true
    error.value = null

    try {
      const loadedNotes = await insightApi.getNotes(requestedBookId)
      if (!isActiveNotesLoad(requestId, requestedBookId)) return
      notes.value = loadedNotes.items.map(mapInsightApiNote)
      nextCursor.value = loadedNotes.nextCursor
    } catch (e) {
      if (!isActiveNotesLoad(requestId, requestedBookId)) return
      error.value = e instanceof Error ? e.message : '加载笔记失败'
    } finally {
      if (requestId === notesLoadRequestId) {
        isLoading.value = false
      }
    }
  }

  async function loadMoreNotes(): Promise<void> {
    const requestedBookId = currentBookId.value
    const cursor = nextCursor.value
    const requestId = notesLoadRequestId
    if (!requestedBookId || !cursor || isLoadingMore.value) return
    isLoadingMore.value = true
    error.value = null
    try {
      const loadedNotes = await insightApi.getNotes(requestedBookId, undefined, cursor)
      if (!isActiveNotesLoad(requestId, requestedBookId)) return
      const known = new Set(notes.value.map(note => note.id))
      notes.value.push(
        ...loadedNotes.items.map(mapInsightApiNote).filter(note => !known.has(note.id))
      )
      nextCursor.value = loadedNotes.nextCursor
    } catch (e) {
      if (isActiveNotesLoad(requestId, requestedBookId)) {
        error.value = e instanceof Error ? e.message : '加载更多笔记失败'
      }
    } finally {
      if (isActiveNotesLoad(requestId, requestedBookId)) isLoadingMore.value = false
    }
  }

  async function loadNoteDetail(noteId: string): Promise<NoteData | null> {
    const requestedBookId = currentBookId.value
    const listRequestId = notesLoadRequestId
    const requestId = ++noteDetailRequestId
    if (!requestedBookId) return null
    try {
      const detail = mapInsightApiNote(await insightApi.getNoteDetail(noteId))
      if (
        requestId !== noteDetailRequestId
        || listRequestId !== notesLoadRequestId
        || currentBookId.value !== requestedBookId
      ) return null
      const index = notes.value.findIndex(note => note.id === noteId)
      if (index >= 0) notes.value[index] = detail
      return detail
    } catch (e) {
      if (
        requestId !== noteDetailRequestId
        || listRequestId !== notesLoadRequestId
        || currentBookId.value !== requestedBookId
      ) return null
      error.value = e instanceof Error ? e.message : '加载笔记详情失败'
      return null
    }
  }

  async function addNote(
    note: NewInsightNoteInput,
  ): Promise<NoteData | null> {
    if (!currentBookId.value) return null

    const optimisticNote: NoteData = {
      id: `note_${Date.now()}`,
      type: note.type,
      content: note.content,
      pageNum: note.pageNum,
      title: note.title,
      tags: note.tags,
      question: note.question,
      answer: note.answer,
      citations: note.citations,
      comment: note.comment,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    notes.value.unshift(optimisticNote)

    function rollbackOptimisticNote(): void {
      notes.value = notes.value.filter(existing => existing.id !== optimisticNote.id)
    }

    try {
      const createdNote = await insightApi.createNote(currentBookId.value, {
        type: note.type,
        content: note.content,
        pageNum: note.pageNum,
        title: note.title,
        tags: note.tags,
        question: note.question,
        answer: note.answer,
        citations: note.citations,
        comment: note.comment
      })

      const newNote = mapInsightApiNote(createdNote)
      const index = notes.value.findIndex(existing => existing.id === optimisticNote.id)
      if (index !== -1) {
        notes.value[index] = newNote
      } else {
        notes.value.unshift(newNote)
      }
      return newNote
    } catch (e) {
      error.value = e instanceof Error ? e.message : '添加笔记失败'
    }
    rollbackOptimisticNote()
    return null
  }

  async function updateNote(noteId: string, updates: Partial<NoteData>): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      const updatedNote = await insightApi.updateNote(currentBookId.value, noteId, {
        content: updates.content,
        pageNum: updates.pageNum,
        title: updates.title,
        tags: updates.tags,
        question: updates.question,
        answer: updates.answer,
        citations: updates.citations,
        comment: updates.comment
      })

      const index = notes.value.findIndex(note => note.id === noteId)
      if (index !== -1) {
        notes.value[index] = mapInsightApiNote(updatedNote)
      }
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : '更新笔记失败'
    }
    return false
  }

  async function deleteNote(noteId: string): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      await insightApi.deleteNote(noteId)
      notes.value = notes.value.filter(n => n.id !== noteId)
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : '删除笔记失败'
    }
    return false
  }

  function setNoteTypeFilter(filter: NoteType | 'all'): void {
    noteTypeFilter.value = filter
  }

  function clearNotes(): void {
    notesLoadRequestId += 1
    noteDetailRequestId += 1
    notes.value = []
    nextCursor.value = null
    isLoadingMore.value = false
  }

  return {
    notes,
    noteTypeFilter,
    filteredNotes,
    isLoading,
    isLoadingMore,
    nextCursor,
    error,
    loadNotes,
    loadMoreNotes,
    loadNoteDetail,
    addNote,
    updateNote,
    deleteNote,
    setNoteTypeFilter,
    clearNotes
  }
}
