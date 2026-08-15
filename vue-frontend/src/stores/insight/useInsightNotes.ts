import { ref } from 'vue'
import type { Ref } from 'vue'
import type { NoteData, NoteType, NoteUpdateInput } from '@/types/insight'
import * as insightApi from '@/api/insight'

export interface UseInsightNotesOptions {
  currentBookId: Ref<string | null>
}

export type NewInsightNoteInput = Pick<NoteData, 'content' | 'type'>
  & Partial<Pick<NoteData, 'citations' | 'comment' | 'pageNum' | 'question' | 'tags' | 'title'>>

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

  function isActiveNotesLoad(
    requestId: number,
    requestedBookId: string,
    requestedFilter: NoteType | 'all'
  ): boolean {
    return requestId === notesLoadRequestId
      && currentBookId.value === requestedBookId
      && noteTypeFilter.value === requestedFilter
  }

  async function loadNotes(): Promise<void> {
    const requestedBookId = currentBookId.value
    const requestedFilter = noteTypeFilter.value
    const requestId = ++notesLoadRequestId
    if (!requestedBookId) {
      notes.value = []
      nextCursor.value = null
      isLoading.value = false
      isLoadingMore.value = false
      error.value = null
      return
    }

    isLoading.value = true
    isLoadingMore.value = false
    notes.value = []
    nextCursor.value = null
    error.value = null

    try {
      const loadedNotes = await insightApi.getNotes(
        requestedBookId,
        requestedFilter === 'all' ? undefined : requestedFilter
      )
      if (!isActiveNotesLoad(requestId, requestedBookId, requestedFilter)) return
      notes.value = loadedNotes.items
      nextCursor.value = loadedNotes.nextCursor
    } catch (e) {
      if (!isActiveNotesLoad(requestId, requestedBookId, requestedFilter)) return
      error.value = e instanceof Error ? e.message : '加载笔记失败'
    } finally {
      if (requestId === notesLoadRequestId) {
        isLoading.value = false
      }
    }
  }

  async function loadMoreNotes(): Promise<void> {
    const requestedBookId = currentBookId.value
    const requestedFilter = noteTypeFilter.value
    const cursor = nextCursor.value
    const requestId = notesLoadRequestId
    if (!requestedBookId || !cursor || isLoadingMore.value) return
    isLoadingMore.value = true
    error.value = null
    try {
      const loadedNotes = await insightApi.getNotes(
        requestedBookId,
        requestedFilter === 'all' ? undefined : requestedFilter,
        cursor
      )
      if (!isActiveNotesLoad(requestId, requestedBookId, requestedFilter)) return
      const known = new Set(notes.value.map(note => note.id))
      notes.value.push(
        ...loadedNotes.items.filter(note => !known.has(note.id))
      )
      nextCursor.value = loadedNotes.nextCursor
    } catch (e) {
      if (isActiveNotesLoad(requestId, requestedBookId, requestedFilter)) {
        error.value = e instanceof Error ? e.message : '加载更多笔记失败'
      }
    } finally {
      if (isActiveNotesLoad(requestId, requestedBookId, requestedFilter)) {
        isLoadingMore.value = false
      }
    }
  }

  async function loadNoteDetail(noteId: string): Promise<NoteData | null> {
    const requestedBookId = currentBookId.value
    const listRequestId = notesLoadRequestId
    const requestId = ++noteDetailRequestId
    if (!requestedBookId) return null
    try {
      const detail = await insightApi.getNoteDetail(noteId)
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
    const requestedBookId = currentBookId.value
    if (!requestedBookId) return null
    error.value = null
    try {
      const createdNote = await insightApi.createNote(requestedBookId, {
        type: note.type,
        content: note.content,
        pageNum: note.pageNum,
        title: note.title,
        tags: note.tags,
        question: note.question,
        citations: note.citations,
        comment: note.comment
      })

      const newNote = createdNote
      if (
        currentBookId.value === requestedBookId
        && (noteTypeFilter.value === 'all' || noteTypeFilter.value === newNote.type)
        && !notes.value.some(existing => existing.id === newNote.id)
      ) {
        notes.value.unshift(newNote)
      }
      return newNote
    } catch (e) {
      if (currentBookId.value === requestedBookId) {
        error.value = e instanceof Error ? e.message : '添加笔记失败'
      }
      throw e
    }
  }

  async function updateNote(noteId: string, updates: NoteUpdateInput): Promise<void> {
    const requestedBookId = currentBookId.value
    if (!requestedBookId) return
    error.value = null

    try {
      const updatedNote = await insightApi.updateNote(requestedBookId, noteId, updates)
      if (currentBookId.value !== requestedBookId) return

      const index = notes.value.findIndex(note => note.id === noteId)
      if (index !== -1) {
        if (noteTypeFilter.value === 'all' || noteTypeFilter.value === updatedNote.type) {
          notes.value[index] = updatedNote
        } else {
          notes.value.splice(index, 1)
        }
      }
    } catch (e) {
      if (currentBookId.value === requestedBookId) {
        error.value = e instanceof Error ? e.message : '更新笔记失败'
      }
      throw e
    }
  }

  async function deleteNote(noteId: string): Promise<void> {
    const requestedBookId = currentBookId.value
    if (!requestedBookId) return
    error.value = null

    try {
      await insightApi.deleteNote(noteId)
      if (currentBookId.value !== requestedBookId) return
      notes.value = notes.value.filter(n => n.id !== noteId)
    } catch (e) {
      if (currentBookId.value === requestedBookId) {
        error.value = e instanceof Error ? e.message : '删除笔记失败'
      }
      throw e
    }
  }

  async function setNoteTypeFilter(filter: NoteType | 'all'): Promise<void> {
    if (noteTypeFilter.value === filter) return
    noteTypeFilter.value = filter
    await loadNotes()
  }

  function clearNotes(): void {
    notesLoadRequestId += 1
    noteDetailRequestId += 1
    notes.value = []
    nextCursor.value = null
    isLoading.value = false
    isLoadingMore.value = false
    error.value = null
  }

  return {
    notes,
    noteTypeFilter,
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
