import { ref, computed } from 'vue'
import type { Ref } from 'vue'
import type { NoteData, NoteType } from '@/types/insight'
import { filterValidInsightNotes, mapInsightApiNote } from '@/stores/insight/insightNotesModels'
import * as insightApi from '@/api/insight'

export interface UseInsightNotesOptions {
  currentBookId: Ref<string | null>
}

type NewNoteInput = Omit<NoteData, 'id' | 'createdAt' | 'updatedAt'>
  & Partial<Pick<NoteData, 'id' | 'createdAt' | 'updatedAt'>>

export function useInsightNotes(options: UseInsightNotesOptions) {
  const { currentBookId } = options
  let notesLoadRequestId = 0

  const notes = ref<NoteData[]>([])

  const noteTypeFilter = ref<NoteType | 'all'>('all')

  const isLoading = ref(false)

  const error = ref<string | null>(null)

  const filteredNotes = computed(() => {
    if (noteTypeFilter.value === 'all') {
      return notes.value
    }
    return notes.value.filter(note => note.type === noteTypeFilter.value)
  })

  function getStorageKey(): string | null {
    return currentBookId.value ? `manga_notes_${currentBookId.value}` : null
  }

  function isActiveNotesLoad(requestId: number, requestedBookId: string): boolean {
    return requestId === notesLoadRequestId && currentBookId.value === requestedBookId
  }

  function saveNotesToStorage(): void {
    const storageKey = getStorageKey()
    if (!storageKey) return
    try {
      localStorage.setItem(storageKey, JSON.stringify(notes.value))
    } catch {
      return
    }
  }

  function loadNotesFromStorage(): void {
    const storageKey = getStorageKey()
    if (!storageKey) {
      notes.value = []
      return
    }
    try {
      const stored = localStorage.getItem(storageKey)
      const parsed = stored ? JSON.parse(stored) : []
      notes.value = Array.isArray(parsed) ? filterValidInsightNotes(parsed) : []
    } catch {
      notes.value = []
    }
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
      const response = await insightApi.getNotes(requestedBookId)
      if (!isActiveNotesLoad(requestId, requestedBookId)) return
      if (!response.success) {
        error.value = response.error || '加载笔记失败'
        loadNotesFromStorage()
        return
      }
      if (response.notes) {
        notes.value = response.notes.map(mapInsightApiNote)
        saveNotesToStorage()
      }
    } catch (e) {
      if (!isActiveNotesLoad(requestId, requestedBookId)) return
      error.value = e instanceof Error ? e.message : '加载笔记失败'
      loadNotesFromStorage()
    } finally {
      if (requestId === notesLoadRequestId) {
        isLoading.value = false
      }
    }
  }

  async function addNote(note: NewNoteInput): Promise<NoteData | null> {
    if (!currentBookId.value) return null

    const optimisticNote: NoteData = {
      id: note.id || `note_${Date.now()}`,
      type: note.type,
      content: note.content,
      pageNum: note.pageNum,
      title: note.title,
      tags: note.tags,
      question: note.question,
      answer: note.answer,
      citations: note.citations,
      comment: note.comment,
      createdAt: note.createdAt || new Date().toISOString(),
      updatedAt: note.updatedAt || new Date().toISOString()
    }
    notes.value.unshift(optimisticNote)
    saveNotesToStorage()

    if (note.id) {
      return optimisticNote
    }

    function rollbackOptimisticNote(): void {
      notes.value = notes.value.filter(existing => existing.id !== optimisticNote.id)
      saveNotesToStorage()
    }

    try {
      const response = await insightApi.createNote(currentBookId.value, {
        type: note.type,
        content: note.content,
        page_num: note.pageNum,
        title: note.title,
        tags: note.tags,
        question: note.question,
        answer: note.answer,
        citations: note.citations,
        comment: note.comment
      })

      if (response.success && response.note) {
        const newNote = mapInsightApiNote(response.note)
        const index = notes.value.findIndex(existing => existing.id === optimisticNote.id)
        if (index !== -1) {
          notes.value[index] = newNote
        } else {
          notes.value.unshift(newNote)
        }
        saveNotesToStorage()
        return newNote
      }
      error.value = response.error || '添加笔记失败'
    } catch (e) {
      error.value = e instanceof Error ? e.message : '添加笔记失败'
    }
    rollbackOptimisticNote()
    return null
  }

  async function updateNote(noteId: string, updates: Partial<NoteData>): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      const response = await insightApi.updateNote(currentBookId.value, noteId, {
        content: updates.content,
        page_num: updates.pageNum,
        title: updates.title,
        tags: updates.tags,
        question: updates.question,
        answer: updates.answer,
        citations: updates.citations,
        comment: updates.comment
      })

      if (response.success) {
        const index = notes.value.findIndex(n => n.id === noteId)
        if (index !== -1) {
          notes.value[index] = { ...notes.value[index], ...updates, updatedAt: new Date().toISOString() }
          saveNotesToStorage()
        }
        return true
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '更新笔记失败'
    }
    return false
  }

  async function deleteNote(noteId: string): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      const response = await insightApi.deleteNote(currentBookId.value, noteId)

      if (response.success) {
        notes.value = notes.value.filter(n => n.id !== noteId)
        saveNotesToStorage()
        return true
      }
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
    notes.value = []
  }

  return {
    notes,
    noteTypeFilter,
    filteredNotes,
    isLoading,
    error,
    loadNotes,
    loadNotesFromStorage,
    saveNotesToStorage,
    addNote,
    updateNote,
    deleteNote,
    setNoteTypeFilter,
    clearNotes
  }
}
