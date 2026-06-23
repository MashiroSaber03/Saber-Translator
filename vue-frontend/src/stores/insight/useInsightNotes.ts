/**
 * 笔记管理 Composable
 *
 * 管理 Insight 笔记的增删改查
 */

import { ref, computed } from 'vue'
import type { Ref } from 'vue'
import type { NoteData, NoteType } from '@/types/insight'
import { toCamelCase } from '@/types/insight/converters'
import * as insightApi from '@/api/insight'

export interface UseInsightNotesOptions {
  currentBookId: Ref<string | null>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNoteType(value: unknown): value is NoteType {
  return value === 'text' || value === 'qa'
}

function hasOptionalString(value: Record<string, unknown>, key: string): boolean {
  return value[key] === undefined || typeof value[key] === 'string'
}

function hasOptionalNumber(value: Record<string, unknown>, key: string): boolean {
  return value[key] === undefined || (typeof value[key] === 'number' && Number.isFinite(value[key]))
}

function isCitation(value: unknown): value is { page: number; content: string } {
  return isRecord(value)
    && typeof value.page === 'number'
    && Number.isFinite(value.page)
    && typeof value.content === 'string'
}

function isNoteData(value: unknown): value is NoteData {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string' || !isNoteType(value.type) || typeof value.content !== 'string') {
    return false
  }
  if (!hasOptionalNumber(value, 'pageNum')) return false
  if (!hasOptionalString(value, 'createdAt')) return false
  if (!hasOptionalString(value, 'updatedAt')) return false
  if (!hasOptionalString(value, 'title')) return false
  if (!hasOptionalString(value, 'question')) return false
  if (!hasOptionalString(value, 'answer')) return false
  if (!hasOptionalString(value, 'comment')) return false
  if (value.tags !== undefined && (!Array.isArray(value.tags) || !value.tags.every(tag => typeof tag === 'string'))) {
    return false
  }
  if (value.citations !== undefined && (!Array.isArray(value.citations) || !value.citations.every(isCitation))) {
    return false
  }
  return true
}

export function useInsightNotes(options: UseInsightNotesOptions) {
  const { currentBookId } = options

  /** 笔记列表 */
  const notes = ref<NoteData[]>([])

  /** 笔记类型筛选 */
  const noteTypeFilter = ref<NoteType | 'all'>('all')

  /** 是否正在加载 */
  const isLoading = ref(false)

  /** 错误信息 */
  const error = ref<string | null>(null)

  /** 过滤后的笔记列表 */
  const filteredNotes = computed(() => {
    if (noteTypeFilter.value === 'all') {
      return notes.value
    }
    return notes.value.filter(note => note.type === noteTypeFilter.value)
  })

  function getStorageKey(): string | null {
    return currentBookId.value ? `manga_notes_${currentBookId.value}` : null
  }

  function saveNotesToStorage(): void {
    const storageKey = getStorageKey()
    if (!storageKey) return
    try {
      localStorage.setItem(storageKey, JSON.stringify(notes.value))
    } catch (e) {
      console.warn('保存本地笔记失败:', e)
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
      notes.value = Array.isArray(parsed) ? parsed.filter(isNoteData) : []
    } catch (e) {
      console.warn('加载本地笔记失败:', e)
      notes.value = []
    }
  }

  /**
   * 从 API 加载笔记
   */
  async function loadNotes(): Promise<void> {
    if (!currentBookId.value) {
      notes.value = []
      return
    }

    isLoading.value = true
    error.value = null

    try {
      const response = await insightApi.getNotes(currentBookId.value)
      if (response.success && response.notes) {
        // 使用转换器自动将 snake_case 转为 camelCase
        notes.value = response.notes.map(note =>
          toCamelCase(note as Record<string, unknown>) as unknown as NoteData
        )
        saveNotesToStorage()
      }
    } catch (e) {
      console.error('加载笔记失败:', e)
      error.value = e instanceof Error ? e.message : '加载笔记失败'
      loadNotesFromStorage()
    } finally {
      isLoading.value = false
    }
  }

  /**
   * 添加笔记
   */
  async function addNote(note: Omit<NoteData, 'id' | 'createdAt' | 'updatedAt'>): Promise<NoteData | null> {
    if (!currentBookId.value) return null

    const noteInput = note as Partial<NoteData>
    const optimisticNote: NoteData = {
      id: noteInput.id || `note_${Date.now()}`,
      type: note.type,
      content: note.content,
      pageNum: note.pageNum,
      title: note.title,
      tags: note.tags,
      question: note.question,
      answer: note.answer,
      citations: note.citations,
      comment: note.comment,
      createdAt: noteInput.createdAt || new Date().toISOString(),
      updatedAt: noteInput.updatedAt || new Date().toISOString()
    }
    notes.value.unshift(optimisticNote)
    saveNotesToStorage()

    if (noteInput.id) {
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
        // 使用转换器自动转换
        const newNote = toCamelCase(response.note as Record<string, unknown>) as unknown as NoteData
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
      console.error('添加笔记失败:', e)
      error.value = e instanceof Error ? e.message : '添加笔记失败'
    }
    rollbackOptimisticNote()
    return null
  }

  /**
   * 更新笔记
   */
  async function updateNote(noteId: string, updates: Partial<NoteData>): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      // 传递完整的更新数据，包括扩展字段
      const response = await insightApi.updateNote(currentBookId.value, noteId, {
        content: updates.content,
        page_num: updates.pageNum,
        // 扩展字段
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
      console.error('更新笔记失败:', e)
      error.value = e instanceof Error ? e.message : '更新笔记失败'
    }
    return false
  }

  /**
   * 删除笔记
   */
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
      console.error('删除笔记失败:', e)
      error.value = e instanceof Error ? e.message : '删除笔记失败'
    }
    return false
  }

  /**
   * 设置笔记类型筛选
   */
  function setNoteTypeFilter(filter: NoteType | 'all'): void {
    noteTypeFilter.value = filter
  }

  /**
   * 清空笔记（切换书籍时调用）
   */
  function clearNotes(): void {
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
