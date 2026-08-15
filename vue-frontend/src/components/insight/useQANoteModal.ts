import { ref, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import type { QAMessage } from '@/types/insight'
import { showToast } from '@/utils/toast'

type QANoteSourceMessage = Pick<QAMessage, 'citations' | 'content' | 'id'>

export function useQANoteModal(insightStore: ReturnType<typeof useInsightStore>) {
  const showNoteModal = ref(false)
  const pendingQAData = ref<{
    messageId: string
    bookId: string
    question: string
    answer: string
    citations: Array<{ page: number }>
  } | null>(null)
  const noteTitle = ref('')
  const noteComment = ref('')
  const isSavingNote = ref(false)
  let saveRequestSequence = 0

  function openNoteModal(message: QANoteSourceMessage): void {
    if (!insightStore.currentBookId) return

    const userMessage = insightStore.qaHistory.find(item => item.role === 'user')
    if (!userMessage) return
    pendingQAData.value = {
      messageId: message.id,
      bookId: insightStore.currentBookId,
      question: userMessage.content,
      answer: message.content,
      citations: message.citations || [],
    }
    noteTitle.value = ''
    noteComment.value = ''
    showNoteModal.value = true
  }

  function closeNoteModal(): void {
    if (isSavingNote.value) return
    resetNoteModal()
  }

  function resetNoteModal(): void {
    saveRequestSequence += 1
    isSavingNote.value = false
    showNoteModal.value = false
    pendingQAData.value = null
  }

  async function saveNote(): Promise<void> {
    const bookId = insightStore.currentBookId
    const pending = pendingQAData.value
    if (!bookId || !pending || pending.bookId !== bookId || isSavingNote.value) return
    const requestId = ++saveRequestSequence
    isSavingNote.value = true

    const customTitle = noteTitle.value.trim()
    const comment = noteComment.value.trim()

    const noteData = {
      type: 'qa' as const,
      title: customTitle || pending.question,
      content: pending.answer,
      question: pending.question,
      citations: pending.citations.map(citation => ({
        ...citation,
        content: '',
      })),
      ...(comment ? { comment } : {}),
    }

    try {
      await insightStore.addNote(noteData)
      if (
        requestId !== saveRequestSequence ||
        insightStore.currentBookId !== bookId ||
        pendingQAData.value !== pending
      )
        return
      const message = insightStore.qaHistory.find(item => item.id === pending.messageId)
      if (message) {
        message.saved = true
      }
      resetNoteModal()
    } catch (error) {
      if (requestId === saveRequestSequence && insightStore.currentBookId === bookId) {
        showToast(
          insightStore.notesError
            ?? (error instanceof Error ? error.message : '保存笔记失败'),
          'error',
        )
      }
    } finally {
      if (requestId === saveRequestSequence) isSavingNote.value = false
    }
  }

  watch(
    () => insightStore.currentBookId,
    () => resetNoteModal()
  )

  return {
    closeNoteModal,
    isSavingNote,
    noteComment,
    noteTitle,
    openNoteModal,
    pendingQAData,
    saveNote,
    showNoteModal,
  }
}
