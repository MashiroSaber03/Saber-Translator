import { ref } from 'vue'
import { useInsightStore } from '@/stores/insightStore'

interface QANoteSourceMessage {
  id: string
  content: string
  citations?: Array<{ page: number }>
}

export function useQANoteModal(insightStore: ReturnType<typeof useInsightStore>) {
  const showNoteModal = ref(false)
  const pendingQAData = ref<{
    messageId: string
    question: string
    answer: string
    citations: Array<{ page: number }>
  } | null>(null)
  const noteTitle = ref('')
  const noteComment = ref('')

  function openNoteModal(message: QANoteSourceMessage): void {
    if (!insightStore.currentBookId) return

    const userMessage = insightStore.qaHistory.find(item => item.role === 'user')
    pendingQAData.value = {
      messageId: message.id,
      question: userMessage?.content || '',
      answer: message.content,
      citations: message.citations || [],
    }
    noteTitle.value = ''
    noteComment.value = ''
    showNoteModal.value = true
  }

  function closeNoteModal(): void {
    showNoteModal.value = false
    pendingQAData.value = null
  }

  async function saveNote(): Promise<void> {
    if (!insightStore.currentBookId || !pendingQAData.value) return

    const now = new Date().toISOString()
    const noteData = {
      id: Date.now().toString(),
      type: 'qa' as const,
      title: noteTitle.value || pendingQAData.value.question.substring(0, 30),
      content: pendingQAData.value.answer,
      question: pendingQAData.value.question,
      answer: pendingQAData.value.answer,
      citations: pendingQAData.value.citations,
      comment: noteComment.value || undefined,
      createdAt: now,
      updatedAt: now,
    }

    try {
      await insightStore.addNote(noteData)
      const message = insightStore.qaHistory.find(item => item.id === pendingQAData.value?.messageId)
      if (message) {
        message.saved = true
      }
      closeNoteModal()
    } catch (error) {
      console.error('保存笔记失败:', error)
      alert('保存笔记失败')
    }
  }

  return {
    closeNoteModal,
    noteComment,
    noteTitle,
    openNoteModal,
    pendingQAData,
    saveNote,
    showNoteModal,
  }
}
