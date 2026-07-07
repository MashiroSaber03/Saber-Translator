import { ref } from 'vue'

export interface QAMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  isLoading?: boolean
  mode?: string
  citations?: Array<{ page: number }>
  saved?: boolean
}

export function useInsightQA() {
  const qaHistory = ref<QAMessage[]>([])
  const isStreaming = ref(false)

  function addUserMessage(content: string): QAMessage {
    const message: QAMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    }
    qaHistory.value.push(message)
    return message
  }

  function addAssistantMessage(mode?: string): QAMessage {
    const message: QAMessage = {
      id: `assistant_${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isLoading: true,
      mode
    }
    qaHistory.value.push(message)
    return message
  }

  function updateAssistantMessage(
    messageId: string,
    updates: Partial<Pick<QAMessage, 'content' | 'isLoading' | 'citations' | 'mode'>>
  ): void {
    const index = qaHistory.value.findIndex(m => m.id === messageId)
    if (index !== -1) {
      qaHistory.value[index] = { ...qaHistory.value[index], ...updates }
    }
  }

  function appendToAssistantMessage(messageId: string, chunk: string): void {
    const index = qaHistory.value.findIndex(m => m.id === messageId)
    if (index !== -1) {
      const message = qaHistory.value[index]
      if (message) {
        message.content += chunk
      }
    }
  }

  function markAsSaved(messageId: string): void {
    const index = qaHistory.value.findIndex(m => m.id === messageId)
    if (index !== -1) {
      const message = qaHistory.value[index]
      if (message) {
        message.saved = true
      }
    }
  }

  function clearHistory(): void {
    qaHistory.value = []
  }

  function removeLastMessage(): void {
    if (qaHistory.value.length > 0) {
      qaHistory.value.pop()
    }
  }

  function setStreaming(value: boolean): void {
    isStreaming.value = value
  }

  return {
    qaHistory,
    isStreaming,
    addUserMessage,
    addAssistantMessage,
    updateAssistantMessage,
    appendToAssistantMessage,
    markAsSaved,
    clearHistory,
    removeLastMessage,
    setStreaming
  }
}
