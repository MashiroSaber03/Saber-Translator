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

  function clearHistory(): void {
    qaHistory.value = []
  }

  function setStreaming(value: boolean): void {
    isStreaming.value = value
  }

  return {
    qaHistory,
    isStreaming,
    clearHistory,
    setStreaming
  }
}
