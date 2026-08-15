import { ref } from 'vue'
import type { QAMessage } from '@/types/insight'

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
