/**
 * 问答管理 Composable
 *
 * 管理 Insight 问答历史
 */

import { ref } from 'vue'
import type { Ref } from 'vue'

/** 问答消息 */
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

export interface UseInsightQAOptions {
  currentBookId: Ref<string | null>
}

export function useInsightQA(options: UseInsightQAOptions) {
  const { currentBookId } = options

  /** 问答历史 */
  const qaHistory = ref<QAMessage[]>([])

  /** 是否正在流式响应 */
  const isStreaming = ref(false)

  /**
   * 添加用户消息
   */
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

  /**
   * 添加助手消息（初始加载状态）
   */
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

  /**
   * 更新助手消息
   */
  function updateAssistantMessage(
    messageId: string,
    updates: Partial<Pick<QAMessage, 'content' | 'isLoading' | 'citations' | 'mode'>>
  ): void {
    const index = qaHistory.value.findIndex(m => m.id === messageId)
    if (index !== -1) {
      qaHistory.value[index] = { ...qaHistory.value[index], ...updates }
    }
  }

  /**
   * 追加内容到助手消息（流式响应）
   */
  function appendToAssistantMessage(messageId: string, chunk: string): void {
    const index = qaHistory.value.findIndex(m => m.id === messageId)
    if (index !== -1) {
      const message = qaHistory.value[index]
      if (message) {
        message.content += chunk
      }
    }
  }

  /**
   * 标记消息已保存为笔记
   */
  function markAsSaved(messageId: string): void {
    const index = qaHistory.value.findIndex(m => m.id === messageId)
    if (index !== -1) {
      const message = qaHistory.value[index]
      if (message) {
        message.saved = true
      }
    }
  }

  /**
   * 清空问答历史
   */
  function clearHistory(): void {
    qaHistory.value = []
  }

  /**
   * 删除最后一条消息（用于错误恢复）
   */
  function removeLastMessage(): void {
    if (qaHistory.value.length > 0) {
      qaHistory.value.pop()
    }
  }

  /**
   * 设置流式响应状态
   */
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
