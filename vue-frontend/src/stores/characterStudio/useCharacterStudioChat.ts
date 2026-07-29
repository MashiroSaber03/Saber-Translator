import { ref, type Ref } from 'vue'

import {
  abortCharacterStudioChatOperation,
  regenerateCharacterStudioChatMessage,
  streamCharacterStudioChatMessage,
} from '@/api/characterStudio'
import {
  applyAssistantRuntimeState,
  applyAssistantStreamContent,
  findRegenerationUserMessageIndex,
} from '@/stores/characterStudioChatSession'
import type {
  CharacterStudioChatAttachment,
  CharacterStudioChatMessage,
  CharacterStudioChatSession,
  CharacterStudioDocument,
} from '@/types/characterStudio'
import { deepClone } from '@/utils/deepClone'

interface CharacterStudioChatOptions {
  bookId: Ref<string>
  currentDocument: Ref<CharacterStudioDocument | null>
  activeChatSession: Ref<CharacterStudioChatSession | null>
  activeWorkspaceTab: Ref<'chat' | 'assistant' | 'runtime'>
  errorMessage: Ref<string>
  applySession: (session: CharacterStudioChatSession) => void
  flushPendingRehydrate: () => Promise<void>
}

export function useCharacterStudioChat(options: CharacterStudioChatOptions) {
  const isChatStreaming = ref(false)
  const activeChatOperationId = ref<string | null>(null)
  let abortController: AbortController | null = null
  let rollbackSession: CharacterStudioChatSession | null = null
  let streamRunId = 0

  function createActionError(error: unknown, fallback: string): Error {
    const nextError = error instanceof Error ? error : new Error(fallback)
    options.errorMessage.value = nextError.message || fallback
    return nextError
  }

  function clearErrorMessage(): void {
    options.errorMessage.value = ''
  }

  function createOptimisticAttachment(file: File): CharacterStudioChatAttachment {
    return {
      attachment_id: `temp-att-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      filename: file.name,
      mime_type: file.type || 'application/octet-stream',
      asset_path: URL.createObjectURL(file),
      created_at: new Date().toISOString(),
    }
  }

  function revokeAttachmentUrls(attachments: CharacterStudioChatAttachment[]): void {
    attachments.forEach(item => {
      if (item.asset_path?.startsWith('blob:')) URL.revokeObjectURL(item.asset_path)
    })
  }

  function revokeOptimisticSessionAssets(session: CharacterStudioChatSession | null): void {
    session?.messages.forEach(message => revokeAttachmentUrls(message.attachments || []))
  }

  function createOptimisticMessage(
    role: 'user' | 'assistant',
    content: string,
    attachments: CharacterStudioChatAttachment[] = [],
  ): CharacterStudioChatMessage {
    const now = new Date().toISOString()
    return {
      message_id: `temp-msg-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      role,
      content,
      attachments,
      runtime_log: [],
      variables_snapshot: deepClone(options.activeChatSession.value?.variables || {}),
      generation_meta: {},
      created_at: now,
      updated_at: now,
    }
  }

  function isActiveStream(
    runId: number,
    controller: AbortController,
    requestedBookId: string,
    requestedDocId: string,
    requestedSessionId: string,
  ): boolean {
    return runId === streamRunId
      && abortController === controller
      && options.bookId.value === requestedBookId
      && options.currentDocument.value?.id === requestedDocId
      && options.activeChatSession.value?.session_id === requestedSessionId
  }

  function abortActiveChatStream(): void {
    if (!abortController) return
    revokeOptimisticSessionAssets(options.activeChatSession.value)
    abortController.abort()
    abortController = null
    activeChatOperationId.value = null
    if (rollbackSession) {
      options.activeChatSession.value = rollbackSession
      rollbackSession = null
    }
  }

  async function abortActiveChatOperation(): Promise<void> {
    const operationId = activeChatOperationId.value
    const sessionId = options.activeChatSession.value?.session_id
    if (!operationId || !sessionId) return
    clearErrorMessage()
    try {
      const session = await abortCharacterStudioChatOperation(
        sessionId,
        operationId,
      )
      const controller = abortController
      streamRunId += 1
      revokeOptimisticSessionAssets(options.activeChatSession.value)
      rollbackSession = null
      abortController = null
      activeChatOperationId.value = null
      controller?.abort()
      isChatStreaming.value = false
      options.applySession(session)
    } catch (error) {
      throw createActionError(error, '中止聊天生成失败')
    } finally {
      void options.flushPendingRehydrate()
    }
  }

  async function sendChatMessage(content: string, attachments: File[] = []): Promise<void> {
    const document = options.currentDocument.value
    const activeSession = options.activeChatSession.value
    if (!options.bookId.value || !document || !activeSession) return
    if (!content.trim() && attachments.length === 0) return
    if (abortController) abortActiveChatStream()

    const controller = new AbortController()
    abortController = controller
    const runId = ++streamRunId
    isChatStreaming.value = true
    clearErrorMessage()
    options.activeWorkspaceTab.value = 'chat'
    const requestedBookId = options.bookId.value
    const requestedDocId = document.id
    const previousSession = deepClone(activeSession)
    const requestedSessionId = previousSession.session_id
    rollbackSession = previousSession
    const optimisticSession = deepClone(activeSession)
    optimisticSession.messages.push(
      createOptimisticMessage('user', content, attachments.map(createOptimisticAttachment)),
      createOptimisticMessage('assistant', ''),
    )
    options.activeChatSession.value = optimisticSession

    try {
      await streamCharacterStudioChatMessage(requestedBookId, requestedDocId, {
        sessionId: requestedSessionId,
        content,
        attachments,
        signal: controller.signal,
        onAccepted: operationId => {
          if (isActiveStream(
            runId,
            controller,
            requestedBookId,
            requestedDocId,
            requestedSessionId,
          )) {
            activeChatOperationId.value = operationId
          }
        },
        onEvent: event => {
          if (!isActiveStream(runId, controller, requestedBookId, requestedDocId, requestedSessionId)) return
          const session = options.activeChatSession.value
          if (event.type === 'assistant_delta' && session) {
            applyAssistantStreamContent(session, event.content)
          } else if (event.type === 'runtime' && session) {
            applyAssistantRuntimeState(session, event.runtime_log, event.variables)
          } else if (event.type === 'state') {
            revokeOptimisticSessionAssets(session)
            rollbackSession = null
            options.applySession(event.session as CharacterStudioChatSession)
          } else if (event.type === 'error') {
            options.errorMessage.value = event.message
          }
        },
      })
    } catch (error) {
      if (controller.signal.aborted) return
      revokeOptimisticSessionAssets(options.activeChatSession.value)
      options.activeChatSession.value = previousSession
      throw createActionError(error, '发送聊天消息失败')
    } finally {
      if (abortController === controller) {
        abortController = null
        activeChatOperationId.value = null
        rollbackSession = null
      }
      if (runId === streamRunId) isChatStreaming.value = false
      void options.flushPendingRehydrate()
    }
  }

  async function regenerateChatMessage(messageId: string): Promise<void> {
    const document = options.currentDocument.value
    const activeSession = options.activeChatSession.value
    if (!options.bookId.value || !document || !activeSession) return
    if (abortController) abortActiveChatStream()

    const controller = new AbortController()
    abortController = controller
    const runId = ++streamRunId
    isChatStreaming.value = true
    clearErrorMessage()
    const requestedBookId = options.bookId.value
    const requestedDocId = document.id
    const previousSession = deepClone(activeSession)
    const requestedSessionId = previousSession.session_id
    rollbackSession = previousSession
    const userIndex = findRegenerationUserMessageIndex(previousSession.messages, messageId)
    if (userIndex >= 0) {
      const optimisticSession = deepClone(previousSession)
      optimisticSession.messages = optimisticSession.messages.slice(0, userIndex + 1)
      optimisticSession.messages.push(createOptimisticMessage('assistant', ''))
      options.activeChatSession.value = optimisticSession
    }

    try {
      await regenerateCharacterStudioChatMessage(
        requestedBookId,
        requestedDocId,
        requestedSessionId,
        messageId,
        event => {
          if (!isActiveStream(runId, controller, requestedBookId, requestedDocId, requestedSessionId)) return
          const session = options.activeChatSession.value
          if (event.type === 'assistant_delta' && session) {
            applyAssistantStreamContent(session, event.content)
          } else if (event.type === 'runtime' && session) {
            applyAssistantRuntimeState(session, event.runtime_log, event.variables)
          } else if (event.type === 'state') {
            revokeOptimisticSessionAssets(session)
            rollbackSession = null
            options.applySession(event.session as CharacterStudioChatSession)
          } else if (event.type === 'error') {
            options.errorMessage.value = event.message
          }
        },
        controller.signal,
        operationId => {
          if (isActiveStream(
            runId,
            controller,
            requestedBookId,
            requestedDocId,
            requestedSessionId,
          )) {
            activeChatOperationId.value = operationId
          }
        },
      )
    } catch (error) {
      if (controller.signal.aborted) return
      revokeOptimisticSessionAssets(options.activeChatSession.value)
      options.activeChatSession.value = previousSession
      throw createActionError(error, '消息重生失败')
    } finally {
      if (abortController === controller) {
        abortController = null
        activeChatOperationId.value = null
        rollbackSession = null
      }
      if (runId === streamRunId) isChatStreaming.value = false
      void options.flushPendingRehydrate()
    }
  }

  return {
    activeChatOperationId,
    isChatStreaming,
    abortActiveChatOperation,
    abortActiveChatStream,
    sendChatMessage,
    regenerateChatMessage,
  }
}
