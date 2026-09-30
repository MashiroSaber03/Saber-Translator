import { ref, type Ref } from 'vue'

import {
  abortCharacterStudioChatOperation,
  editCharacterStudioChatMessage,
  regenerateCharacterStudioChatMessage,
  streamCharacterStudioChatMessage,
  type CharacterStudioChatStreamEvent,
} from '@/api/characterStudio'
import {
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
  reloadChatState: (documentId: string) => Promise<void>
}

export function useCharacterStudioChat(options: CharacterStudioChatOptions) {
  const isChatStreaming = ref(false)
  const activeChatOperationId = ref<string | null>(null)
  const acceptedChatSubmissionCount = ref(0)
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
    attachments: CharacterStudioChatAttachment[] = []
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
    requestedSessionId: string
  ): boolean {
    return (
      runId === streamRunId &&
      abortController === controller &&
      options.bookId.value === requestedBookId &&
      options.currentDocument.value?.id === requestedDocId &&
      options.activeChatSession.value?.session_id === requestedSessionId
    )
  }

  function abortActiveChatStream(): void {
    if (!abortController) return
    streamRunId += 1
    revokeOptimisticSessionAssets(options.activeChatSession.value)
    abortController.abort()
    abortController = null
    activeChatOperationId.value = null
    isChatStreaming.value = false
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
      const session = await abortCharacterStudioChatOperation(sessionId, operationId)
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

  type ChatSubmission =
    | { kind: 'send'; content: string; attachments: File[] }
    | { kind: 'regenerate'; messageId: string }
    | { kind: 'edit'; messageId: string; content: string }

  async function runChatGeneration(submission: ChatSubmission): Promise<void> {
    const document = options.currentDocument.value
    const activeSession = options.activeChatSession.value
    if (!options.bookId.value || !document || !activeSession) return
    if (isChatStreaming.value || abortController) return
    if (submission.kind === 'send' && !submission.content.trim() && !submission.attachments.length)
      return
    if (submission.kind === 'edit' && !submission.content.trim()) return

    const previousSession = deepClone(activeSession)
    const optimisticSession = deepClone(activeSession)
    if (submission.kind === 'send') {
      optimisticSession.messages.push(
        createOptimisticMessage('user', submission.content, submission.attachments.map(createOptimisticAttachment))
      )
    } else {
      const userIndex = findRegenerationUserMessageIndex(previousSession.messages, submission.messageId)
      if (userIndex < 0) return
      optimisticSession.messages = optimisticSession.messages.slice(0, userIndex + 1)
      if (submission.kind === 'edit') optimisticSession.messages[userIndex]!.content = submission.content
    }
    optimisticSession.messages.push(createOptimisticMessage('assistant', ''))

    const controller = new AbortController()
    abortController = controller
    const runId = ++streamRunId
    const requestedBookId = options.bookId.value
    const requestedDocId = document.id
    const requestedSessionId = previousSession.session_id
    const isActive = () =>
      isActiveStream(runId, controller, requestedBookId, requestedDocId, requestedSessionId)
    let operationAccepted = false
    isChatStreaming.value = true
    clearErrorMessage()
    options.activeWorkspaceTab.value = 'chat'
    rollbackSession = previousSession
    options.activeChatSession.value = optimisticSession

    const onAccepted = (operationId: string) => {
      operationAccepted = true
      if (!isActive()) return
      activeChatOperationId.value = operationId
      if (submission.kind === 'send') acceptedChatSubmissionCount.value += 1
    }
    const onEvent = (event: CharacterStudioChatStreamEvent) => {
      if (!isActive()) return
      const session = options.activeChatSession.value
      if (event.type === 'assistant_delta' && session) {
        applyAssistantStreamContent(session, event.content)
      } else if (event.type === 'state') {
        revokeOptimisticSessionAssets(session)
        rollbackSession = null
        options.applySession(event.session)
      }
    }
    try {
      if (submission.kind === 'send') {
        await streamCharacterStudioChatMessage({
          sessionId: requestedSessionId,
          baseSessionRevision: previousSession.revision,
          content: submission.content,
          attachments: submission.attachments,
          signal: controller.signal,
          onAccepted,
          onEvent,
        })
      } else if (submission.kind === 'edit') {
        await editCharacterStudioChatMessage(
          requestedSessionId, previousSession.revision, submission.messageId,
          submission.content, onEvent, controller.signal, onAccepted
        )
      } else {
        await regenerateCharacterStudioChatMessage(
          requestedSessionId, previousSession.revision, submission.messageId,
          onEvent, controller.signal, onAccepted
        )
      }
    } catch (error) {
      if (controller.signal.aborted || !isActive()) return
      revokeOptimisticSessionAssets(options.activeChatSession.value)
      if (operationAccepted) {
        try {
          await options.reloadChatState(requestedDocId)
        } catch {
          if (isActive()) options.activeChatSession.value = previousSession
        }
      } else {
        options.activeChatSession.value = previousSession
      }
      if (isActive()) throw createActionError(error, '聊天生成失败')
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

  async function sendChatMessage(content: string, attachments: File[] = []): Promise<void> {
    await runChatGeneration({ kind: 'send', content, attachments })
  }

  async function regenerateChatMessage(messageId: string): Promise<void> {
    await runChatGeneration({ kind: 'regenerate', messageId })
  }

  async function editChatMessage(messageId: string, content: string): Promise<void> {
    await runChatGeneration({ kind: 'edit', messageId, content })
  }

  return {
    activeChatOperationId,
    acceptedChatSubmissionCount,
    isChatStreaming,
    abortActiveChatOperation,
    abortActiveChatStream,
    sendChatMessage,
    regenerateChatMessage,
    editChatMessage,
  }
}
