import type { CharacterStudioChatMessage, CharacterStudioChatSession } from '@/types/characterStudio'
import { deepClone } from '@/utils/deepClone'

function getLastAssistantMessage(session: CharacterStudioChatSession): CharacterStudioChatMessage | null {
  const message = session.messages.at(-1)
  return message?.role === 'assistant' ? message : null
}

export function applyAssistantStreamContent(
  session: CharacterStudioChatSession,
  content: string,
): boolean {
  const message = getLastAssistantMessage(session)
  if (!message) return false
  message.content = content
  return true
}

export function applyAssistantRuntimeState(
  session: CharacterStudioChatSession,
  runtimeLog: Array<Record<string, unknown>>,
  variables: Record<string, unknown>,
): boolean {
  const message = getLastAssistantMessage(session)
  if (!message) return false
  message.runtime_log = deepClone(runtimeLog)
  message.variables_snapshot = deepClone(variables)
  return true
}

export function findRegenerationUserMessageIndex(
  messages: CharacterStudioChatMessage[],
  messageId: string,
): number {
  const anchorIndex = messages.findIndex(item => item.message_id === messageId)
  if (anchorIndex < 0) return -1
  if (messages[anchorIndex]?.role !== 'assistant') return anchorIndex

  for (let index = anchorIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') {
      return index
    }
  }
  return anchorIndex
}
