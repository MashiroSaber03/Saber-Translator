import type { CharacterStudioChatMessage, CharacterStudioChatSession } from '@/types/characterStudio'

export function hasUnsummarizedChatMessages(session: CharacterStudioChatSession | null): boolean {
  const lastMessage = session?.messages.at(-1)
  return !!lastMessage && lastMessage.message_id !== session?.summary_through_message_id
}

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

export function findRegenerationUserMessageIndex(
  messages: CharacterStudioChatMessage[],
  messageId: string,
): number {
  const anchorIndex = messages.findIndex(item => item.message_id === messageId)
  if (anchorIndex < 0) return -1
  if (messages[anchorIndex]?.role === 'user') return anchorIndex
  if (messages[anchorIndex]?.role !== 'assistant') return -1

  for (let index = anchorIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') {
      return index
    }
  }
  return -1
}
