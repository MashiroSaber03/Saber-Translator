import type {
  CharacterStudioDocument,
  CharacterStudioGreetingOption,
} from '@/types/characterStudio'

export function buildCharacterStudioGreetingOptions(
  document: CharacterStudioDocument | null | undefined,
): CharacterStudioGreetingOption[] {
  if (!document) return []

  const greetings: CharacterStudioGreetingOption[] = []
  const firstMessage = String(document.coreMessages.first_message || '').trim()
  if (firstMessage) {
    greetings.push({
      greeting_id: 'first_message',
      label: '主问候',
      content: firstMessage,
      source: { type: 'first_message', index: 0 },
    })
  }

  for (const [index, item] of (document.coreMessages.alternate_greetings || []).entries()) {
    const content = String(item || '').trim()
    if (!content) continue
    greetings.push({
      greeting_id: `alternate_${index + 1}`,
      label: `备用问候 ${index + 1}`,
      content,
      source: { type: 'alternate_greetings', index },
    })
  }

  return greetings
}
