const FAILURE_PREFIXES = ['生成失败:', '提示词生成失败:']

export function isUsableImagePrompt(prompt: string | null | undefined): boolean {
  const text = (prompt || '').trim()
  if (!text) {
    return false
  }

  return !FAILURE_PREFIXES.some(prefix => text.startsWith(prefix))
}

export function normalizeImagePrompt(prompt: string | null | undefined, maxLines = 12): string {
  const lines = (prompt || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)

  if (lines.length === 0) {
    return ''
  }

  return lines.slice(0, maxLines).join('\n')
}

export function hasUsableStoryContent(page: {
  continuity_text?: string
  story_text?: string
  dialogue_text?: string
}): boolean {
  const story = (page.story_text || '').trim()
  const continuity = (page.continuity_text || '').trim()
  const dialogue = (page.dialogue_text || '').trim()
  return Boolean(story) && (Boolean(continuity) || Boolean(dialogue))
}
