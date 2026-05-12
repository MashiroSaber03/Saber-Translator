const FAILURE_PREFIXES = ['生成失败:', '提示词生成失败:']

export function isUsableImagePrompt(prompt: string | null | undefined): boolean {
  const text = (prompt || '').trim()
  if (!text) {
    return false
  }

  return !FAILURE_PREFIXES.some(prefix => text.startsWith(prefix))
}

export function normalizeImagePrompt(prompt: string | null | undefined): string {
  const lines = (prompt || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)

  if (lines.length === 0) {
    return ''
  }

  return lines.slice(0, 5).join('\n')
}
