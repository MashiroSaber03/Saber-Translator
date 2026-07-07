export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Fall back to the temporary textarea path below.
  }

  return copyTextWithTextarea(text)
}

function copyTextWithTextarea(text: string): boolean {
  if (typeof document === 'undefined' || !document.body) return false

  const textArea = document.createElement('textarea')
  textArea.value = text
  textArea.setAttribute('readonly', '')
  textArea.style.position = 'fixed'
  textArea.style.top = '-9999px'
  textArea.style.opacity = '0'

  try {
    document.body.appendChild(textArea)
    textArea.focus()
    textArea.select()
    return document.execCommand?.('copy') ?? false
  } catch {
    return false
  } finally {
    textArea.remove()
  }
}
