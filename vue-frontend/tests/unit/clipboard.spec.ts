import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { copyTextToClipboard } from '@/utils/clipboard'

function setClipboard(writeText: (text: string) => Promise<void>): void {
  Object.defineProperty(globalThis.navigator, 'clipboard', {
    value: { writeText },
    configurable: true,
  })
}

function setExecCommand(execCommand: (command: string) => boolean): void {
  Object.defineProperty(document, 'execCommand', {
    value: execCommand,
    configurable: true,
  })
}

describe('clipboard utility', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    document.body.innerHTML = ''
  })

  it('uses the async Clipboard API when available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    const execCommand = vi.fn().mockReturnValue(true)
    setClipboard(writeText)
    setExecCommand(execCommand)

    await expect(copyTextToClipboard('hello')).resolves.toBe(true)

    expect(writeText).toHaveBeenCalledWith('hello')
    expect(execCommand).not.toHaveBeenCalled()
    expect(document.querySelector('textarea')).toBeNull()
  })

  it('falls back to a temporary textarea when Clipboard API fails', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('denied'))
    const execCommand = vi.fn().mockReturnValue(true)
    setClipboard(writeText)
    setExecCommand(execCommand)

    await expect(copyTextToClipboard('fallback')).resolves.toBe(true)

    expect(execCommand).toHaveBeenCalledWith('copy')
    expect(document.querySelector('textarea')).toBeNull()
  })

  it('returns false when neither copy path succeeds', async () => {
    setClipboard(vi.fn().mockRejectedValue(new Error('denied')))
    setExecCommand(vi.fn().mockImplementation(() => {
      throw new Error('blocked')
    }))

    await expect(copyTextToClipboard('blocked')).resolves.toBe(false)
    expect(document.querySelector('textarea')).toBeNull()
  })

  it('keeps production owners on the shared clipboard boundary', () => {
    for (const file of [
      'src/views/BookshelfView.vue',
      'src/components/edit/useBubbleEditor.ts',
      'src/components/insight/settings/PromptsSettingsTab.vue',
      'src/components/insight/studio/CharacterStudioPreview.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain("from '@/utils/clipboard'")
      expect(source, file).not.toContain('navigator.clipboard')
      expect(source, file).not.toContain('document.execCommand')
      expect(source, file).not.toContain("document.createElement('textarea')")
    }
  })
})
