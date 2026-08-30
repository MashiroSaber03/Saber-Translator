// @vitest-environment jsdom

import { readFileSync } from 'node:fs'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const popupState = {
  token: 'stored-token-with-more-than-thirty-two-characters',
  serverPort: 5000,
  hostname: 'reader.example',
  preference: {
    disabled: false,
    method: 'adapter',
    mode: 'standard',
    glossaryEnabled: false,
    autoTermsEnabled: false,
    panelOpen: false,
  },
}

describe('extension popup', () => {
  beforeEach(() => {
    vi.resetModules()
    document.body.innerHTML = '<main id="app"></main>'
  })

  it('uses one connection action and submits the current form values', async () => {
    const sendMessage = vi.fn(async (request: { type: string }) => {
      if (request.type === 'get-popup-state') return { ok: true, data: popupState }
      return { ok: true, data: { status: 'ready' } }
    })
    vi.stubGlobal('chrome', {
      runtime: {
        getManifest: () => ({ version: '1.0.0' }),
        sendMessage,
      },
      tabs: {
        query: vi.fn(async () => []),
        reload: vi.fn(async () => undefined),
      },
    })

    await import('./popup')
    await vi.waitFor(() => {
      expect(document.querySelector('.status')?.textContent).toContain('已连接 Saber')
    })

    const form = document.querySelector<HTMLFormElement>('.connection')!
    const token = document.querySelector<HTMLInputElement>('#saber-pairing-token')!
    const port = document.querySelector<HTMLInputElement>('#saber-server-port')!
    expect(form.querySelectorAll('button')).toHaveLength(1)
    expect(document.body.textContent).not.toContain('重新检测')

    token.value = 'fresh-token-with-more-than-thirty-two-characters'
    port.value = '5100'
    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
    await vi.waitFor(() => {
      expect(sendMessage).toHaveBeenCalledWith({
        type: 'save-connection',
        token: token.value,
        serverPort: 5100,
      })
    })
  })

  it('keeps both connection fields aligned with one explicit type scale', () => {
    const css = readFileSync('src/popup.css', 'utf8')
    expect(css).toContain('.row > .field { margin: 0; }')
    expect(css).toContain('align-items: end')
    expect(css).toContain('font-size: 13px')
  })
})
