import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createHash } from 'node:crypto'
import type { BackgroundRequest, BackgroundResponse, ExtensionSettings } from './types'
import { DEFAULT_PREFERENCE } from './storage'

const id = 'test-extension'
let local: Record<string, unknown>
let session: Record<string, unknown>
let listener: (request: unknown, sender: chrome.runtime.MessageSender, reply: (value: unknown) => void) => boolean

function area(values: Record<string, unknown>) {
  return {
    setAccessLevel: vi.fn().mockResolvedValue(undefined),
    get: vi.fn(async (key: string) => structuredClone({ [key]: values[key] })),
    set: vi.fn(async (patch: Record<string, unknown>) => { Object.assign(values, structuredClone(patch)) }),
    remove: vi.fn(async (key: string) => { delete values[key] }),
  }
}

beforeEach(async () => {
  vi.resetModules()
  local = {}
  session = {}
  const event = () => ({ addListener: vi.fn() })
  vi.stubGlobal('chrome', {
    storage: { local: area(local), session: area(session) },
    runtime: {
      id, onInstalled: event(),
      onMessage: { addListener: (callback: typeof listener) => { listener = callback } },
    },
    contextMenus: { onClicked: event(), update: vi.fn().mockResolvedValue(undefined) },
    tabs: { query: vi.fn().mockResolvedValue([]), onActivated: event(), onUpdated: event(), onRemoved: event() },
  })
  await import('./background')
})

afterEach(() => vi.unstubAllGlobals())

function request<T>(message: BackgroundRequest, sender: chrome.runtime.MessageSender = {
  id, url: `chrome-extension://${id}/popup.html`,
}): Promise<BackgroundResponse<T>> {
  return new Promise(resolve => listener(message, sender, value => resolve(value as BackgroundResponse<T>)))
}

describe('extension background boundary', () => {
  it('keeps a new active session when clearing the previous session concurrently', async () => {
    const pageUrl = 'https://comic.example/chapter'
    const sender = { id, url: pageUrl, tab: { id: 4, url: pageUrl } } as chrome.runtime.MessageSender
    const discovery = { stopped: true, usingAdapter: false, rule: null }
    await request({ type: 'set-active-session', pageUrl, sessionId: 'old', discovery }, sender)
    const key = 'saber-active-browser-session-v1:4'
    const previous = structuredClone(session[key])
    let release!: () => void
    vi.mocked(chrome.storage.session.get).mockImplementationOnce(async () => {
      await new Promise<void>(resolve => { release = resolve })
      return { [key]: previous }
    })
    const clearing = request({ type: 'clear-active-session', pageUrl, sessionId: 'old' }, sender)
    await vi.waitFor(() => expect(release).toBeDefined())
    const setting = request({ type: 'set-active-session', pageUrl, sessionId: 'new', discovery }, sender)
    await Promise.resolve()
    release()
    await Promise.all([clearing, setting])
    expect(session[key]).toMatchObject({ sessionId: 'new' })
  })

  it('hashes source identities in the background using the existing SHA-256 format', async () => {
    const value = 'image:http://comic.example/chapter/page.png'
    expect(await request({ type: 'hash-source', value })).toEqual({
      ok: true, data: createHash('sha256').update(value).digest('hex'),
    })
  })

  it('preserves concurrent domain and connection updates', async () => {
    const responses = await Promise.all([
      request({ type: 'set-preference', hostname: 'a.example', preference: { ...DEFAULT_PREFERENCE, disabled: true } }),
      request({ type: 'set-preference', hostname: 'b.example', preference: { ...DEFAULT_PREFERENCE, mode: 'hq' } }),
      request({ type: 'save-connection', token: 'test-token-with-at-least-32-characters', serverPort: 5193 }),
    ])
    expect(responses.every(response => response.ok)).toBe(true)
    const settings = local['saber-extension-settings-v1'] as ExtensionSettings
    expect(settings.domains['a.example']?.disabled).toBe(true)
    expect(settings.domains['b.example']?.mode).toBe('hq')
    expect(settings.serverPort).toBe(5193)
    expect(settings.token).toBe('test-token-with-at-least-32-characters')
  })

  it('restores saved discovery scope and never expands an older session without scope', async () => {
    const pageUrl = 'https://comic.example/chapter'
    const sender = { id, url: pageUrl, tab: { id: 4, url: pageUrl } } as chrome.runtime.MessageSender
    const discovery = { stopped: true, usingAdapter: false, rule: null }
    await request({ type: 'set-active-session', pageUrl, sessionId: 'one-image', discovery }, sender)
    expect(await request({ type: 'get-active-session', pageUrl }, sender)).toEqual({
      ok: true, data: { sessionId: 'one-image', discovery },
    })
    session['saber-active-browser-session-v1:4'] = { pageUrl, sessionId: 'older-session' }
    expect(await request({ type: 'get-active-session', pageUrl }, sender)).toEqual({
      ok: true, data: { sessionId: 'older-session', discovery },
    })
  })

  it('does not expose pairing settings to a content script', async () => {
    const response = await request({ type: 'get-popup-state' }, { id, url: 'https://comic.example' })
    expect(response).toMatchObject({ ok: false, error: { code: 'extension_page_required' } })
  })
})
