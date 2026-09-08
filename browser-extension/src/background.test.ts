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
    get: vi.fn(async (key: string | null) => structuredClone(key === null ? values : { [key]: values[key] })),
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
    alarms: { create: vi.fn(), onAlarm: event() },
    contextMenus: { onClicked: event(), update: vi.fn().mockResolvedValue(undefined) },
    tabs: { sendMessage: vi.fn().mockResolvedValue({ ok: true, data: { opened: true } }), query: vi.fn().mockResolvedValue([{ id: 4 }]), onActivated: event(), onUpdated: event(), onRemoved: event() },
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
  it('discards a session whose page closes before creation finishes', async () => {
    local['saber-extension-settings-v1'] = { token: 'test-token-with-at-least-32-characters', serverPort: 5000, domains: {} }
    const pageUrl = 'https://comic.example/chapter'
    const sender = { id, url: pageUrl, documentId: 'old', tab: { id: 4, url: pageUrl } } as chrome.runtime.MessageSender
    await request({ type: 'page-opened', pageUrl }, sender)
    let resolve!: (response: Response) => void
    const fetch = vi.fn().mockImplementationOnce(() => new Promise<Response>(done => { resolve = done }))
      .mockResolvedValue(new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetch)
    const pending = request({ type: 'create-session', payload: { pageUrl } }, sender)
    await vi.waitFor(() => expect(resolve).toBeDefined())
    await request({ type: 'page-closed', pageUrl }, sender)
    await request({ type: 'page-opened', pageUrl }, { ...sender, documentId: 'new' })
    resolve(Response.json({ id: 'late-session' }))
    expect(await pending).toMatchObject({ ok: false, error: { code: 'page_closed' } })
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining('/late-session/discard'), expect.anything())
    expect(session['saber-active-browser-session-v1:4']).toMatchObject({ documentId: 'new' })
  })

  it('does not let an old alarm observation close a newly opened document', async () => {
    const pageUrl = 'https://comic.example/chapter'
    session['saber-active-browser-session-v1:4'] = { pageUrl, documentId: 'old' }
    let resolveTab!: (tab: chrome.tabs.Tab) => void
    chrome.tabs.get = vi.fn(() => new Promise<chrome.tabs.Tab>(resolve => { resolveTab = resolve })) as typeof chrome.tabs.get
    const alarm = vi.mocked(chrome.alarms.onAlarm.addListener).mock.calls[0]![0]
    alarm({ name: 'saber-live-pages', scheduledTime: Date.now() })
    await vi.waitFor(() => expect(resolveTab).toBeDefined())
    await request({ type: 'page-opened', pageUrl }, {
      id, url: pageUrl, documentId: 'new', tab: { id: 4, url: pageUrl },
    } as chrome.runtime.MessageSender)
    resolveTab({ id: 4, url: 'https://comic.example/elsewhere' } as chrome.tabs.Tab)
    await vi.waitFor(() => expect(chrome.storage.session.get).toHaveBeenCalledTimes(3))
    expect(session['saber-active-browser-session-v1:4']).toMatchObject({ documentId: 'new' })
    expect(chrome.storage.session.remove).not.toHaveBeenCalled()
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

  it('does not expose pairing settings to a content script', async () => {
    const response = await request({ type: 'get-popup-state' }, { id, url: 'https://comic.example' })
    expect(response).toMatchObject({ ok: false, error: { code: 'extension_page_required' } })
  })
})
