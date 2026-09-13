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
    action: { onClicked: event(), setBadgeText: vi.fn(), setTitle: vi.fn() },
    tabs: { sendMessage: vi.fn().mockResolvedValue({ opened: true }), query: vi.fn().mockResolvedValue([{ id: 4 }]), onActivated: event(), onUpdated: event(), onRemoved: event() },
  })
  await import('./background')
})

afterEach(() => vi.unstubAllGlobals())

function request<T>(message: BackgroundRequest, sender: chrome.runtime.MessageSender = {
  id, url: `chrome-extension://${id}/panel.html`,
}): Promise<BackgroundResponse<T>> {
  return new Promise(resolve => listener(message, sender, value => resolve(value as BackgroundResponse<T>)))
}

describe('extension background boundary', () => {
  it('accepts the current SPA URL even when sender.url retains the title page', async () => {
    const pageUrl = 'https://mangadex.org/chapter/chapter-id'
    const response = await request({ type: 'page-opened', pageUrl }, {
      id, url: 'https://mangadex.org/title/title-id', documentId: 'same-document',
      tab: { id: 4, url: pageUrl },
    } as chrome.runtime.MessageSender)
    expect(response).toEqual({ ok: true, data: { opened: true } })
    expect(session['saber-active-browser-session-v1:4']).toMatchObject({ pageUrl, documentId: 'same-document' })
  })

  it('still rejects an old page request after the tab navigates elsewhere', async () => {
    const oldUrl = 'https://comic.example/old'
    const response = await request({ type: 'page-opened', pageUrl: oldUrl }, {
      id, url: oldUrl, tab: { id: 4, url: 'https://comic.example/new' },
    } as chrome.runtime.MessageSender)
    expect(response).toMatchObject({ ok: false, error: { code: 'stale_page_context' } })
    expect(session).toEqual({})
  })
  it('opens the existing floating panel from the toolbar without another page', async () => {
    const click = vi.mocked(chrome.action.onClicked.addListener).mock.calls[0]![0]
    click({ id: 4, url: 'https://comic.example/chapter' } as chrome.tabs.Tab)
    await vi.waitFor(() => expect(chrome.action.setBadgeText).toHaveBeenCalledWith({ tabId: 4, text: '' }))
    expect(chrome.tabs.sendMessage).toHaveBeenCalledWith(4, { type: 'open-panel' }, { frameId: 0 })
  })

  it('explains restricted pages through the toolbar instead of opening a fallback page', async () => {
    const click = vi.mocked(chrome.action.onClicked.addListener).mock.calls[0]![0]
    click({ id: 4, url: 'chrome://extensions' } as chrome.tabs.Tab)
    await vi.waitFor(() => expect(chrome.action.setTitle).toHaveBeenCalledWith({ tabId: 4, title: expect.stringContaining('普通网页') }))
    expect(chrome.tabs.sendMessage).not.toHaveBeenCalled()
  })

  it('uses the hosting tab rather than the currently active tab for connection settings', async () => {
    const response = await request({ type: 'get-connection-state' }, {
      id, url: `chrome-extension://${id}/panel.html`,
      tab: { id: 7, url: 'https://reader.example/chapter' },
    } as chrome.runtime.MessageSender)
    expect(response).toMatchObject({ ok: true, data: { hostname: 'reader.example' } })
  })
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
    const response = await request({ type: 'get-connection-state' }, { id, url: 'https://comic.example' })
    expect(response).toMatchObject({ ok: false, error: { code: 'extension_page_required' } })
  })

  it('merges concurrent preference fields without undoing a disabled site', async () => {
    await request({ type: 'set-preference', hostname: 'comic.example', preference: { disabled: true, mode: 'hq' } })
    await Promise.all([
      request({ type: 'set-preference', hostname: 'comic.example', preference: { glossaryEnabled: true } }),
      request({ type: 'set-preference', hostname: 'comic.example', preference: { fabPosition: { x: 30, y: 60 } } }),
    ])
    expect((local['saber-extension-settings-v1'] as ExtensionSettings).domains['comic.example'])
      .toMatchObject({ disabled: true, mode: 'hq', glossaryEnabled: true, fabPosition: { x: 30, y: 60 } })
  })

  it('only removes a learned rule when explicitly cleared', async () => {
    const rule = { selector: 'main img', kind: 'image' as const, confirmedAt: 1 }
    await request({ type: 'set-preference', hostname: 'comic.example', preference: { rule } })
    await request({ type: 'set-preference', hostname: 'comic.example', preference: { glossaryEnabled: true } })
    expect((local['saber-extension-settings-v1'] as ExtensionSettings).domains['comic.example']?.rule).toEqual(rule)
    await request({ type: 'set-preference', hostname: 'comic.example', preference: { rule: null } })
    expect((local['saber-extension-settings-v1'] as ExtensionSettings).domains['comic.example']?.rule).toBeUndefined()
  })

  it('notifies other tabs on the same hostname only when enabled state changes', async () => {
    chrome.tabs.query = vi.fn().mockResolvedValue([
      { id: 4, url: 'https://comic.example/a' },
      { id: 5, url: 'https://comic.example/b' },
      { id: 6, url: 'https://other.example/c' },
    ])
    await request({ type: 'set-preference', hostname: 'comic.example', preference: { disabled: true } },
      { id, url: 'https://comic.example/a', tab: { id: 4 } } as chrome.runtime.MessageSender)
    expect(chrome.tabs.sendMessage).toHaveBeenCalledTimes(1)
    expect(chrome.tabs.sendMessage).toHaveBeenCalledWith(5,
      { type: 'site-enabled-changed', hostname: 'comic.example', disabled: true }, { frameId: 0 })
    await request({ type: 'set-preference', hostname: 'comic.example', preference: { glossaryEnabled: true } })
    expect(chrome.tabs.sendMessage).toHaveBeenCalledTimes(1)
  })
})
