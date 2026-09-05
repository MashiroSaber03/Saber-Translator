// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ImageCandidate } from './discovery'
import { DEFAULT_PREFERENCE } from './storage'
import type { BrowserSessionDto } from './types'
import { ExtensionUi, type UiCallbacks } from './ui'
import { UI_STYLES } from './uiStyles'

function callbacks(): UiCallbacks {
  return {
    onActivity: vi.fn(),
    onDiscover: vi.fn(),
    onConfirm: vi.fn(),
    onPreferenceChange: vi.fn(),
    onPanelOpenChange: vi.fn(),
    onPanelPositionChange: vi.fn(),
    onFabPositionChange: vi.fn(),
    onToggleGlobal: vi.fn().mockResolvedValue(true),
    onTogglePage: vi.fn().mockResolvedValue(true),
    onRetryPage: vi.fn(),
    onRetryUploads: vi.fn(),
    onRetryStart: vi.fn(),
    onStopDiscovery: vi.fn(),
    onCancel: vi.fn(),
    onLoadLibraryBooks: vi.fn().mockResolvedValue([]),
    onImport: vi.fn().mockResolvedValue({
      destination: 'new',
      bookId: 'book',
      bookTitle: 'Example chapter',
      chapterId: 'chapter',
      chapterTitle: 'Example chapter',
      importedPages: 1,
      omittedPages: 0,
      termsAdded: 0,
    }),
    onDisableSite: vi.fn(),
    onDeleteAdaptation: vi.fn(),
    onCopyDiagnostics: vi.fn(),
  }
}

const mounted: ExtensionUi[] = []

function mountUi(handlers: UiCallbacks): ExtensionUi {
  const ui = new ExtensionUi(handlers, DEFAULT_PREFERENCE, 'Example chapter', true)
  mounted.push(ui)
  return ui
}

let trustedEvents = true
beforeEach(() => {
  document.documentElement.replaceChildren(document.createElement('body'))
  trustedEvents = true
  const addListener = ShadowRoot.prototype.addEventListener
  // jsdom cannot create trusted input. Native event behavior is covered in Chromium.
  vi.spyOn(ShadowRoot.prototype, 'addEventListener').mockImplementation(function (
    this: ShadowRoot, type, listener, options,
  ) {
    addListener.call(this, type, event => {
      const input = trustedEvents ? { isTrusted: true } as Event : event
      if (typeof listener === 'function') listener.call(this, input)
      else listener?.handleEvent(input)
    }, options)
  })
})
afterEach(() => {
  for (const ui of mounted.splice(0)) ui.remove()
  vi.restoreAllMocks()
})

describe('isolated extension UI', () => {
  it('hides local data from the host page and rejects synthetic actions', () => {
    trustedEvents = false
    const handlers = callbacks()
    const ui = mountUi(handlers)
    expect(ui.host.shadowRoot).toBeNull()
    const buttons = [...ui.shadow.querySelectorAll<HTMLButtonElement>('button')]
    buttons.find(button => button.textContent === '识别漫画图片')!.click()
    ui.shadow.querySelector('select')!.dispatchEvent(new Event('change', { bubbles: true }))
    expect(handlers.onDiscover).not.toHaveBeenCalled()
    expect(handlers.onPreferenceChange).not.toHaveBeenCalled()
    expect(handlers.onActivity).not.toHaveBeenCalled()
  })
  it('uses a Shadow DOM and retains settings while the panel is closed and reopened', () => {
    const handlers = callbacks()
    const ui = mountUi(handlers)
    const panel = ui.shadow.querySelector<HTMLElement>('.saber-panel')
    const fab = ui.shadow.querySelector<HTMLButtonElement>('.saber-fab')
    const mode = ui.shadow.querySelectorAll<HTMLSelectElement>('.saber-select')[1]
    expect(panel).not.toBeNull()
    expect(fab).not.toBeNull()
    expect(mode).not.toBeUndefined()

    fab!.click()
    expect(panel!.dataset.open).toBe('true')
    expect(handlers.onPanelOpenChange).toHaveBeenCalledWith(true)
    mode!.value = 'hq'
    mode!.dispatchEvent(new Event('change'))
    ui.setOpen(false)
    ui.setOpen(true)

    expect(mode!.value).toBe('hq')
    expect(handlers.onPreferenceChange).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'hq' }),
    )
    expect(document.querySelector('.saber-panel')).toBeNull()
  })

  it('contains pink theme tokens and a dark color-scheme override', () => {
    expect(UI_STYLES).toContain('#f43f8c')
    expect(UI_STYLES).toContain('@media (prefers-color-scheme: dark)')
    expect(UI_STYLES).toContain('appearance: none')
    expect(UI_STYLES).toContain('.saber-candidates { display: flex; flex-wrap: wrap;')
    expect(UI_STYLES).not.toContain('.saber-image-actions')
  })

  it('anchors the shadow host to the viewport without blocking the page', () => {
    expect(UI_STYLES).toMatch(
      /:host \{[\s\S]*?position: fixed;[\s\S]*?inset: 0;[\s\S]*?z-index: 2147483646;[\s\S]*?pointer-events: none;/,
    )
    expect(UI_STYLES).toContain('.saber-root {\n  position: absolute; inset: 0; pointer-events: none;')
    expect(UI_STYLES).toMatch(/\.saber-(?:fab|panel) \{[\s\S]*?pointer-events: auto;/)
  })

  it('keeps growing progress content in the panel scroll flow', () => {
    const ui = mountUi(callbacks())
    const progressSection = ui.shadow.querySelector<HTMLElement>('.saber-progress-section')
    const progressActions = ui.shadow.querySelector<HTMLElement>('.saber-progress-actions')

    expect(progressSection).not.toBeNull()
    expect(progressActions?.closest('.saber-progress-section')).toBe(progressSection)
    expect(UI_STYLES).toContain('.saber-progress-section { flex: 0 0 auto; }')
    expect(UI_STYLES).toContain('.saber-progress-actions { margin-top: 10px; }')
  })

  it('previews blob-backed candidates and falls back only after a load error', () => {
    const ui = mountUi(callbacks())
    const source = document.createElement('img')
    const candidate: ImageCandidate = {
      id: 'blob-page',
      kind: 'image',
      element: source,
      bindings: [source],
      sourceUrl: 'blob:https://mangadex.org/page-1',
      sourceIdentity: 'image:blob:https://mangadex.org/page-1',
      width: 1_444,
      height: 2_048,
    }

    ui.showCandidates([candidate])

    const preview = ui.shadow.querySelector<HTMLImageElement>('.saber-candidate img')
    expect(preview?.src).toBe(candidate.sourceUrl)
    expect(ui.shadow.querySelector('.saber-candidate__fallback')).toBeNull()

    preview?.dispatchEvent(new Event('error'))
    expect(ui.shadow.querySelector('.saber-candidate img')).toBeNull()
    expect(ui.shadow.querySelector('.saber-candidate__fallback')?.textContent)
      .toBe('页面图片')
  })

  it('shows count-based image preparation progress and clears it for translation', () => {
    const ui = mountUi(callbacks())
    const section = ui.shadow.querySelector<HTMLElement>('.saber-preparation')!
    const count = ui.shadow.querySelector<HTMLElement>('.saber-preparation__count')!
    const meter = ui.shadow.querySelector<HTMLProgressElement>('.saber-preparation__meter')!
    const detail = ui.shadow.querySelector<HTMLElement>('.saber-preparation__detail')!

    expect(section.hidden).toBe(true)
    ui.showPreparationProgress(3, 5, 1)

    expect(section.hidden).toBe(false)
    expect(count.textContent).toBe('3 / 5')
    expect(meter.max).toBe(5)
    expect(meter.value).toBe(3)
    expect(detail.textContent).toBe('成功 2 · 失败 1')
    expect(ui.shadow.querySelector('.saber-banner__text')?.textContent)
      .toContain('正在准备漫画图片')

    ui.hidePreparationProgress()
    expect(section.hidden).toBe(true)
  })

  it('keeps per-page original and retranslation actions in one collapsed list', async () => {
    const handlers = callbacks()
    const ui = mountUi(handlers)
    const session: BrowserSessionDto = {
      id: 'session',
      pageUrl: 'https://example.test/chapter',
      pageTitle: 'Chapter',
      bookId: 'book',
      chapterId: 'chapter',
      mode: 'standard',
      glossaryEnabled: false,
      autoTermsEnabled: false,
      state: 'partial',
      pendingStart: false,
      expiresAt: null,
      counts: { total: 2, queued: 0, translating: 0, completed: 1, failed: 1, cancelled: 0 },
      pages: [
        {
          id: 'completed-page',
          clientPageKey: 'completed-key',
          ordinal: 1,
          pageId: 'page-1',
          state: 'completed',
          resultReady: true,
          retryCount: 0,
          error: null,
        },
        {
          id: 'failed-page',
          clientPageKey: 'failed-key',
          ordinal: 2,
          pageId: 'page-2',
          state: 'failed',
          resultReady: false,
          retryCount: 0,
          error: { code: 'translation_failed', message: 'boom' },
        },
      ],
    }

    ui.showSession(session)
    const details = ui.shadow.querySelector<HTMLDetailsElement>('.saber-page-actions')!
    expect(details.open).toBe(false)
    expect(details.querySelector('summary')?.textContent).toBe('单页操作 · 2')
    const buttons = [...details.querySelectorAll<HTMLButtonElement>('button')]
    buttons.find(button => button.textContent === '查看原图')?.click()
    buttons.find(button => button.textContent === '重翻')?.click()
    buttons.find(button => button.textContent === '重试')?.click()
    await Promise.resolve()

    expect(handlers.onTogglePage).toHaveBeenCalledWith('completed-page')
    expect(handlers.onRetryPage).toHaveBeenCalledWith('completed-page')
    expect(handlers.onRetryPage).toHaveBeenCalledWith('failed-page')
  })

  it('imports a terminal task once even if fields change while awaiting import', async () => {
    const handlers = callbacks()
    let finish!: () => void
    vi.mocked(handlers.onImport).mockImplementation(async () => {
      await new Promise<void>(resolve => { finish = resolve })
      return {
        destination: 'existing', bookId: 'existing-book', bookTitle: 'My series',
        chapterId: 'chapter', chapterTitle: 'Chapter 5', importedPages: 1,
        omittedPages: 0, termsAdded: 0,
      }
    })
    vi.mocked(handlers.onLoadLibraryBooks).mockResolvedValue([
      { id: 'existing-book', title: 'My series', chapterCount: 4 },
    ])
    const ui = mountUi(handlers)
    ui.showSession({
      id: 'session',
      pageUrl: 'https://example.test/chapter',
      pageTitle: 'Example chapter',
      bookId: 'hidden-book',
      chapterId: 'chapter',
      mode: 'standard',
      glossaryEnabled: false,
      autoTermsEnabled: false,
      state: 'completed',
      pendingStart: false,
      expiresAt: null,
      counts: { total: 1, queued: 0, translating: 0, completed: 1, failed: 0, cancelled: 0 },
      pages: [{
        id: 'browser-page',
        clientPageKey: 'key',
        ordinal: 1,
        pageId: 'page',
        state: 'completed',
        resultReady: true,
        retryCount: 0,
        error: null,
      }],
    })

    const importButton = [...ui.shadow.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent === '导入到书架')!
    importButton.click()
    await vi.waitFor(() => {
      expect(ui.shadow.querySelector<HTMLElement>('.saber-import-overlay')?.dataset.open)
        .toBe('true')
    })
    const existing = ui.shadow.querySelector<HTMLInputElement>(
      'input[value="existing"]',
    )!
    existing.click()
    const chapterTitle = [...ui.shadow.querySelectorAll<HTMLInputElement>('.saber-input')]
      .find(input => input.closest('label')?.textContent?.includes('章节名称'))!
    chapterTitle.value = 'Chapter 5'
    chapterTitle.dispatchEvent(new Event('input'))
    const confirm = [...ui.shadow.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent === '确认导入')!
    confirm.click()
    chapterTitle.value = 'Chapter 6'
    chapterTitle.dispatchEvent(new Event('input'))
    confirm.click()
    expect(handlers.onImport).toHaveBeenCalledOnce()
    finish()
    await vi.waitFor(() => {
      expect(handlers.onImport).toHaveBeenCalledWith({
        destination: 'existing',
        targetBookId: 'existing-book',
        chapterTitle: 'Chapter 5',
      })
    })
  })

  it('shows upload retry only for image-import failures', () => {
    const handlers = callbacks()
    const ui = mountUi(handlers)
    const buttons = [...ui.shadow.querySelectorAll<HTMLButtonElement>('button')]
    expect(buttons.filter(button => button.textContent === '识别漫画图片')).toHaveLength(1)
    expect(buttons.some(button => button.textContent === '手选同类')).toBe(false)

    const diagnostics = buttons.find(button => button.textContent === '复制诊断')
    const retryUploads = buttons.find(button => button.textContent === '重试上传')
    expect(diagnostics?.parentElement?.hidden).toBe(true)
    ui.showError({ code: 'saber_unreachable', message: '无法连接 Saber' })
    expect(diagnostics?.parentElement?.hidden).toBe(false)
    expect(retryUploads?.hidden).toBe(true)

    ui.showUploadError(2, { code: 'source_timeout', message: '图片下载超时' })
    expect(retryUploads?.hidden).toBe(false)
    retryUploads?.click()
    expect(handlers.onRetryUploads).toHaveBeenCalledOnce()
    ui.clearUploadError()
    expect(retryUploads?.hidden).toBe(true)
  })

  it('uses the selected discovery method without a duplicate picker action', () => {
    const handlers = callbacks()
    const ui = mountUi(handlers)
    const method = ui.shadow.querySelector<HTMLSelectElement>('.saber-select')!
    const discover = [...ui.shadow.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent === '识别漫画图片')!

    method.value = 'similar'
    method.dispatchEvent(new Event('change'))
    expect(discover.textContent).toBe('选择一张漫画图片')
    discover.click()
    expect(handlers.onDiscover).toHaveBeenCalledWith('similar')
  })

  it('only offers deletion for a saved site adaptation', () => {
    const withoutRule = callbacks()
    const emptyUi = mountUi(withoutRule)
    const emptyButton = [...emptyUi.shadow.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent === '删除已保存适配')!
    expect(emptyButton.hidden).toBe(true)

    const withRule = callbacks()
    const savedUi = new ExtensionUi(withRule, {
      ...DEFAULT_PREFERENCE,
      rule: {
        selector: '.reader-page > img.comic',
        kind: 'image',
        confirmedAt: 1,
      },
    }, 'Example chapter', false)
    mounted.push(savedUi)
    const savedButton = [...savedUi.shadow.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent === '删除已保存适配')!
    expect(savedButton.hidden).toBe(false)

    savedButton.click()
    expect(withRule.onDeleteAdaptation).toHaveBeenCalledOnce()
    savedUi.setAdaptationSaved(false)
    expect(savedButton.hidden).toBe(true)
  })

  it('drags the panel from its header and persists one clamped position', () => {
    const handlers = callbacks()
    const ui = mountUi(handlers)
    const panel = ui.shadow.querySelector<HTMLElement>('.saber-panel')!
    const header = ui.shadow.querySelector<HTMLElement>('.saber-header')!
    vi.spyOn(panel, 'getBoundingClientRect').mockReturnValue({
      x: 100,
      y: 80,
      left: 100,
      top: 80,
      right: 436,
      bottom: 560,
      width: 336,
      height: 480,
      toJSON: () => ({}),
    })
    const pointer = (type: string, x: number, y: number): Event => {
      const event = new Event(type, { bubbles: true, cancelable: true })
      Object.assign(event, { button: 0, pointerId: 1, clientX: x, clientY: y })
      return event
    }

    header.dispatchEvent(pointer('pointerdown', 120, 100))
    header.dispatchEvent(pointer('pointermove', 170, 140))
    header.dispatchEvent(pointer('pointerup', 170, 140))

    expect(panel.style.left).toBe('150px')
    expect(panel.style.top).toBe('120px')
    expect(handlers.onPanelPositionChange).toHaveBeenCalledWith({ x: 150, y: 120 })
  })

  it('drags the floating button without treating the gesture as a click', () => {
    const handlers = callbacks()
    const ui = mountUi(handlers)
    const panel = ui.shadow.querySelector<HTMLElement>('.saber-panel')!
    const fab = ui.shadow.querySelector<HTMLButtonElement>('.saber-fab')!
    vi.spyOn(fab, 'getBoundingClientRect').mockReturnValue({
      x: 900,
      y: 600,
      left: 900,
      top: 600,
      right: 946,
      bottom: 646,
      width: 46,
      height: 46,
      toJSON: () => ({}),
    })
    const pointer = (type: string, x: number, y: number): Event => {
      const event = new Event(type, { bubbles: true, cancelable: true })
      Object.assign(event, { button: 0, pointerId: 2, clientX: x, clientY: y })
      return event
    }

    fab.dispatchEvent(pointer('pointerdown', 920, 620))
    window.dispatchEvent(pointer('pointermove', 720, 520))
    window.dispatchEvent(pointer('pointerup', 720, 520))
    fab.click()

    expect(fab.style.left).toBe('700px')
    expect(fab.style.top).toBe('500px')
    expect(panel.dataset.open).toBe('false')
    expect(handlers.onFabPositionChange).toHaveBeenCalledWith({ x: 700, y: 500 })
  })
})
