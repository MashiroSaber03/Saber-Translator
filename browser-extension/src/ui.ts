import type { ImageCandidate } from './discovery'
import type {
  BrowserLibraryBook,
  BrowserSessionDto,
  BrowserSessionImportCommand,
  BrowserSessionImportResult,
  DetectionMethod,
  DomainPreference,
  PanelPosition,
  LearnedRule,
} from './types'
import type { StudioAction, StudioState } from './studio/protocol'
import { HOST_STYLES } from './hostStyles'

export interface UiCallbacks {
  onDiscover(method: DetectionMethod): void
  onConfirm(candidateIds: string[]): void
  onPreferenceChange(preference: DomainPreference): void
  onPanelOpenChange(open: boolean): void
  onFabPositionChange(position: PanelPosition): void
  onToggleGlobal(): Promise<boolean>
  onTogglePage(browserPageId: string): Promise<boolean | null>
  onRetryPage(browserPageId: string): void
  onRetryUploads(): void
  onRetryStart(): void
  onRestart(): void
  onStopDiscovery(): void
  onCancel(): void
  onLoadLibraryBooks(): Promise<BrowserLibraryBook[]>
  onImport(command: BrowserSessionImportCommand): Promise<BrowserSessionImportResult>
  onDisableSite(): void
  onDeleteAdaptation(): void
  onCopyDiagnostics(): void
}

interface Drag {
  start: PanelPosition
  origin: PanelPosition
  moved: boolean
}

/** The page owns image/session operations; the extension frame owns all interface rendering. */
export class ExtensionUi {
  readonly host = document.createElement('div')
  readonly shadow: ShadowRoot
  private readonly panel = document.createElement('div')
  private readonly frame = document.createElement('iframe')
  private readonly fab = document.createElement('button')
  private readonly pickMask = document.createElement('div')
  private readonly dragMask = document.createElement('div')
  private readonly origin = chrome.runtime.getURL('').replace(/\/$/, '')
  private readonly state: StudioState
  private panelDrag: Drag | null = null
  private fabDrag: Drag | null = null
  private suppressFabClick = false

  constructor(
    private readonly callbacks: UiCallbacks,
    preference: DomainPreference,
    title: string,
    knownSite: boolean
  ) {
    this.state = {
      open: preference.panelOpen,
      title,
      preference: { ...preference },
      tab: 'translate',
      view: 'idle',
      notice: {
        title: knownSite ? '已识别漫画站点' : '准备好开始阅读',
        message: '识别当前页面的漫画图片，确认后开始翻译。',
        tone: 'ready',
      },
      candidates: [],
      session: null,
      originalPageIds: [],
      translated: true,
      preparation: null,
      uploadError: null,
      retryStart: false,
      terms: [],
      imported: null,
    }
    this.host.id = 'saber-translator-extension-root'
    this.shadow = this.host.attachShadow({ mode: 'closed' })
    const style = document.createElement('style')
    style.textContent = HOST_STYLES
    this.fab.className = 'saber-fab'
    this.fab.type = 'button'
    this.fab.textContent = 'S'
    this.fab.setAttribute('aria-label', 'Saber 漫画翻译')
    this.fab.title = 'Saber 漫画翻译 · 拖动调整位置'
    this.panel.className = 'saber-panel'
    this.frame.title = 'Saber 漫画翻译'
    this.frame.src = chrome.runtime.getURL('panel.html')
    this.frame.addEventListener('load', () => this.publish())
    this.panel.append(this.frame)
    this.pickMask.className = 'saber-pick-mask'
    const tip = document.createElement('span')
    tip.textContent = '点击一张漫画图片，识别同类图片 · Esc 取消'
    this.pickMask.append(tip)
    this.dragMask.className = 'saber-drag-mask'
    this.shadow.append(style, this.fab, this.panel, this.pickMask, this.dragMask)
    window.addEventListener('message', this.onMessage)
    window.addEventListener('resize', this.reclamp)
    window.addEventListener('pointermove', this.pointerMove)
    window.addEventListener('pointerup', this.pointerEnd)
    window.addEventListener('pointercancel', this.pointerEnd)
    window.addEventListener('blur', this.pointerEnd)
    this.fab.addEventListener('pointerdown', event => {
      if (!event.isTrusted || event.button !== 0) return
      this.suppressFabClick = false
      this.fabDrag = this.beginDrag(this.fab, {
        x: event.clientX,
        y: event.clientY,
      })
      this.fab.setPointerCapture(event.pointerId)
    })
    this.fab.addEventListener('click', event => {
      if (event.isTrusted && !this.suppressFabClick) this.togglePanel()
    })
    document.documentElement.append(this.host)
    if (preference.fabPosition) this.place(this.fab, preference.fabPosition)
    this.setOpen(preference.panelOpen)
  }

  private publish(): void {
    this.frame.contentWindow?.postMessage(
      { channel: 'saber:state', state: this.state },
      this.origin
    )
  }
  private onMessage = (event: MessageEvent): void => {
    if (
      !event.isTrusted ||
      event.source !== this.frame.contentWindow ||
      event.origin !== this.origin ||
      event.data?.channel !== 'saber:command'
    )
      return
    const { id, action, payload } = event.data
    void this.command(action, payload).then(
      result => {
        if (id !== undefined)
          this.frame.contentWindow?.postMessage(
            { channel: 'saber:response', id, ok: true, result },
            this.origin
          )
      },
      error => {
        if (id !== undefined)
          this.frame.contentWindow?.postMessage(
            {
              channel: 'saber:response',
              id,
              ok: false,
              error: error instanceof Error ? error.message : String(error),
            },
            this.origin
          )
      }
    )
  }
  private async command(action: StudioAction, payload: any): Promise<unknown> {
    switch (action) {
      case 'ready':
        return this.publish()
      case 'tab':
        this.state.tab = payload
        return
      case 'close':
        this.setOpen(false, true)
        this.fab.focus()
        return
      case 'drag-start':
        this.panelDrag = this.beginDrag(this.panel, payload)
        this.dragMask.dataset.open = 'true'
        return
      case 'preference':
        this.state.preference = { ...this.state.preference, ...payload }
        this.callbacks.onPreferenceChange(this.state.preference)
        return this.publish()
      case 'discover':
        this.callbacks.onDiscover(payload)
        return
      case 'confirm':
        this.callbacks.onConfirm(payload)
        return
      case 'back':
        this.state.view = 'idle'
        this.state.candidates = []
        return this.publish()
      case 'toggle-global':
        this.state.translated = await this.callbacks.onToggleGlobal()
        this.publish()
        return this.state.translated
      case 'toggle-page': {
        const translated = await this.callbacks.onTogglePage(payload)
        if (translated !== null)
          this.state.originalPageIds = translated
            ? this.state.originalPageIds.filter(id => id !== payload)
            : [...new Set([...this.state.originalPageIds, payload])]
        this.publish()
        return translated
      }
      case 'retry-page':
        this.callbacks.onRetryPage(payload)
        return
      case 'retry-uploads':
        this.callbacks.onRetryUploads()
        return
      case 'retry-start':
        this.callbacks.onRetryStart()
        return
      case 'restart':
        this.callbacks.onRestart()
        return
      case 'stop-discovery':
        this.callbacks.onStopDiscovery()
        return
      case 'cancel':
        this.callbacks.onCancel()
        return
      case 'books':
        return this.callbacks.onLoadLibraryBooks()
      case 'import': {
        const result = await this.callbacks.onImport(payload)
        this.showImported(result)
        return result
      }
      case 'disable':
        this.callbacks.onDisableSite()
        return
      case 'delete-adaptation':
        this.callbacks.onDeleteAdaptation()
        return
      case 'diagnostics':
        this.callbacks.onCopyDiagnostics()
        return
      default:
        throw new Error('未知插件操作')
    }
  }
  private position(node: HTMLElement): PanelPosition {
    const rect = node.getBoundingClientRect()
    return { x: rect.x, y: rect.y }
  }
  private place(node: HTMLElement, position: PanelPosition): PanelPosition {
    const rect = node.getBoundingClientRect()
    const result = {
      x: Math.max(8, Math.min(position.x, innerWidth - rect.width - 8)),
      y: Math.max(8, Math.min(position.y, innerHeight - rect.height - 8)),
    }
    Object.assign(node.style, {
      left: `${result.x}px`,
      top: `${result.y}px`,
      right: 'auto',
      bottom: 'auto',
    })
    return result
  }
  private placePanelNearFab(): void {
    const gap = 12
    const fab = this.fab.getBoundingClientRect()
    this.panel.style.maxHeight = ''
    const panel = this.panel.getBoundingClientRect()
    const above = fab.top - gap - 8
    const below = innerHeight - fab.bottom - gap - 8
    const left = fab.left - gap - 8
    const right = innerWidth - fab.right - gap - 8
    if (above >= panel.height || below >= panel.height) {
      this.place(this.panel, {
        x: fab.right - panel.width,
        y: above >= panel.height ? fab.top - gap - panel.height : fab.bottom + gap,
      })
    } else if (left >= panel.width || right >= panel.width) {
      this.place(this.panel, {
        x: right >= panel.width ? fab.right + gap : fab.left - gap - panel.width,
        y: fab.top + (fab.height - panel.height) / 2,
      })
    } else {
      // Narrow windows: shorten the scrollable panel rather than cover its trigger.
      this.panel.style.maxHeight = `${Math.max(above, below)}px`
      const height = this.panel.getBoundingClientRect().height
      this.place(this.panel, {
        x: fab.right - panel.width,
        y: above >= below ? fab.top - gap - height : fab.bottom + gap,
      })
    }
  }
  private beginDrag(node: HTMLElement, start: PanelPosition): Drag {
    return { start, origin: this.position(node), moved: false }
  }
  private move(node: HTMLElement, drag: Drag, point: PanelPosition): void {
    const dx = point.x - drag.start.x,
      dy = point.y - drag.start.y
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true
    if (drag.moved) this.place(node, { x: drag.origin.x + dx, y: drag.origin.y + dy })
  }
  private pointerMove = (event: PointerEvent): void => {
    if (!event.isTrusted) return
    if (this.panelDrag)
      this.move(this.panel, this.panelDrag, { x: event.screenX, y: event.screenY })
    if (this.fabDrag) this.move(this.fab, this.fabDrag, { x: event.clientX, y: event.clientY })
  }
  private pointerEnd = (): void => {
    this.panelDrag = null
    this.dragMask.dataset.open = 'false'
    if (this.fabDrag?.moved) {
      this.suppressFabClick = true
      const position = this.position(this.fab)
      this.state.preference.fabPosition = position
      this.callbacks.onFabPositionChange(position)
      if (this.state.open) this.placePanelNearFab()
    }
    this.fabDrag = null
  }
  private reclamp = (): void => {
    this.place(this.fab, this.position(this.fab))
    if (this.state.open) this.placePanelNearFab()
  }
  setOpen(open: boolean, persist = false): void {
    const opening = open && !this.state.open
    this.state.open = open
    this.panel.dataset.open = String(open)
    this.fab.setAttribute('aria-expanded', String(open))
    if (open && (opening || !this.panel.style.left)) this.placePanelNearFab()
    if (persist) {
      this.state.preference.panelOpen = open
      this.callbacks.onPanelOpenChange(open)
    }
    this.publish()
  }
  togglePanel(): void {
    this.setOpen(this.panel.dataset.open !== 'true', true)
  }
  setAdaptation(rule: LearnedRule | null): void {
    if (rule) this.state.preference.rule = rule
    else delete this.state.preference.rule
    this.publish()
  }
  showCandidates(candidates: ImageCandidate[]): void {
    this.state.candidates = candidates.map(({ id, sourceUrl, width, height }) => ({
      id,
      sourceUrl,
      width,
      height,
    }))
    this.state.view = candidates.length ? 'candidates' : 'idle'
    this.state.tab = 'translate'
    this.setOpen(true)
    this.setStatus(
      candidates.length ? `找到 ${candidates.length} 张图片` : '未找到漫画图片',
      candidates.length
        ? '取消勾选不需要的图片，然后开始翻译。'
        : '可切换为点选同类图片，再试一次。',
      candidates.length ? 'ready' : 'error'
    )
  }
  showSession(session: BrowserSessionDto, originals: ReadonlySet<string> = new Set()): void {
    if (this.state.session?.id !== session.id) {
      this.state.imported = null
      this.state.translated = true
      this.state.terms = []
    }
    this.state.session = session
    this.state.originalPageIds = [...originals]
    this.state.preparation = null
    this.state.retryStart = session.pendingStart
    this.state.view = session.state === 'cancelled' && !session.pages.length ? 'idle' : 'progress'
    const titles = {
      idle: '等待图片',
      queued: '任务已进入 Saber 队列',
      translating: '正在逐张生成译图',
      completed: '当前图片已全部完成',
      partial: '部分图片翻译失败',
      failed: '图片翻译失败',
      cancelled: '任务已取消',
    }
    const busy = session.pages.some(page => page.state === 'queued' || page.state === 'translating')
    this.setStatus(
      session.taskState === 'paused'
        ? '任务已暂停，可在任务中心恢复'
        : session.taskState === 'interrupted'
          ? '任务已中断，可在任务中心继续'
          : titles[session.state],
      session.pendingStart
        ? '图片已准备好，可以重试启动翻译。'
        : '译图显示在原位置；需要保留请先导入书架。',
      ['failed', 'partial'].includes(session.state) ? 'error' : busy ? 'busy' : 'ready'
    )
  }
  showPreparationProgress(processed: number, total: number, failed: number): void {
    this.state.preparation = { processed, total, failed }
    this.state.view = 'progress'
    this.setOpen(true)
    this.setStatus('正在准备漫画图片', '正在读取图片并发送到本机 Saber。', 'busy')
  }
  hidePreparationProgress(): void {
    this.state.preparation = null
    this.publish()
  }
  showImported(result: BrowserSessionImportResult): void {
    this.state.imported = result
    this.setStatus(
      `已导入《${result.bookTitle}》`,
      `${result.chapterTitle} · 已导入 ${result.importedPages} 张图片${result.omittedPages ? `，忽略 ${result.omittedPages} 张未进入任务的图片` : ''}`
    )
  }
  showTerms(entries: Array<{ source?: string; target?: string }>): void {
    this.state.terms = entries
    this.publish()
  }
  setStatus(title: string, message: string, tone: 'ready' | 'busy' | 'error' = 'ready'): void {
    this.state.notice = { title, message, tone }
    this.fab.dataset.state = tone
    this.publish()
  }
  showError(error: { code: string; message: string }): void {
    this.state.retryStart = false
    const hints: Record<string, string> = {
      not_paired: '请点击浏览器工具栏中的扩展图标完成配对。',
      saber_unreachable: '请启动 Saber，并确认扩展端口与 GUI 一致。',
      invalid_extension_token: '令牌已失效，请重新配对。',
      integration_disabled: '请在 Saber GUI 中允许浏览器扩展连接。',
      dom_agent_unavailable: '请在配置中填写网页识别助手配置。',
    }
    this.setOpen(true)
    this.setStatus(
      error.message,
      hints[error.code] ?? '请重试，或切换另一种图片识别方式。',
      'error'
    )
  }
  showStartError(error: { code: string; message: string }): void {
    this.showError(error)
    this.state.retryStart = true
    this.publish()
  }
  showUploadError(count: number, error: { code: string; message: string }): void {
    this.state.uploadError = { count, message: error.message }
    this.setOpen(true)
    this.setStatus(`${count} 张图片尚未导入`, '其他已成功导入的图片会继续处理。', 'error')
  }
  clearUploadError(): void {
    this.state.uploadError = null
    this.publish()
  }
  startPicking(): void {
    this.fab.focus()
    this.pickMask.dataset.open = 'true'
    this.panel.style.visibility = 'hidden'
  }
  stopPicking(): void {
    this.pickMask.dataset.open = 'false'
    this.panel.style.visibility = ''
    if (this.state.notice.tone === 'busy')
      this.setStatus('可以重新选择图片', '点击图片选择按钮，或切换其他识别方式。')
  }
  pickingMask(): HTMLElement {
    return this.pickMask
  }
  remove(): void {
    window.removeEventListener('message', this.onMessage)
    window.removeEventListener('resize', this.reclamp)
    window.removeEventListener('pointermove', this.pointerMove)
    window.removeEventListener('pointerup', this.pointerEnd)
    window.removeEventListener('pointercancel', this.pointerEnd)
    window.removeEventListener('blur', this.pointerEnd)
    this.host.remove()
  }
}
