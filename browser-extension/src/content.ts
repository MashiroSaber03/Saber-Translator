import {
  adapterFor,
  candidateForSource,
  domSummary,
  elementCandidate,
  isKnownComicHost,
  ruleFromCandidate,
  scanAdapter,
  scanGeneric,
  scanRule,
  similarTo,
  validateSuggestedRule,
  type ImageCandidate,
} from './discovery'
import { ReplacementManager } from './replacement'
import type {
  BackgroundRequest,
  BackgroundResponse,
  BrowserLibraryBook,
  BrowserPageDto,
  BrowserSessionDto,
  BrowserSessionImportCommand,
  BrowserSessionImportResult,
  ContextTranslateMessage,
  DetectionMethod,
  DomDetectionResult,
  DomainPreference,
  LearnedRule,
  PanelPosition,
  ResultImagePayload,
  UploadSource,
} from './types'
import { ExtensionUi } from './ui'

class ExtensionRequestError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean,
  ) {
    super(message)
  }
}

async function send<T>(request: BackgroundRequest): Promise<T> {
  const response = await chrome.runtime.sendMessage(request) as BackgroundResponse<T>
  if (!response?.ok) {
    const error = response?.error ?? {
      code: 'extension_error',
      message: '扩展后台没有返回结果',
      retryable: true,
    }
    throw new ExtensionRequestError(error.code, error.message, error.retryable)
  }
  return response.data
}

function normalizedPageUrlFrom(value: string): string {
  const url = new URL(value)
  url.hash = ''
  return url.toString()
}

function normalizedPageUrl(): string {
  return normalizedPageUrlFrom(location.href)
}

function domAgentPageUrl(): string {
  const url = new URL(location.href)
  url.username = ''
  url.password = ''
  url.search = ''
  url.hash = ''
  return url.toString()
}

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return [...new Uint8Array(digest)]
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('')
}

function errorDetails(error: unknown): { code: string; message: string } {
  if (error instanceof ExtensionRequestError) {
    return { code: error.code, message: error.message }
  }
  return {
    code: 'extension_error',
    message: error instanceof Error ? error.message : '发生未知错误',
  }
}

function extensionFor(candidate: ImageCandidate): string {
  const url = candidate.sourceUrl ?? ''
  const path = (() => {
    try { return new URL(url).pathname } catch { return '' }
  })()
  const match = /\.([a-z0-9]{2,8})$/i.exec(path)
  const extension = match?.[1]?.toLowerCase()
  return extension && ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff'].includes(extension)
    ? extension
    : 'png'
}

function metadataSourceUrl(candidate: ImageCandidate): string | undefined {
  const value = candidate.sourceUrl
  return value?.startsWith('http://') || value?.startsWith('https://')
    ? value
    : undefined
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => typeof reader.result === 'string'
      ? resolve(reader.result)
      : reject(new Error('图片数据读取失败'))
    reader.onerror = () => reject(reader.error ?? new Error('图片数据读取失败'))
    reader.readAsDataURL(blob)
  })
}

function resultObjectUrl(result: ResultImagePayload): string {
  const binary = atob(result.base64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return URL.createObjectURL(new Blob([bytes], { type: result.mimeType }))
}

async function waitForImageDecode(image: HTMLImageElement): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | null = null
  try {
    await Promise.race([
      image.decode(),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new ExtensionRequestError(
          'source_timeout',
          '等待网页图片加载超时',
          true,
        )), 30_000)
      }),
    ])
  } catch (error) {
    if (error instanceof ExtensionRequestError) throw error
    // A direct canvas read can still succeed when decode() rejects.
  } finally {
    if (timer !== null) clearTimeout(timer)
  }
}

async function elementDataUrl(candidate: ImageCandidate): Promise<string> {
  if (candidate.element instanceof HTMLCanvasElement) {
    try {
      return candidate.element.toDataURL('image/png')
    } catch {
      throw new ExtensionRequestError(
        'canvas_unreadable',
        'Canvas 受到跨域保护，无法读取像素',
        false,
      )
    }
  }
  if (!(candidate.element instanceof HTMLImageElement)) {
    throw new ExtensionRequestError('source_fetch_failed', '无法读取背景图片像素', true)
  }
  const image = candidate.element
  if (!image.complete) {
    await waitForImageDecode(image)
  }
  const width = image.naturalWidth || image.width
  const height = image.naturalHeight || image.height
  if (!width || !height) {
    throw new ExtensionRequestError('image_not_loaded', '图片尚未加载完成', true)
  }
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas 2D 上下文不可用')
  try {
    context.drawImage(image, 0, 0)
    return canvas.toDataURL('image/png')
  } catch {
    throw new ExtensionRequestError(
      'source_forbidden',
      '图片源拒绝下载且页面像素受到跨域保护',
      false,
    )
  }
}

async function uploadSource(candidate: ImageCandidate): Promise<UploadSource> {
  if (candidate.kind === 'canvas') {
    return { kind: 'data-url', value: await elementDataUrl(candidate) }
  }
  const source = candidate.sourceUrl
  if (!source) {
    return { kind: 'data-url', value: await elementDataUrl(candidate) }
  }
  if (source.startsWith('data:')) {
    return { kind: 'data-url', value: source }
  }
  if (source.startsWith('blob:')) {
    try {
      const blob = await fetch(source).then(response => response.blob())
      return { kind: 'data-url', value: await blobToDataUrl(blob) }
    } catch {
      throw new ExtensionRequestError('blob_expired', '页面中的 Blob 图片已经失效', true)
    }
  }
  return { kind: 'url', value: source }
}

interface TaskContext {
  generation: number
  sessionId: string
}

interface UploadFailure {
  candidate: ImageCandidate
  error: { code: string; message: string }
}

interface UploadBatchResult {
  uploaded: number
  failed: number
}

export class PageController {
  private readonly pageUrl = normalizedPageUrl()
  private readonly hostname = location.hostname
  private preference!: DomainPreference
  private ui: ExtensionUi | null = null
  private readonly replacement = new ReplacementManager()
  private session: BrowserSessionDto | null = null
  private candidates: ImageCandidate[] = []
  private readonly candidatesByIdentity = new Map<string, ImageCandidate>()
  private readonly candidatesByClientKey = new Map<string, ImageCandidate>()
  private readonly clientKeys = new Map<string, string>()
  private readonly pageIdsByClientKey = new Map<string, string>()
  private readonly appliedRetries = new Map<string, number>()
  private readonly resultUrls = new Map<string, string>()
  private readonly originalPageIds = new Set<string>()
  private readonly uploadFailures = new Map<string, UploadFailure>()
  private readonly uploadsInFlight = new Set<string>()
  private readonly ordinalsByIdentity = new Map<string, number>()
  private initialization: Promise<void> | null = null
  private activeMethod: DetectionMethod
  private usingAdapter = false
  private activeRule: LearnedRule | null = null
  private observer: MutationObserver | null = null
  private observerTimer: number | null = null
  private pollTimer: number | null = null
  private termsPollTick = 0
  private disposed = false
  private discoveryStopped = false
  private cancelled = false
  private imported = false
  private deletingAdaptation = false
  private taskStarting = false
  private retryingUploadsFor: number | null = null
  private nextOrdinal = 1
  private taskGeneration = 0
  private readonly pollsInFlight = new Set<number>()
  private replacementQueue: Promise<void> = Promise.resolve()
  private readonly imageLoadListener = (event: Event) => {
    if (event.target instanceof HTMLImageElement) this.scheduleLazyScan()
  }

  constructor() {
    this.activeMethod = 'adapter'
  }

  private currentTask(): TaskContext | null {
    return this.session && !this.imported
      ? { generation: this.taskGeneration, sessionId: this.session.id }
      : null
  }

  private isCurrentTask(task: TaskContext): boolean {
    return !this.disposed
      && task.generation === this.taskGeneration
      && task.sessionId === this.session?.id
  }

  private async queueReplacement<T>(
    task: TaskContext | null,
    operation: () => Promise<T>,
  ): Promise<T | null> {
    let result: T | null = null
    const run = async (): Promise<void> => {
      if (task && !this.isCurrentTask(task)) return
      result = await operation()
    }
    const pending = this.replacementQueue.then(run, run)
    this.replacementQueue = pending.catch(() => undefined)
    await pending
    return result
  }

  initialize(): Promise<void> {
    this.initialization ??= this.initializeOnce()
    return this.initialization
  }

  private async initializeOnce(): Promise<void> {
    this.preference = await send<DomainPreference>({
      type: 'get-preference',
      hostname: this.hostname,
    })
    if (this.preference.disabled) return
    this.activeMethod = this.preference.method
    this.activeRule = this.preference.rule ?? null
    this.ui = new ExtensionUi(
      {
        onDiscover: method => void this.discover(method),
        onConfirm: ids => void this.confirm(ids),
        onPreferenceChange: preference => void this.updatePreference(preference),
        onPanelOpenChange: panelOpen => void this.updatePanelPreference({ panelOpen }),
        onPanelPositionChange: panelPosition => void this.updatePanelPreference({ panelPosition }),
        onFabPositionChange: fabPosition => void this.updatePanelPreference({ fabPosition }),
        onToggleGlobal: () => this.toggleAllPages(),
        onTogglePage: browserPageId => this.togglePage(browserPageId),
        onRetryPage: browserPageId => void this.retry(browserPageId),
        onRetryUploads: () => void this.retryFailedUploads(),
        onStopDiscovery: () => this.stopDiscovery(),
        onCancel: () => void this.cancel(),
        onLoadLibraryBooks: () => this.loadLibraryBooks(),
        onImport: command => this.importToLibrary(command),
        onDisableSite: () => void this.disableSite(),
        onDeleteAdaptation: () => void this.deleteAdaptation(),
        onCopyDiagnostics: () => void this.copyDiagnostics(),
      },
      this.preference,
      document.title || this.hostname,
      isKnownComicHost(this.hostname),
    )
    await this.restoreActiveSession()
  }

  private async rememberActiveSession(sessionId: string): Promise<void> {
    try {
      await send<{ saved: boolean }>({
        type: 'set-active-session',
        pageUrl: this.pageUrl,
        sessionId,
      })
    } catch (error) {
      console.warn('Saber extension could not remember the active session', error)
    }
  }

  private async forgetActiveSession(sessionId?: string): Promise<void> {
    try {
      await send<{ cleared: boolean }>({
        type: 'clear-active-session',
        pageUrl: this.pageUrl,
        ...(sessionId ? { sessionId } : {}),
      })
    } catch (error) {
      console.warn('Saber extension could not clear the active session', error)
    }
  }

  private async restoreCandidates(): Promise<ImageCandidate[]> {
    if (this.activeRule) {
      const ruled = scanRule(this.activeRule)
      if (ruled.length) {
        this.usingAdapter = false
        return ruled
      }
      this.activeRule = null
      delete this.preference.rule
      this.ui?.setAdaptationSaved(false)
      try {
        await this.persistPreference()
      } catch (error) {
        console.warn('Saber extension could not remove a stale adaptation rule', error)
      }
    }
    const adapter = adapterFor(this.hostname)
    if (this.activeMethod === 'adapter' && adapter) {
      const adapted = scanAdapter(adapter)
      if (adapted.length) {
        this.usingAdapter = true
        return adapted
      }
    }
    this.usingAdapter = false
    return scanGeneric()
  }

  private async restoreActiveSession(): Promise<void> {
    let remembered: { sessionId: string } | null
    let session: BrowserSessionDto
    try {
      remembered = await send<{ sessionId: string } | null>({
        type: 'get-active-session',
        pageUrl: this.pageUrl,
      })
      if (!remembered) return
      session = await send<BrowserSessionDto>({
        type: 'get-session',
        sessionId: remembered.sessionId,
      })
      if (
        normalizedPageUrlFrom(session.pageUrl) !== this.pageUrl
        || session.state === 'cancelled'
        || session.pages.length === 0
      ) {
        await this.forgetActiveSession(remembered.sessionId)
        return
      }
    } catch (error) {
      if (error instanceof ExtensionRequestError && !error.retryable) {
        await this.forgetActiveSession()
      }
      return
    }

    try {
      const generation = ++this.taskGeneration
      this.session = session
      this.imported = false
      this.cancelled = false
      this.discoveryStopped = false
      this.nextOrdinal = Math.max(...session.pages.map(page => page.ordinal)) + 1
      const candidates = await this.restoreCandidates()
      this.candidates = candidates
      this.registerCandidates(candidates)
      const pagesByClientKey = new Map(
        session.pages.map(page => [page.clientPageKey, page])
      )
      for (const page of session.pages) {
        this.pageIdsByClientKey.set(page.clientPageKey, page.id)
      }
      for (const candidate of candidates) {
        const clientPageKey = await this.clientKey(candidate)
        if (this.disposed || generation !== this.taskGeneration) return
        const page = pagesByClientKey.get(clientPageKey)
        if (!page) continue
        this.candidatesByClientKey.set(clientPageKey, candidate)
        this.ordinalsByIdentity.set(candidate.sourceIdentity, page.ordinal)
      }
      const task = { generation, sessionId: session.id }
      this.showSession(session)
      await this.applyCompletedPages(session, task)
      if (!this.isCurrentTask(task)) return
      this.startObserver()
      this.startPolling(250, task)
    } catch (error) {
      this.ui?.showError(errorDetails(error))
    }
  }

  private ordinalFor(candidate: ImageCandidate): number {
    const existing = this.ordinalsByIdentity.get(candidate.sourceIdentity)
    if (existing !== undefined) return existing
    const ordinal = this.nextOrdinal++
    this.ordinalsByIdentity.set(candidate.sourceIdentity, ordinal)
    return ordinal
  }

  private showSession(session: BrowserSessionDto): void {
    this.ui?.showSession(session, this.originalPageIds)
    this.showUploadFailures()
  }

  private showUploadFailures(): void {
    const firstFailure = this.uploadFailures.values().next().value as UploadFailure | undefined
    if (firstFailure) {
      this.ui?.showUploadError(this.uploadFailures.size, firstFailure.error)
    } else {
      this.ui?.clearUploadError()
    }
  }

  async translateContextImage(srcUrl: string): Promise<void> {
    if (this.preference?.disabled) return
    if (!this.ui) await this.initialize()
    if (!this.ui) return
    const all = scanGeneric()
    const candidate = candidateForSource(all, srcUrl)
      ?? [...document.querySelectorAll('img')]
        .map(elementCandidate)
        .find((item): item is ImageCandidate => item?.sourceUrl === srcUrl)
      ?? null
    if (!candidate) {
      this.ui.showError({ code: 'image_not_found', message: '没有找到右键选择的图片元素' })
      return
    }
    if (this.taskStarting) return
    this.taskStarting = true
    this.registerCandidates([candidate])
    this.cancelled = false
    this.ui.setOpen(true)
    this.ui.setStatus('正在提交单张图片', '右键操作已经视为确认，无需再次选择。', 'busy')
    try {
      this.discoveryStopped = true
      const task = await this.createSession()
      if (!task) return
      const result = await this.uploadCandidates([candidate], task, true)
      if (!this.isCurrentTask(task)) return
      if (result.uploaded === 0) {
        this.showUploadFailures()
        return
      }
      await this.startUploadedPages(task)
      this.startPolling(250, task)
    } catch (error) {
      this.ui.showError(errorDetails(error))
    } finally {
      this.taskStarting = false
    }
  }

  async dispose(): Promise<void> {
    this.disposed = true
    this.taskGeneration += 1
    this.stopDiscovery()
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer)
    try {
      await this.queueReplacement(null, () => this.replacement.restoreAll())
    } catch (error) {
      console.warn('Saber extension could not fully restore the page', error)
    }
    for (const url of this.resultUrls.values()) URL.revokeObjectURL(url)
    this.resultUrls.clear()
    this.originalPageIds.clear()
    this.ui?.remove()
    this.ui = null
  }

  private async discover(method: DetectionMethod): Promise<void> {
    if (!this.ui) return
    this.ui.setStatus('正在识别页面图片', '只读取图片节点和尺寸，不会自动上传。', 'busy')
    try {
      let found: ImageCandidate[]
      if (this.activeRule) {
        const ruled = scanRule(this.activeRule)
        if (ruled.length) {
          this.activeMethod = method
          this.usingAdapter = false
          this.candidates = ruled
          this.registerCandidates(ruled)
          this.ui.showCandidates(ruled)
          return
        }
        this.activeRule = null
        delete this.preference.rule
        await this.persistPreference()
        this.ui.setAdaptationSaved(false)
      }
      if (method === 'similar') {
        this.activeMethod = method
        this.usingAdapter = false
        this.startSimilarPicker()
        return
      }
      if (method === 'dom-agent') {
        const generic = scanGeneric()
        if (!generic.length) {
          this.ui.showCandidates([])
          return
        }
        const result = await send<DomDetectionResult>({
          type: 'dom-detection',
          payload: {
            pageUrl: domAgentPageUrl(),
            pageTitle: document.title,
            nodes: domSummary(generic),
          },
        })
        const selected = new Set(result.nodeIds)
        const selectedCandidates = generic.filter(candidate => selected.has(candidate.id))
        const suggested = validateSuggestedRule(result.selector, selectedCandidates)
        if (suggested) {
          this.activeRule = suggested.rule
          found = suggested.candidates
        } else {
          this.activeRule = null
          found = selectedCandidates
        }
        this.usingAdapter = false
      } else {
        const adapter = adapterFor(this.hostname)
        if (adapter) {
          const adapted = scanAdapter(adapter)
          this.usingAdapter = adapted.length > 0
          this.activeRule = null
          found = this.usingAdapter ? adapted : scanGeneric()
        } else {
          this.usingAdapter = false
          found = scanGeneric()
        }
      }
      this.activeMethod = method
      this.candidates = found
      this.registerCandidates(found)
      this.ui.showCandidates(found)
    } catch (error) {
      this.ui.showError(errorDetails(error))
    }
  }

  private startSimilarPicker(): void {
    if (!this.ui) return
    const mask = this.ui.pickingMask()
    this.ui.startPicking()
    const cleanup = () => {
      mask.removeEventListener('click', choose)
      document.removeEventListener('keydown', keydown, true)
      this.ui?.stopPicking()
    }
    const choose = (event: MouseEvent) => {
      event.preventDefault()
      mask.style.pointerEvents = 'none'
      const target = document.elementFromPoint(event.clientX, event.clientY)
      mask.style.pointerEvents = ''
      const candidate = target ? elementCandidate(target) : null
      if (!candidate) {
        this.ui?.showError({ code: 'not_an_image', message: '请选择漫画图片或 Canvas' })
        return
      }
      cleanup()
      this.activeMethod = 'similar'
      this.usingAdapter = false
      this.activeRule = ruleFromCandidate(candidate)
      this.candidates = similarTo(candidate)
      this.registerCandidates(this.candidates)
      this.ui?.showCandidates(this.candidates)
    }
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') cleanup()
    }
    mask.addEventListener('click', choose)
    document.addEventListener('keydown', keydown, true)
  }

  private async confirm(ids: string[]): Promise<void> {
    if (!this.ui) return
    const selected = this.candidates.filter(candidate => ids.includes(candidate.id))
    if (!selected.length) {
      this.ui.showError({ code: 'no_selection', message: '请至少选择一张图片' })
      return
    }
    if (this.taskStarting) return
    this.taskStarting = true
    this.cancelled = false
    this.discoveryStopped = false
    if (!this.activeRule && !this.usingAdapter) {
      this.activeRule = ruleFromCandidate(selected[0]!)
    }
    const confirmedPreference: DomainPreference = {
      ...this.preference,
      method: this.activeMethod,
    }
    if (this.activeRule) confirmedPreference.rule = this.activeRule
    else delete confirmedPreference.rule
    this.preference = confirmedPreference
    try {
      await this.persistPreference()
      this.ui.setAdaptationSaved(Boolean(this.activeRule))
      this.ui.setStatus('正在导入漫画图片', '图片会按网页顺序进入当前隐藏会话。', 'busy')
      const task = await this.createSession()
      if (!task) return
      const result = await this.uploadCandidates(selected, task, true)
      if (this.cancelled || !this.isCurrentTask(task)) return
      if (result.uploaded === 0) {
        this.showUploadFailures()
        return
      }
      await this.startUploadedPages(task)
      if (!this.discoveryStopped) this.startObserver()
      this.startPolling(250, task)
    } catch (error) {
      this.ui.showError(errorDetails(error))
    } finally {
      this.taskStarting = false
    }
  }

  private async createSession(): Promise<TaskContext | null> {
    const generation = ++this.taskGeneration
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer)
    this.pollTimer = null
    this.disconnectObserver()
    await this.forgetActiveSession(this.session?.id)
    this.session = null
    this.imported = false
    await this.queueReplacement(null, () => this.replacement.restoreAll())
    if (this.disposed || generation !== this.taskGeneration) return null
    for (const url of this.resultUrls.values()) URL.revokeObjectURL(url)
    this.resultUrls.clear()
    this.originalPageIds.clear()
    this.candidatesByClientKey.clear()
    this.pageIdsByClientKey.clear()
    this.appliedRetries.clear()
    this.uploadFailures.clear()
    this.uploadsInFlight.clear()
    this.retryingUploadsFor = null
    this.ordinalsByIdentity.clear()
    this.ui?.clearUploadError()
    this.ui?.hidePreparationProgress()
    this.nextOrdinal = 1
    this.termsPollTick = 0
    const session = await send<BrowserSessionDto>({
      type: 'create-session',
      payload: {
        pageUrl: this.pageUrl,
        pageTitle: document.title || this.hostname,
        mode: this.preference.mode,
        glossaryEnabled: this.preference.glossaryEnabled,
        autoTermsEnabled: this.preference.autoTermsEnabled,
      },
    })
    if (this.disposed || generation !== this.taskGeneration) return null
    this.session = session
    await this.rememberActiveSession(session.id)
    if (this.disposed || generation !== this.taskGeneration) return null
    this.ui?.showTerms([])
    this.showSession(session)
    return { generation, sessionId: session.id }
  }

  private async uploadCandidates(
    candidates: ImageCandidate[],
    task: TaskContext,
    reportProgress = false,
  ): Promise<UploadBatchResult> {
    const queued = candidates.map(candidate => ({
      candidate,
      ordinal: this.ordinalFor(candidate),
      distance: Math.abs(
        candidate.element.getBoundingClientRect().top - window.innerHeight / 2,
      ),
    })).sort((left, right) => left.distance - right.distance)
    let index = 0
    let uploaded = 0
    let processed = 0
    let batchFailed = 0
    if (reportProgress && this.isCurrentTask(task)) {
      this.ui?.showPreparationProgress(0, queued.length, 0)
    }
    const workers = Array.from({ length: Math.min(2, queued.length) }, async () => {
      while (index < queued.length) {
        if (this.cancelled || !this.isCurrentTask(task)) return
        const current = index
        index += 1
        const item = queued[current]
        if (!item) continue
        const uploadKey = `${task.generation}:${item.candidate.sourceIdentity}`
        if (this.uploadsInFlight.has(uploadKey)) {
          if (reportProgress && this.isCurrentTask(task)) {
            processed += 1
            this.ui?.showPreparationProgress(processed, queued.length, batchFailed)
          }
          continue
        }
        this.uploadsInFlight.add(uploadKey)
        try {
          if (await this.uploadCandidate(item.candidate, item.ordinal, task)) {
            uploaded += 1
            this.uploadFailures.delete(item.candidate.sourceIdentity)
          }
        } catch (error) {
          if (this.isCurrentTask(task)) {
            batchFailed += 1
            this.uploadFailures.set(item.candidate.sourceIdentity, {
              candidate: item.candidate,
              error: errorDetails(error),
            })
          }
        } finally {
          this.uploadsInFlight.delete(uploadKey)
          if (reportProgress && this.isCurrentTask(task)) {
            processed += 1
            this.ui?.showPreparationProgress(processed, queued.length, batchFailed)
          }
        }
      }
    })
    await Promise.all(workers)
    if (this.isCurrentTask(task)) this.showUploadFailures()
    return { uploaded, failed: this.uploadFailures.size }
  }

  private async clientKey(candidate: ImageCandidate): Promise<string> {
    const existing = this.clientKeys.get(candidate.sourceIdentity)
    if (existing) return existing
    const identity = candidate.kind === 'canvas'
      ? `canvas:${await elementDataUrl(candidate)}`
      : candidate.sourceIdentity
    const key = await sha256(identity)
    this.clientKeys.set(candidate.sourceIdentity, key)
    return key
  }

  private async uploadCandidate(
    candidate: ImageCandidate,
    ordinal: number,
    task: TaskContext,
  ): Promise<boolean> {
    if (!this.isCurrentTask(task)) return false
    const clientPageKey = await this.clientKey(candidate)
    if (!this.isCurrentTask(task)) return false
    this.candidatesByClientKey.set(clientPageKey, candidate)
    const existingId = this.pageIdsByClientKey.get(clientPageKey)
    if (existingId) return true
    let source = await uploadSource(candidate)
    try {
      const page = await send<BrowserPageDto>({
        type: 'upload-page',
        payload: {
          sessionId: task.sessionId,
          clientPageKey,
          ordinal,
          logicalPath: `${String(ordinal).padStart(5, '0')}.${extensionFor(candidate)}`,
          sourceUrl: metadataSourceUrl(candidate),
          source,
        },
      })
      if (!this.isCurrentTask(task)) return false
      this.pageIdsByClientKey.set(clientPageKey, page.id)
      return true
    } catch (error) {
      if (
        source.kind === 'url'
        && error instanceof ExtensionRequestError
        && ['source_fetch_failed', 'source_forbidden', 'source_timeout'].includes(error.code)
      ) {
        source = {
          kind: 'data-url',
          value: await elementDataUrl(candidate),
        }
        const page = await send<BrowserPageDto>({
          type: 'upload-page',
          payload: {
            sessionId: task.sessionId,
            clientPageKey,
            ordinal,
            logicalPath: `${String(ordinal).padStart(5, '0')}.png`,
            sourceUrl: metadataSourceUrl(candidate),
            source,
          },
        })
        if (!this.isCurrentTask(task)) return false
        this.pageIdsByClientKey.set(clientPageKey, page.id)
        return true
      }
      throw error
    }
  }

  private startObserver(): void {
    this.observer?.disconnect()
    document.removeEventListener('load', this.imageLoadListener, true)
    document.addEventListener('load', this.imageLoadListener, true)
    this.observer = new MutationObserver(() => {
      if (this.discoveryStopped || this.disposed) return
      this.scheduleLazyScan()
    })
    this.observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: [
        'src',
        'srcset',
        'data-src',
        'data-original',
        'data-lazy-src',
        'data-url',
        'data-srcset',
        'class',
        'style',
      ],
    })
  }

  private scheduleLazyScan(): void {
    if (this.discoveryStopped || this.disposed) return
    if (this.observerTimer !== null) return
    this.observerTimer = window.setTimeout(() => {
      this.observerTimer = null
      void this.discoverLazyImages().catch((error) => {
        this.ui?.showError(errorDetails(error))
      })
    }, 500)
  }

  private async discoverLazyImages(): Promise<void> {
    const task = this.currentTask()
    if (!task || this.discoveryStopped) return
    await this.queueReplacement(
      task,
      () => this.replacement.reconcileDisplayedResults(),
    )
    if (!this.isCurrentTask(task) || this.discoveryStopped) return
    const adapter = this.usingAdapter ? adapterFor(this.hostname) : null
    const found = adapter
      ? scanAdapter(adapter)
      : this.activeRule
        ? scanRule(this.activeRule)
        : scanGeneric()
    const added: ImageCandidate[] = []
    for (const candidate of found) {
      const existing = this.candidatesByIdentity.get(candidate.sourceIdentity)
      if (existing) {
        for (const binding of candidate.bindings) {
          if (!existing.bindings.includes(binding)) existing.bindings.push(binding)
        }
        await this.replacement.syncBindings(existing)
        continue
      }
      added.push(candidate)
    }
    if (!added.length) return
    this.registerCandidates(added)
    const result = await this.uploadCandidates(added, task)
    if (!this.isCurrentTask(task)) return
    if (result.uploaded === 0) return
    await this.startUploadedPages(task)
    this.startPolling(250, task)
  }

  private registerCandidates(candidates: ImageCandidate[]): void {
    for (const candidate of candidates) {
      const existing = this.candidatesByIdentity.get(candidate.sourceIdentity)
      if (existing) {
        for (const binding of candidate.bindings) {
          if (!existing.bindings.includes(binding)) existing.bindings.push(binding)
        }
      } else {
        this.candidatesByIdentity.set(candidate.sourceIdentity, candidate)
      }
    }
  }

  private startPolling(
    delay = 250,
    task: TaskContext | null = this.currentTask(),
  ): void {
    if (!task || !this.isCurrentTask(task)) return
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer)
    this.pollTimer = window.setTimeout(() => void this.poll(task), delay)
  }

  private async startUploadedPages(task: TaskContext): Promise<void> {
    if (!this.isCurrentTask(task) || this.cancelled) return
    const session = await send<BrowserSessionDto>({
      type: 'start-session',
      sessionId: task.sessionId,
    })
    if (!this.isCurrentTask(task)) return
    this.session = session
    this.showSession(this.session)
  }

  private async poll(task: TaskContext): Promise<void> {
    if (!this.isCurrentTask(task)) return
    if (this.pollsInFlight.has(task.generation)) {
      this.startPolling(250, task)
      return
    }
    this.pollsInFlight.add(task.generation)
    try {
      let session = await send<BrowserSessionDto>({
        type: 'get-session',
        sessionId: task.sessionId,
      })
      if (!this.isCurrentTask(task)) return
      if (
        !this.cancelled
        && session.state !== 'cancelled'
        && session.pages.some(page => page.pageId === null && page.state === 'queued')
      ) {
        session = await send<BrowserSessionDto>({
          type: 'start-session',
          sessionId: task.sessionId,
        })
        if (!this.isCurrentTask(task)) return
      }
      this.session = session
      if (session.state === 'cancelled') {
        await this.forgetActiveSession(session.id)
      }
      for (const page of session.pages) {
        this.pageIdsByClientKey.set(page.clientPageKey, page.id)
      }
      this.showSession(session)
      await this.applyCompletedPages(session, task)
      if (!this.isCurrentTask(task)) return
      this.termsPollTick += 1
      const busy = session.state === 'queued' || session.state === 'translating'
      if (
        (this.preference.glossaryEnabled || this.preference.autoTermsEnabled)
        && (this.termsPollTick % 3 === 0 || !busy)
      ) {
        const terms = await send<{ glossary?: { entries?: Array<{ source?: string; target?: string }> } }>({
          type: 'get-terms',
          sessionId: task.sessionId,
        })
        if (!this.isCurrentTask(task)) return
        this.ui?.showTerms(terms.glossary?.entries ?? [])
      }
      if (busy) this.startPolling(1_500, task)
    } catch (error) {
      if (!this.isCurrentTask(task)) return
      if (error instanceof ExtensionRequestError && !error.retryable) {
        await this.forgetActiveSession(task.sessionId)
      }
      this.ui?.showError(errorDetails(error))
      if (!(error instanceof ExtensionRequestError) || error.retryable) {
        this.startPolling(5_000, task)
      }
    } finally {
      this.pollsInFlight.delete(task.generation)
    }
  }

  private async applyCompletedPages(
    session: BrowserSessionDto,
    task: TaskContext,
  ): Promise<void> {
    for (const page of session.pages) {
      if (!this.isCurrentTask(task)) return
      if (!page.resultReady || page.state !== 'completed') continue
      const candidate = this.candidatesByClientKey.get(page.clientPageKey)
      if (!candidate) continue
      if (this.appliedRetries.get(page.id) === page.retryCount) continue
      const result = await send<ResultImagePayload>({
        type: 'fetch-result',
        sessionId: task.sessionId,
        browserPageId: page.id,
      })
      const resultUrl = resultObjectUrl(result)
      if (!this.isCurrentTask(task)) {
        URL.revokeObjectURL(resultUrl)
        return
      }
      try {
        const showingTranslated = await this.queueReplacement(
          task,
          () => this.replacement.apply(candidate, resultUrl),
        )
        if (showingTranslated === null) {
          URL.revokeObjectURL(resultUrl)
          return
        }
        if (showingTranslated) this.originalPageIds.delete(page.id)
        else this.originalPageIds.add(page.id)
      } catch (error) {
        URL.revokeObjectURL(resultUrl)
        if (candidate.element instanceof HTMLCanvasElement) {
          throw new ExtensionRequestError(
            'canvas_unreadable',
            'Canvas 受到跨域保护，无法替换或恢复像素',
            false,
          )
        }
        throw error
      }
      const previousUrl = this.resultUrls.get(page.id)
      this.resultUrls.set(page.id, resultUrl)
      if (previousUrl) URL.revokeObjectURL(previousUrl)
      this.appliedRetries.set(page.id, page.retryCount)
    }
  }

  private async retry(browserPageId: string): Promise<void> {
    const task = this.currentTask()
    if (!task) return
    try {
      const page = await send<BrowserPageDto>({
        type: 'retry-page',
        sessionId: task.sessionId,
        browserPageId,
      })
      if (!this.isCurrentTask(task)) return
      this.appliedRetries.delete(page.id)
      this.startPolling(250, task)
    } catch (error) {
      this.ui?.showError(errorDetails(error))
    }
  }

  private async toggleAllPages(): Promise<boolean> {
    try {
      const showingTranslated = await this.queueReplacement(
        this.currentTask(),
        () => this.replacement.toggleGlobal(),
      )
      if (showingTranslated === null) return true
      if (showingTranslated) {
        this.originalPageIds.clear()
      } else {
        for (const page of this.session?.pages ?? []) {
          if (page.state === 'completed' && page.resultReady) {
            this.originalPageIds.add(page.id)
          }
        }
      }
      if (this.session && !this.imported) this.showSession(this.session)
      return showingTranslated
    } catch (error) {
      this.ui?.showError(errorDetails(error))
      throw error
    }
  }

  private async togglePage(browserPageId: string): Promise<boolean | null> {
    const task = this.currentTask()
    const page = this.session?.pages.find(item => item.id === browserPageId)
    if (!task || !page || page.state !== 'completed' || !page.resultReady) return null
    const candidate = this.candidatesByClientKey.get(page.clientPageKey)
    if (!candidate) return null
    try {
      const showingTranslated = await this.queueReplacement(
        task,
        () => this.replacement.toggle(candidate),
      )
      if (!this.isCurrentTask(task) || showingTranslated === null) return null
      if (showingTranslated) this.originalPageIds.delete(browserPageId)
      else this.originalPageIds.add(browserPageId)
      if (this.session) this.showSession(this.session)
      return showingTranslated
    } catch (error) {
      if (this.isCurrentTask(task)) this.ui?.showError(errorDetails(error))
      throw error
    }
  }

  private async retryFailedUploads(): Promise<void> {
    const task = this.currentTask()
    if (
      !task
      || this.uploadFailures.size === 0
      || this.retryingUploadsFor === task.generation
    ) return
    this.retryingUploadsFor = task.generation
    const candidates = [...this.uploadFailures.values()].map(item => item.candidate)
    this.ui?.setStatus(
      '正在重试图片导入',
      `本次重试 ${candidates.length} 张尚未进入 Saber 的图片。`,
      'busy',
    )
    try {
      const result = await this.uploadCandidates(candidates, task, true)
      if (!this.isCurrentTask(task)) return
      if (result.uploaded === 0) {
        this.showUploadFailures()
        return
      }
      await this.startUploadedPages(task)
      this.startPolling(250, task)
    } catch (error) {
      if (this.isCurrentTask(task)) this.ui?.showError(errorDetails(error))
    } finally {
      if (this.retryingUploadsFor === task.generation) {
        this.retryingUploadsFor = null
      }
    }
  }

  private async updatePreference(preference: DomainPreference): Promise<void> {
    const methodChanged = preference.method !== this.preference.method
    this.activeMethod = preference.method
    if (methodChanged) {
      this.activeRule = null
      this.preference = { ...preference }
      delete this.preference.rule
    } else {
      this.preference = { ...preference }
      if (this.activeRule) this.preference.rule = this.activeRule
      else delete this.preference.rule
    }
    try {
      await this.persistPreference()
      if (methodChanged) this.ui?.setAdaptationSaved(false)
      const task = this.currentTask()
      if (task) {
        const session = await send<BrowserSessionDto>({
          type: 'patch-session',
          sessionId: task.sessionId,
          payload: {
            mode: this.preference.mode,
            glossaryEnabled: this.preference.glossaryEnabled,
            autoTermsEnabled: this.preference.autoTermsEnabled,
          },
        })
        if (!this.isCurrentTask(task)) return
        this.session = session
        this.showSession(this.session)
        this.startPolling(250, task)
      }
    } catch (error) {
      this.ui?.showError(errorDetails(error))
    }
  }

  private async updatePanelPreference(
    patch: {
      panelOpen?: boolean
      panelPosition?: PanelPosition
      fabPosition?: PanelPosition
    },
  ): Promise<void> {
    this.preference = { ...this.preference, ...patch }
    try {
      await this.persistPreference()
    } catch (error) {
      this.ui?.showError(errorDetails(error))
    }
  }

  private async persistPreference(): Promise<void> {
    await send<DomainPreference>({
      type: 'set-preference',
      hostname: this.hostname,
      preference: this.preference,
    })
  }

  private async deleteAdaptation(): Promise<void> {
    if (this.deletingAdaptation || !this.preference.rule) return
    this.deletingAdaptation = true
    const preference = { ...this.preference }
    delete preference.rule
    try {
      await send<DomainPreference>({
        type: 'set-preference',
        hostname: this.hostname,
        preference,
      })
      this.preference = preference
      this.activeRule = null
      this.usingAdapter = false
      this.candidates = []
      this.stopDiscovery()
      this.ui?.setAdaptationSaved(false)
      this.ui?.setStatus('已删除当前网站的适配', '正在按当前识别方式重新检测。', 'busy')
      await this.discover(this.preference.method)
    } catch (error) {
      this.ui?.showError(errorDetails(error))
    } finally {
      this.deletingAdaptation = false
    }
  }

  private stopDiscovery(): void {
    this.discoveryStopped = true
    this.disconnectObserver()
    this.ui?.setStatus('已停止继续发现', '已排队和正在处理的图片不会受到影响。')
  }

  private disconnectObserver(): void {
    this.observer?.disconnect()
    this.observer = null
    document.removeEventListener('load', this.imageLoadListener, true)
    if (this.observerTimer !== null) window.clearTimeout(this.observerTimer)
    this.observerTimer = null
  }

  private async cancel(): Promise<void> {
    const task = this.currentTask()
    if (!task) return
    this.cancelled = true
    this.stopDiscovery()
    try {
      const session = await send<BrowserSessionDto>({
        type: 'cancel-session',
        sessionId: task.sessionId,
      })
      if (!this.isCurrentTask(task)) return
      this.session = session
      this.showSession(this.session)
      await this.forgetActiveSession(session.id)
    } catch (error) {
      this.ui?.showError(errorDetails(error))
    }
  }

  private async loadLibraryBooks(): Promise<BrowserLibraryBook[]> {
    if (!this.currentTask()) return []
    try {
      const response = await send<{ items: BrowserLibraryBook[] }>({
        type: 'list-library-books',
      })
      return response.items
    } catch (error) {
      this.ui?.showError(errorDetails(error))
      throw error
    }
  }

  private async importToLibrary(
    command: BrowserSessionImportCommand,
  ): Promise<BrowserSessionImportResult> {
    const task = this.currentTask()
    if (!task) throw new Error('当前网页任务已经结束')
    this.stopDiscovery()
    try {
      const result = await send<BrowserSessionImportResult>({
        type: 'import-session',
        sessionId: task.sessionId,
        payload: command,
      })
      if (!this.isCurrentTask(task)) throw new Error('当前网页任务已经切换')
      this.imported = true
      this.taskGeneration += 1
      if (this.pollTimer !== null) window.clearTimeout(this.pollTimer)
      this.pollTimer = null
      await this.forgetActiveSession(task.sessionId)
      return result
    } catch (error) {
      this.ui?.showError(errorDetails(error))
      throw error
    }
  }

  private async disableSite(): Promise<void> {
    this.preference = { ...this.preference, disabled: true }
    try {
      await this.persistPreference()
    } catch (error) {
      this.preference = { ...this.preference, disabled: false }
      this.ui?.showError(errorDetails(error))
      return
    }
    await this.forgetActiveSession(this.session?.id)
    await this.dispose()
  }

  private async copyDiagnostics(): Promise<void> {
    const diagnostics = {
      extensionVersion: chrome.runtime.getManifest().version,
      pageUrl: this.pageUrl,
      hostname: this.hostname,
      detectionMethod: this.preference.method,
      candidateCount: this.candidatesByIdentity.size,
      session: this.session
        ? {
            id: this.session.id,
            state: this.session.state,
            counts: this.session.counts,
            errors: this.session.pages
              .filter(page => page.error)
              .map(page => ({ ordinal: page.ordinal, error: page.error })),
          }
        : null,
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(diagnostics, null, 2))
      this.ui?.setStatus('诊断信息已复制', '内容未自动上传，可直接发送给开发者。')
    } catch {
      this.ui?.showError({ code: 'clipboard_failed', message: '浏览器拒绝写入剪贴板' })
    }
  }
}

let controller: PageController | null = null
let activePageUrl = normalizedPageUrl()

async function startController(): Promise<void> {
  const next = new PageController()
  controller = next
  try {
    await next.initialize()
  } catch (error) {
    console.warn('Saber extension initialization failed', error)
  }
}

if (typeof chrome !== 'undefined' && chrome.runtime?.id) {
  chrome.runtime.onMessage.addListener((message: unknown) => {
    const candidate = message as Partial<ContextTranslateMessage>
    if (candidate?.type === 'context-translate-image' && typeof candidate.srcUrl === 'string') {
      void controller?.translateContextImage(candidate.srcUrl)
    }
  })

  void startController()

  window.setInterval(() => {
    const nextUrl = normalizedPageUrl()
    if (nextUrl === activePageUrl) return
    activePageUrl = nextUrl
    const previous = controller
    controller = null
    void previous?.dispose().finally(() => startController())
  }, 1_000)
}
