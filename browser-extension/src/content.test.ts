// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createHash } from 'node:crypto'
import type { ExtensionUi } from './ui'
import { PageController } from './content'
import { elementCandidate, type ImageCandidate } from './discovery'
import { DEFAULT_PREFERENCE } from './storage'
import type {
  BrowserPageDto,
  BrowserSessionDto,
  BrowserSessionImportCommand,
  BrowserSessionImportResult,
} from './types'

interface TestController {
  ui: ExtensionUi
  discoveryStopped: boolean
  observer: MutationObserver | null
  retryFailedUploads(): Promise<void>
  startUploadedPages(task: { generation: number; sessionId: string }): Promise<void>
  cancel(): Promise<void>
  translateContextImage(source: string): Promise<void>
  candidatesByClientKey: Map<string, ImageCandidate>
  applyCompletedPages(session: BrowserSessionDto, task: { generation: number; sessionId: string }): Promise<void>
  initialize(): Promise<void>
  dispose(): Promise<void>
  createSession(): Promise<{ generation: number; sessionId: string } | null>
  uploadCandidates(
    candidates: ImageCandidate[],
    task: { generation: number; sessionId: string },
    reportProgress?: boolean,
  ): Promise<number>
  poll(task: { generation: number; sessionId: string }): Promise<void>
  currentTask(): { generation: number; sessionId: string } | null
  importToLibrary(command: BrowserSessionImportCommand): Promise<BrowserSessionImportResult>
  scheduleLazyScan(): void
  discoverLazyImages(): Promise<void>
  session: BrowserSessionDto | null
}

function session(id: string, pages: BrowserPageDto[] = []): BrowserSessionDto {
  return {
    id,
    pageUrl: 'https://example.test/chapter',
    pageTitle: 'Example chapter',
    bookId: crypto.randomUUID(),
    chapterId: crypto.randomUUID(),
    mode: 'standard',
    glossaryEnabled: false,
    autoTermsEnabled: false,
    state: pages.length ? 'queued' : 'idle',
    pendingStart: false,
    expiresAt: new Date(Date.now() + 60_000).toISOString(),
    counts: {
      total: pages.length,
      queued: pages.length,
      translating: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
    },
    pages,
  }
}

function successful<T>(data: T): { ok: true; data: T } {
  return { ok: true, data }
}

function deferred<T>(): {
  promise: Promise<T>
  resolve(value: T): void
} {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

let sendMessage: ReturnType<typeof vi.fn>

function defaultResponse(request: { type: string; value?: string }) {
  if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
  if (request.type === 'get-active-session') return successful(null)
  if (request.type === 'set-active-session' || request.type === 'clear-active-session') return successful({})
  if (request.type === 'hash-source') return successful(createHash('sha256').update(request.value!).digest('hex'))
  throw new Error(`unexpected request: ${request.type}`)
}

beforeEach(() => {
  document.documentElement.replaceChildren(document.createElement('body'))
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: vi.fn(),
  } as unknown as CanvasRenderingContext2D)
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(
    'data:image/png;base64,bW9jaw==',
  )
  sendMessage = vi.fn(defaultResponse)
  vi.stubGlobal('chrome', {
    runtime: {
      id: '',
      sendMessage,
      getManifest: () => ({ version: '1.0.0' }),
    },
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('overlapping page operations', () => {
  it('does not restore an old session over a task started during initialization', async () => {
    const restored = deferred<ReturnType<typeof successful<BrowserSessionDto>>>()
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-active-session') return successful({
        sessionId: 'old', discovery: { stopped: true, usingAdapter: false, rule: null },
      })
      if (request.type === 'get-session') return restored.promise
      if (request.type === 'create-session') return successful(session('new'))
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    const initialization = controller.initialize()
    await vi.waitFor(() => expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'get-session' })))
    await controller.createSession()
    restored.resolve(successful({ ...session('old', [{ id: 'old-page' } as BrowserPageDto]), pageUrl: location.href }))
    await initialization
    expect(controller.session?.id).toBe('new')
    await controller.dispose()
  })

  it('does not create a panel when disposed while preferences are loading', async () => {
    const preference = deferred<ReturnType<typeof successful<typeof DEFAULT_PREFERENCE>>>()
    sendMessage.mockImplementation(() => preference.promise)
    const controller = new PageController() as unknown as TestController
    const initialization = controller.initialize()
    await controller.dispose()
    preference.resolve(successful(structuredClone(DEFAULT_PREFERENCE)))
    await initialization
    expect(document.getElementById('saber-translator-extension-root')).toBeNull()
  })

  it('does not clear a newer task when an older storage cleanup finishes late', async () => {
    const cleanup = deferred<ReturnType<typeof successful<object>>>()
    let clears = 0
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'clear-active-session' && ++clears === 1) return cleanup.promise
      if (request.type === 'create-session') return successful(session('new'))
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const old = controller.createSession()
    await controller.createSession()
    cleanup.resolve(successful({}))
    await old
    expect(controller.session?.id).toBe('new')
    await controller.dispose()
  })

  it('finishes displaying completed results before importing deletes the session', async () => {
    const applied = deferred<void>()
    const completed = { ...session('import-session'), state: 'completed' as const }
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'create-session' || request.type === 'get-session') return successful(completed)
      if (request.type === 'import-session') return successful({ bookId: 'book' })
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const task = (await controller.createSession())!
    const apply = vi.spyOn(controller, 'applyCompletedPages').mockImplementation(() => applied.promise)
    const polling = controller.poll(task)
    await vi.waitFor(() => expect(apply).toHaveBeenCalled())
    const importing = controller.importToLibrary({ destination: 'new', bookTitle: 'Book', chapterTitle: 'Chapter' })
    await Promise.resolve()
    expect(sendMessage).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'import-session' }))
    applied.resolve()
    await polling
    await expect(importing).resolves.toMatchObject({ bookId: 'book' })
    await controller.dispose()
  })
})

describe('page task lifecycle', () => {
  it.each([false, true])('restores completed images and the saved stopped=%s scope', async (stopped) => {
    const image = document.createElement('img')
    image.src = 'https://cdn.example.test/restored-page.webp'
    Object.defineProperties(image, {
      naturalWidth: { configurable: true, value: 1_200 },
      naturalHeight: { configurable: true, value: 1_800 },
    })
    document.body.append(image)
    const candidate = elementCandidate(image)!
    const digest = await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode(candidate.sourceIdentity),
    )
    const clientPageKey = [...new Uint8Array(digest)]
      .map(byte => byte.toString(16).padStart(2, '0'))
      .join('')
    class LoadableImage {
      decoding = ''
      onload: (() => void) | null = null
      set src(_value: string) {
        queueMicrotask(() => this.onload?.())
      }
    }
    vi.stubGlobal('Image', LoadableImage)
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:restored-result')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)

    const completedPage: BrowserPageDto = {
      id: 'restored-page',
      clientPageKey,
      ordinal: 1,
      pageId: crypto.randomUUID(),
      state: 'completed',
      resultReady: true,
      retryCount: 0,
      error: null,
    }
    const restoredSession = {
      ...session('restored-session', [completedPage]),
      pageUrl: location.href,
      state: 'completed' as const,
      counts: {
        total: 1,
        queued: 0,
        translating: 0,
        completed: 1,
        failed: 0,
        cancelled: 0,
      },
    }
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-preference') {
        return successful(structuredClone(DEFAULT_PREFERENCE))
      }
      if (request.type === 'get-active-session') {
        return successful({
          sessionId: 'restored-session',
          discovery: { stopped, usingAdapter: false, rule: null },
        })
      }
      if (request.type === 'get-session') return successful(restoredSession)
      if (request.type === 'fetch-result') {
        return successful({ base64: 'cmVzdG9yZWQ=', mimeType: 'image/png' })
      }
      return defaultResponse(request)
    })

    const controller = new PageController() as unknown as TestController
    await controller.initialize()

    expect(controller.currentTask()).toEqual({
      generation: 1,
      sessionId: 'restored-session',
    })
    expect(controller.session?.id).toBe('restored-session')
    expect(controller.discoveryStopped).toBe(stopped)
    expect(controller.observer === null).toBe(stopped)
    expect(image.src).toBe('blob:restored-result')
    expect(image.dataset.saberTranslated).toBe('true')
    expect(sendMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'create-session' })
    )
    await controller.dispose()
  })

  it('ignores a previous session response after a newer task has started', async () => {
    const firstResponse = deferred<{ ok: true; data: BrowserSessionDto }>()
    const secondResponse = deferred<{ ok: true; data: BrowserSessionDto }>()
    let createCount = 0
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
      if (request.type === 'create-session') {
        createCount += 1
        return await (createCount === 1 ? firstResponse.promise : secondResponse.promise)
      }
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()

    const first = controller.createSession()
    await vi.waitFor(() => expect(createCount).toBe(1))
    const second = controller.createSession()
    await vi.waitFor(() => expect(createCount).toBe(2))
    secondResponse.resolve(successful(session('new-session')))
    await expect(second).resolves.toEqual({ generation: 2, sessionId: 'new-session' })
    firstResponse.resolve(successful(session('old-session')))
    await expect(first).resolves.toBeNull()

    expect(controller.session?.id).toBe('new-session')
    await controller.dispose()
  })

  it('isolates in-flight uploads by task generation', async () => {
    const firstUpload = deferred<{ ok: true; data: BrowserPageDto }>()
    const secondUpload = deferred<{ ok: true; data: BrowserPageDto }>()
    let createCount = 0
    let uploadCount = 0
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
      if (request.type === 'create-session') {
        createCount += 1
        return successful(session(createCount === 1 ? 'old-session' : 'new-session'))
      }
      if (request.type === 'upload-page') {
        uploadCount += 1
        return await (uploadCount === 1 ? firstUpload.promise : secondUpload.promise)
      }
      return defaultResponse(request)
    })
    const image = document.createElement('img')
    image.src = 'https://cdn.example.test/page.webp'
    Object.defineProperties(image, {
      naturalWidth: { configurable: true, value: 1_200 },
      naturalHeight: { configurable: true, value: 1_800 },
    })
    document.body.append(image)
    const candidate = elementCandidate(image)!
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const oldTask = await controller.createSession()
    const oldBatch = controller.uploadCandidates([candidate], oldTask!)
    await vi.waitFor(() => expect(uploadCount).toBe(1))

    const newTask = await controller.createSession()
    const newBatch = controller.uploadCandidates([candidate], newTask!)
    await vi.waitFor(() => expect(uploadCount).toBe(2))
    firstUpload.resolve(successful({
      id: 'old-page',
      clientPageKey: 'client-key',
      ordinal: 1,
      pageId: null,
      state: 'queued',
      resultReady: false,
      retryCount: 0,
      error: null,
    }))
    secondUpload.resolve(successful({
      id: 'new-page',
      clientPageKey: 'client-key',
      ordinal: 1,
      pageId: null,
      state: 'queued',
      resultReady: false,
      retryCount: 0,
      error: null,
    }))

    await expect(oldBatch).resolves.toBe(0)
    await expect(newBatch).resolves.toBe(1)
    expect(controller.session?.id).toBe('new-session')
    await controller.dispose()
  })

  it('keeps failed uploads visible and retries them explicitly', async () => {
    let uploadFails = true
    const page: BrowserPageDto = {
      id: crypto.randomUUID(),
      clientPageKey: 'client-page',
      ordinal: 1,
      pageId: null,
      state: 'queued',
      resultReady: false,
      retryCount: 0,
      error: null,
    }
    const activeSession = session('active-session', [page])
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
      if (request.type === 'create-session') return successful(session('active-session'))
      if (request.type === 'upload-page') {
        return uploadFails
          ? {
              ok: false,
              error: { code: 'source_timeout', message: '图片下载超时', retryable: true },
            }
          : successful(page)
      }
      if (request.type === 'start-session' || request.type === 'get-session') {
        return successful(activeSession)
      }
      if (request.type === 'get-terms') return successful({ glossary: { entries: [] } })
      return defaultResponse(request)
    })
    const image = document.createElement('img')
    image.src = 'https://cdn.example.test/page.webp'
    Object.defineProperties(image, {
      naturalWidth: { configurable: true, value: 1_200 },
      naturalHeight: { configurable: true, value: 1_800 },
    })
    document.body.append(image)
    const candidate = elementCandidate(image)
    expect(candidate).not.toBeNull()
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const task = await controller.createSession()
    expect(task).not.toBeNull()

    await expect(controller.uploadCandidates([candidate!], task!, true)).resolves.toBe(0)
    const preparation = controller.ui.shadow.querySelector<HTMLElement>('.saber-preparation')!
    const meter = controller.ui.shadow.querySelector<HTMLProgressElement>(
      '.saber-preparation__meter',
    )!
    expect(preparation.hidden).toBe(false)
    expect(meter.max).toBe(1)
    expect(meter.value).toBe(1)
    expect(preparation.textContent).toContain('成功 0 · 失败 1')
    const retry = [...controller.ui.shadow.querySelectorAll<HTMLButtonElement>('button')]
      .find(button => button.textContent === '重试上传')
    expect(retry?.hidden).toBe(false)

    uploadFails = false
    await controller.retryFailedUploads()
    expect(controller.observer).not.toBeNull()
    await vi.waitFor(() => {
      expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'start-session' }))
      expect(retry?.hidden).toBe(true)
      expect(preparation.hidden).toBe(true)
    })
    await controller.dispose()
  })

  it('does not restart a cancelled session with an unmaterialized page', async () => {
    const cancelledPage: BrowserPageDto = {
      id: 'cancelled-page',
      clientPageKey: 'cancelled-key',
      ordinal: 1,
      pageId: null,
      state: 'cancelled',
      resultReady: false,
      retryCount: 0,
      error: null,
    }
    const cancelledSession = {
      ...session('cancelled-session', [cancelledPage]),
      state: 'cancelled' as const,
      counts: {
        total: 1,
        queued: 0,
        translating: 0,
        completed: 0,
        failed: 0,
        cancelled: 1,
      },
    }
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
      if (request.type === 'create-session' || request.type === 'get-session') {
        return successful(cancelledSession)
      }
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const task = await controller.createSession()

    await controller.poll(task!)

    expect(sendMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'start-session' }),
    )
    await controller.dispose()
  })

  it('ends the backend task generation after a successful library import', async () => {
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
      if (request.type === 'create-session' || request.type === 'get-session') return successful(session('import-session'))
      if (request.type === 'import-session') {
        return successful({
          destination: 'new',
          bookId: 'book',
          bookTitle: 'Book',
          chapterId: 'chapter',
          chapterTitle: 'Chapter',
          importedPages: 1,
          omittedPages: 0,
          termsAdded: 0,
        })
      }
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    await controller.createSession()

    await expect(controller.importToLibrary({
      destination: 'new',
      bookTitle: 'Book',
      chapterTitle: 'Chapter',
    })).resolves.toEqual(expect.objectContaining({ bookId: 'book' }))

    expect(controller.currentTask()).toBeNull()
    await controller.dispose()
  })

  it('offers start retry after a configuration error and resumes an already materialized page', async () => {
    const page: BrowserPageDto = {
      id: 'page', clientPageKey: 'key', ordinal: 1, pageId: 'materialized-page',
      state: 'queued', resultReady: false, retryCount: 0, error: null,
    }
    const pending = { ...session('pending', [page]), pendingStart: true }
    let broken = true
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'create-session' || request.type === 'get-session') return successful(pending)
      if (request.type === 'start-session') {
        return broken ? { ok: false, error: { code: 'validation_error', message: '缺少模型', retryable: false } }
          : successful({ ...pending, pendingStart: false })
      }
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const task = (await controller.createSession())!
    await controller.poll(task)
    const retry = [...controller.ui.shadow.querySelectorAll('button')].find(button => button.textContent === '重试启动')!
    expect(retry.hidden).toBe(false)
    expect(sendMessage).toHaveBeenCalledWith({ type: 'start-session', sessionId: 'pending' })
    broken = false
    await controller.startUploadedPages(task)
    expect(retry.hidden).toBe(true)
    expect(controller.observer).not.toBeNull()
    await controller.dispose()
  })

  it('ignores upload failures arriving after cancellation and ends preparation', async () => {
    const upload = deferred<{ ok: false; error: { code: string; message: string; retryable: boolean } }>()
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'create-session') return successful(session('cancel-upload'))
      if (request.type === 'upload-page') return upload.promise
      if (request.type === 'cancel-session' || request.type === 'get-session') {
        return successful({ ...session('cancel-upload'), state: 'cancelled' })
      }
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const task = (await controller.createSession())!
    const image = document.createElement('img')
    image.src = 'data:image/png;base64,bW9jaw=='
    document.body.append(image)
    const pending = controller.uploadCandidates([elementCandidate(image)!], task, true)
    await vi.waitFor(() => expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'upload-page' })))
    await controller.cancel()
    upload.resolve({ ok: false, error: { code: 'session_conflict', message: 'cancelled', retryable: true } })
    expect(await pending).toBe(0)
    expect(controller.ui.shadow.querySelector<HTMLElement>('.saber-preparation')!.hidden).toBe(true)
    expect(controller.ui.shadow.textContent).not.toContain('张图片尚未导入')
    await controller.dispose()
  })

  it('can start a new task immediately after cancelling in-flight uploads', async () => {
    const upload = deferred<{ ok: false; error: { code: string; message: string; retryable: boolean } }>()
    let creates = 0
    let uploads = 0
    sendMessage.mockImplementation(async (request: { type: string }) => {
      if (request.type === 'create-session') return successful(session(`task-${++creates}`))
      if (request.type === 'upload-page') {
        uploads += 1
        if (uploads === 1) return upload.promise
        return successful({ id: 'new-page' })
      }
      if (request.type === 'cancel-session') return successful({ ...session('task-1'), state: 'cancelled' })
      if (request.type === 'start-session' || request.type === 'get-session') return successful(session(`task-${creates}`))
      return defaultResponse(request)
    })
    const image = document.createElement('img')
    image.src = 'data:image/png;base64,bW9jaw=='
    document.body.append(image)
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const previous = controller.translateContextImage(image.src)
    await vi.waitFor(() => expect(uploads).toBe(1))
    await controller.cancel()
    await controller.translateContextImage(image.src)
    expect(creates).toBe(2)
    expect(uploads).toBe(2)
    upload.resolve({ ok: false, error: { code: 'session_conflict', message: 'cancelled', retryable: true } })
    await previous
    expect(controller.session?.id).toBe('task-2')
    await controller.dispose()
  })

  it('displays later completed pages when one result cannot be downloaded', async () => {
    class LoadableImage {
      onload: (() => void) | null = null
      set src(_value: string) { queueMicrotask(() => this.onload?.()) }
    }
    vi.stubGlobal('Image', LoadableImage)
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:good-result')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    sendMessage.mockImplementation(async (request: { type: string; browserPageId?: string }) => {
      if (request.type === 'create-session') return successful(session('results'))
      if (request.type === 'fetch-result') return request.browserPageId === 'bad'
        ? { ok: false, error: { code: 'result_too_large', message: 'too large', retryable: false } }
        : successful({ base64: 'aW1hZ2U=', mimeType: 'image/png' })
      return defaultResponse(request)
    })
    const controller = new PageController() as unknown as TestController
    await controller.initialize()
    const task = (await controller.createSession())!
    const pages = ['bad', 'good'].map(id => {
      const image = document.createElement('img')
      image.src = `https://comic.example/${id}.png`
      image.id = id
      document.body.append(image)
      controller.candidatesByClientKey.set(id, elementCandidate(image)!)
      return { id, clientPageKey: id, ordinal: id === 'bad' ? 1 : 2, pageId: id,
        state: 'completed' as const, resultReady: true, retryCount: 0, error: null }
    })
    await expect(controller.applyCompletedPages(session('results', pages), task)).rejects.toThrow('too large')
    expect(document.getElementById('good')!.dataset.saberTranslated).toBe('true')
    expect(document.getElementById('bad')!.dataset.saberTranslated).toBeUndefined()
    await controller.dispose()
  })

  it('throttles continuous DOM changes without postponing discovery forever', async () => {
    vi.useFakeTimers()
    try {
      const controller = new PageController() as unknown as TestController
      await controller.initialize()
      const discover = vi.spyOn(controller, 'discoverLazyImages').mockResolvedValue()

      controller.scheduleLazyScan()
      await vi.advanceTimersByTimeAsync(400)
      controller.scheduleLazyScan()
      await vi.advanceTimersByTimeAsync(100)

      expect(discover).toHaveBeenCalledOnce()
      await controller.dispose()
    } finally {
      vi.useRealTimers()
    }
  })
})
