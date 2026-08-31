// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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
  initialize(): Promise<void>
  dispose(): Promise<void>
  createSession(): Promise<{ generation: number; sessionId: string } | null>
  uploadCandidates(
    candidates: ImageCandidate[],
    task: { generation: number; sessionId: string },
    reportProgress?: boolean,
  ): Promise<{ uploaded: number; failed: number }>
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

beforeEach(() => {
  document.documentElement.replaceChildren(document.createElement('body'))
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: vi.fn(),
  } as unknown as CanvasRenderingContext2D)
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(
    'data:image/png;base64,bW9jaw==',
  )
  sendMessage = vi.fn(async (request: { type: string }) => {
    if (request.type === 'get-preference') return successful(structuredClone(DEFAULT_PREFERENCE))
    throw new Error(`unexpected request: ${request.type}`)
  })
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

describe('page task lifecycle', () => {
  it('restores the active tab session and its completed image after reload', async () => {
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
        return successful({ sessionId: 'restored-session' })
      }
      if (request.type === 'get-session') return successful(restoredSession)
      if (request.type === 'fetch-result') {
        return successful({ base64: 'cmVzdG9yZWQ=', mimeType: 'image/png' })
      }
      throw new Error(`unexpected request: ${request.type}`)
    })

    const controller = new PageController() as unknown as TestController
    await controller.initialize()

    expect(controller.currentTask()).toEqual({
      generation: 1,
      sessionId: 'restored-session',
    })
    expect(controller.session?.id).toBe('restored-session')
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
      throw new Error(`unexpected request: ${request.type}`)
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
      throw new Error(`unexpected request: ${request.type}`)
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

    await expect(oldBatch).resolves.toEqual({ uploaded: 0, failed: 0 })
    await expect(newBatch).resolves.toEqual({ uploaded: 1, failed: 0 })
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
      throw new Error(`unexpected request: ${request.type}`)
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

    await expect(controller.uploadCandidates([candidate!], task!, true)).resolves.toEqual({
      uploaded: 0,
      failed: 1,
    })
    const host = document.querySelector<HTMLDivElement>('#saber-translator-extension-root')!
    const preparation = host.shadowRoot!.querySelector<HTMLElement>('.saber-preparation')!
    const meter = host.shadowRoot!.querySelector<HTMLProgressElement>(
      '.saber-preparation__meter',
    )!
    expect(preparation.hidden).toBe(false)
    expect(meter.max).toBe(1)
    expect(meter.value).toBe(1)
    expect(preparation.textContent).toContain('成功 0 · 失败 1')
    const retry = [...document.querySelectorAll('*')]
      .flatMap(element => element.shadowRoot
        ? [...element.shadowRoot.querySelectorAll<HTMLButtonElement>('button')]
        : [])
      .find(button => button.textContent === '重试上传')
    expect(retry?.hidden).toBe(false)

    uploadFails = false
    retry?.click()
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
      throw new Error(`unexpected request: ${request.type}`)
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
      if (request.type === 'create-session') return successful(session('import-session'))
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
      throw new Error(`unexpected request: ${request.type}`)
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
