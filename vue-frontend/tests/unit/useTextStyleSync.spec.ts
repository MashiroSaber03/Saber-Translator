import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { createBubbleState } from '@/utils/bubbleFactory'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { addTestImage } from '../helpers/imageFixtures'
import { useSettingsStore } from '@/stores/settings'

const {
  createStyleJobMock,
  flushPageDocumentMock,
  getPageDocumentMock,
  getPageRenderStatusMock,
  queueMutationMock,
  registerPageDocumentMock,
  refreshTasksMock,
  showToastMock,
} = vi.hoisted(() => ({
  createStyleJobMock: vi.fn(),
  flushPageDocumentMock: vi.fn(),
  getPageDocumentMock: vi.fn(),
  getPageRenderStatusMock: vi.fn(),
  queueMutationMock: vi.fn(),
  registerPageDocumentMock: vi.fn(),
  refreshTasksMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: getPageDocumentMock,
  getPageRenderStatus: getPageRenderStatusMock,
}))

vi.mock('@/api/v2/translation', () => ({
  createChapterStyleApplyJob: createStyleJobMock,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  flushPageDocument: flushPageDocumentMock,
  queuePageDocumentMutation: queueMutationMock,
  registerPageDocument: registerPageDocumentMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => ({ refresh: refreshTasksMock }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

describe('useTextStyleSync backend ownership', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    createStyleJobMock.mockReset().mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-1'],
      status: 'queued',
    })
    flushPageDocumentMock.mockReset().mockResolvedValue(undefined)
    getPageDocumentMock.mockReset()
    getPageRenderStatusMock.mockReset()
    queueMutationMock.mockReset().mockResolvedValue(undefined)
    registerPageDocumentMock.mockReset().mockReturnValue([])
    refreshTasksMock.mockReset().mockResolvedValue(undefined)
    showToastMock.mockReset()
  })

  function mountPage() {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const bubble = createBubbleState({
      autoBgColor: [10, 11, 12],
      autoFgColor: [1, 2, 3],
      coords: [0, 0, 120, 80],
      fontFamily: 'font-id-1',
      fontSize: 22,
      polygon: [],
      translatedText: '第一页译文',
    })
    const image = addTestImage(imageStore, 'page-1.png', '/api/v2/assets/source-1', {
      bubbleStates: [bubble],
      chapterId: 'chapter-1',
      documentRevision: 4,
      fontFamily: 'font-id-1',
    })
    bubbleStore.setBubbles([bubble])
    return { bubbleStore, image, imageStore }
  }

  it('contains no browser render or Base64 pipeline', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useTextStyleSync.ts'),
      'utf8',
    )

    expect(source).not.toContain('executeRender')
    expect(source).not.toContain('buildEditRenderInput')
    expect(source).not.toContain('base64')
    expect(source).toContain('queuePageDocumentMutation')
    expect(source).toContain('createChapterStyleApplyJob')
  })

  it('submits one font domain command without projecting bubble styles in the browser', async () => {
    const { image } = mountPage()
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleTextStyleChanged('fontFamily', 'font-id-2')

    expect(queueMutationMock).toHaveBeenCalledWith(
      image.id,
      4,
      [expect.objectContaining({
        fontFamily: 'font-id-1',
      })],
      {
        defaultFontId: 'font-id-2',
        pageStyleDefaultsPatch: {},
        propagateStyleFields: ['fontFamily'],
      },
    )
  })

  it('delegates automatic color materialization to the backend CAS', async () => {
    const { image, imageStore } = mountPage()
    useSettingsStore().updateTextStyle({ useAutoTextColor: true })
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleAutoTextColorChanged(true)

    expect(imageStore.currentImage?.bubbleStates?.[0]).toMatchObject({
      fillColor: '#FFFFFF',
      textColor: '#000000',
    })
    expect(queueMutationMock).toHaveBeenCalledWith(
      image.id,
      4,
      [expect.objectContaining({
        fillColor: '#FFFFFF',
        textColor: '#000000',
      })],
      {
        pageStyleDefaultsPatch: { useAutoTextColor: true },
        propagateStyleFields: ['textColor', 'fillColor'],
      },
    )
  })

  it('submits apply-to-all as one durable backend job', async () => {
    const { image } = mountPage()
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleApplyToAll({
      fillColor: false,
      fontFamily: false,
      fontSize: true,
      layoutDirection: false,
      lineSpacing: false,
      strokeColor: false,
      strokeEnabled: false,
      strokeWidth: false,
      inlineAlign: true,
      blockAlign: false,
      textColor: false,
    })

    expect(createStyleJobMock).toHaveBeenCalledWith('chapter-1', {
      selectedFields: ['fontSize', 'inlineAlign'],
      sourceDocumentRevision: 4,
      sourcePageId: image.id,
    })
    expect(flushPageDocumentMock).toHaveBeenCalledWith(image.id)
    expect(refreshTasksMock).toHaveBeenCalled()
    expect(showToastMock).toHaveBeenCalledWith(
      '样式应用任务已加入后端任务中心，可安全关闭页面',
      'success',
    )
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('does not report an accepted style job as failed when task refresh fails', async () => {
    mountPage()
    refreshTasksMock.mockRejectedValue(new Error('snapshot unavailable'))
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleApplyToAll({
      fillColor: false,
      fontFamily: false,
      fontSize: true,
      layoutDirection: false,
      lineSpacing: false,
      strokeColor: false,
      strokeEnabled: false,
      strokeWidth: false,
      inlineAlign: false,
      blockAlign: false,
      textColor: false,
    })

    expect(createStyleJobMock).toHaveBeenCalledOnce()
    expect(showToastMock).toHaveBeenCalledWith(
      '样式应用任务已加入后端任务中心，可安全关闭页面',
      'success',
    )
    expect(showToastMock).not.toHaveBeenCalledWith(
      expect.stringContaining('创建样式应用任务失败'),
      'error',
    )
  })

  it('reloads the authoritative page document after a style CAS failure', async () => {
    const { bubbleStore, imageStore } = mountPage()
    const settingsStore = useSettingsStore()
    const authoritativeBubble = createBubbleState({
      backendBubbleId: 'bubble-1',
      fontFamily: settingsStore.settings.textStyle.fontFamily,
      fontSize: 19,
      translatedText: '后端版本',
    })
    const { fontFamily, ...pageStyleDefaults } = settingsStore.settings.textStyle
    queueMutationMock.mockRejectedValue(new Error('revision conflict'))
    getPageDocumentMock.mockResolvedValue({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: fontFamily,
      documentRevision: 8,
      pageId: imageStore.currentImage?.id,
      pageStyleDefaults: { ...pageStyleDefaults, fontSize: 19 },
      pageStyleSchemaVersion: 2,
      renderStatus: 'ready',
    })
    registerPageDocumentMock.mockReturnValue([authoritativeBubble])
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleTextStyleChanged('fontSize', 30)

    expect(getPageDocumentMock).toHaveBeenCalledWith(imageStore.currentImage?.id)
    expect(imageStore.currentImage).toMatchObject({
      documentRevision: 8,
      fontSize: 19,
      hasUnsavedChanges: false,
    })
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('后端版本')
    expect(settingsStore.settings.textStyle.fontSize).toBe(19)
    expect(showToastMock).toHaveBeenCalledWith(
      '文字样式写入后端失败，已恢复后端版本：revision conflict',
      'error',
    )
  })

  it('reports a terminal backend render failure', async () => {
    const { image, imageStore } = mountPage()
    imageStore.updateCurrentImage({ translatedAssetUrl: '/translated/old' })
    getPageRenderStatusMock.mockResolvedValue({
      pageId: image.id,
      documentRevision: 4,
      renderedRevision: 3,
      renderStatus: 'render_failed',
      translatedUrl: '/translated/old',
    })
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleTextStyleChanged('fontSize', 30)
    await vi.waitFor(() => expect(imageStore.currentImage?.translationStatus).toBe('failed'))

    expect(getPageRenderStatusMock).toHaveBeenCalledWith(image.id, expect.any(AbortSignal))
    expect(showToastMock).toHaveBeenCalledWith(
      '文字样式已保存，但后端重渲染失败',
      'error',
    )
  })

  it('stops preview polling after the bounded wait while backend rendering continues', async () => {
    vi.useFakeTimers()
    const { image, imageStore } = mountPage()
    imageStore.updateCurrentImage({ translatedAssetUrl: '/translated/old' })
    getPageRenderStatusMock.mockResolvedValue({
      pageId: image.id,
      documentRevision: 5,
      renderedRevision: 4,
      renderStatus: 'rendering',
      translatedUrl: '/translated/old',
    })
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleTextStyleChanged('fontSize', 30)
    await vi.advanceTimersByTimeAsync(30_500)

    expect(getPageRenderStatusMock.mock.calls.length).toBeGreaterThan(1)
    expect(getPageRenderStatusMock.mock.calls.length).toBeLessThanOrEqual(61)
    expect(showToastMock).toHaveBeenCalledWith(
      '文字样式已保存，后端仍在生成最新预览',
      'info',
    )
  })

  it('stops after finite render-status read retries and reports the real error', async () => {
    vi.useFakeTimers()
    const { imageStore } = mountPage()
    imageStore.updateCurrentImage({ translatedAssetUrl: '/translated/old' })
    getPageRenderStatusMock.mockRejectedValue(new Error('status endpoint unavailable'))
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')

    await useTextStyleSync().handleTextStyleChanged('fontSize', 30)
    await vi.advanceTimersByTimeAsync(2_500)

    expect(getPageRenderStatusMock).toHaveBeenCalledTimes(5)
    expect(showToastMock).toHaveBeenCalledWith(
      '文字样式已保存，但无法读取后端渲染状态：status endpoint unavailable',
      'error',
    )
  })

  it('keeps render refreshes for different pages independent', async () => {
    const { image, imageStore } = mountPage()
    imageStore.updateCurrentImage({ translatedAssetUrl: '/translated/page-1' })
    const second = addTestImage(imageStore, 'page-2.png', '/api/v2/assets/source-2', {
      bubbleStates: [],
      chapterId: 'chapter-1',
      documentRevision: 4,
      translatedAssetUrl: '/translated/page-2',
    })
    const signals: AbortSignal[] = []
    getPageRenderStatusMock.mockImplementation((_pageId: string, signal: AbortSignal) => {
      signals.push(signal)
      return new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'))
        }, { once: true })
      })
    })
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')
    const sync = useTextStyleSync()

    await sync.handleTextStyleChanged('fontSize', 29)
    imageStore.setCurrentImageIndex(1)
    await sync.handleTextStyleChanged('fontSize', 31)

    expect(getPageRenderStatusMock.mock.calls.map(call => call[0])).toEqual([
      image.id,
      second.id,
    ])
    expect(signals).toHaveLength(2)
    expect(signals[0]?.aborted).toBe(false)
    expect(signals[1]?.aborted).toBe(false)
  })

  it('ignores a late same-page render response after a newer style refresh starts', async () => {
    const { image, imageStore } = mountPage()
    imageStore.updateCurrentImage({ translatedAssetUrl: '/translated/original' })
    let resolveFirst!: (value: {
      pageId: string
      documentRevision: number
      renderedRevision: number
      renderStatus: string
      translatedUrl: string
    }) => void
    const signals: AbortSignal[] = []
    getPageRenderStatusMock
      .mockImplementationOnce((_pageId: string, signal: AbortSignal) => {
        signals.push(signal)
        return new Promise(resolve => {
          resolveFirst = resolve
        })
      })
      .mockImplementationOnce((_pageId: string, signal: AbortSignal) => {
        signals.push(signal)
        return Promise.resolve({
          pageId: image.id,
          documentRevision: 6,
          renderedRevision: 6,
          renderStatus: 'ready',
          translatedUrl: '/translated/latest',
        })
      })
    const { useTextStyleSync } = await import('@/composables/useTextStyleSync')
    const sync = useTextStyleSync()

    await sync.handleTextStyleChanged('fontSize', 29)
    await vi.waitFor(() => expect(getPageRenderStatusMock).toHaveBeenCalledTimes(1))
    await sync.handleTextStyleChanged('fontSize', 31)
    await vi.waitFor(() => {
      expect(imageStore.currentImage?.translatedAssetUrl).toBe('/translated/latest')
    })

    expect(signals[0]?.aborted).toBe(true)
    resolveFirst({
      pageId: image.id,
      documentRevision: 5,
      renderedRevision: 5,
      renderStatus: 'ready',
      translatedUrl: '/translated/stale',
    })
    await Promise.resolve()
    await Promise.resolve()

    expect(imageStore.currentImage).toMatchObject({
      renderedRevision: 6,
      translatedAssetUrl: '/translated/latest',
      translationStatus: 'completed',
    })
  })
})
