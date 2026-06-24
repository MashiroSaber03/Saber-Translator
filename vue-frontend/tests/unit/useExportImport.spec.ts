import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { useExportImport as useExportImportFn } from '@/composables/useExportImport'

const {
  cleanTempFilesMock,
  downloadFinalizeMock,
  downloadStartSessionMock,
  downloadUploadImageMock,
  executeRenderMock,
  getDownloadFileUrlMock,
  toastMock,
} = vi.hoisted(() => ({
  cleanTempFilesMock: vi.fn(),
  downloadFinalizeMock: vi.fn(),
  downloadStartSessionMock: vi.fn(),
  downloadUploadImageMock: vi.fn(),
  executeRenderMock: vi.fn(),
  getDownloadFileUrlMock: vi.fn(),
  toastMock: {
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
  },
}))

vi.mock('@/api/system', () => ({
  cleanTempFiles: cleanTempFilesMock,
  downloadFinalize: downloadFinalizeMock,
  downloadStartSession: downloadStartSessionMock,
  downloadUploadImage: downloadUploadImageMock,
  getDownloadFileUrl: getDownloadFileUrlMock,
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeRender: executeRenderMock,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => toastMock,
}))

describe('useExportImport', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    cleanTempFilesMock.mockReset()
    downloadFinalizeMock.mockReset()
    downloadStartSessionMock.mockReset()
    downloadUploadImageMock.mockReset()
    executeRenderMock.mockReset()
    getDownloadFileUrlMock.mockReset()
    toastMock.error.mockReset()
    toastMock.info.mockReset()
    toastMock.success.mockReset()
    toastMock.warning.mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
    document.body.innerHTML = ''
  })

  it('imports translated text and rerenders without routine console output', async () => {
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-import',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          originalText: '原文',
          translatedText: '新译文',
          textboxText: '新译文',
          textDirection: 'vertical',
        }),
      ],
    })

    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,original-image', {
      translatedDataURL: 'data:image/png;base64,existing-render',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          originalText: '原文',
          translatedText: '旧译文',
          textboxText: '旧译文',
          textDirection: 'vertical',
        }),
      ],
    })

    const file = new File([
      JSON.stringify([
        {
          imageIndex: 0,
          bubbles: [
            {
              bubbleIndex: 0,
              original: '原文',
              translated: '新译文',
              textDirection: 'vertical',
            },
          ],
        },
      ]),
    ], 'translations.json', { type: 'application/json' })

    const { useExportImport } = await import('@/composables/useExportImport')
    const { importText } = useExportImport()
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      await importText(file)
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(executeRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      cleanImage: 'original-image',
      translatedTexts: ['新译文'],
    }))
    expect(imageStore.currentImage?.translatedDataURL).toBe('data:image/png;base64,rendered-import')
    expect(toastMock.success).toHaveBeenCalledWith('导入成功！更新了 1 张图片中的 1 个气泡文本，重渲染了 1 张图片')
  })

  it('downloads all images and runs scheduled temp cleanup without routine console output', async () => {
    vi.useFakeTimers()
    downloadStartSessionMock.mockResolvedValue({ success: true, session_id: 'session-1' })
    downloadUploadImageMock.mockResolvedValue({ success: true })
    downloadFinalizeMock.mockResolvedValue({ success: true, file_id: 'file-1' })
    getDownloadFileUrlMock.mockReturnValue('/api/download_file/file-1?format=zip')
    cleanTempFilesMock.mockResolvedValue({ success: true })

    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,original-image')

    const { useExportImport } = await import('@/composables/useExportImport')
    const { downloadAllImages } = useExportImport()
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      await downloadAllImages('zip')
      await vi.advanceTimersByTimeAsync(60_000)
      expect(cleanTempFilesMock).toHaveBeenCalled()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
      clickSpy.mockRestore()
    }

    expect(downloadStartSessionMock).toHaveBeenCalledWith(1)
    expect(downloadUploadImageMock).toHaveBeenCalledWith(
      'session-1',
      'data:image/png;base64,original-image',
      0,
      'page-1.png',
    )
    expect(toastMock.success).toHaveBeenCalledWith('已成功处理 1 张图片（全部为原始图片），下载即将开始')
  })

  it('clears pending download progress reset timer when the owner unmounts', async () => {
    vi.useFakeTimers()
    let exportImport: ReturnType<typeof useExportImportFn> | null = null
    downloadStartSessionMock.mockResolvedValue({ success: true, session_id: 'session-1' })
    downloadUploadImageMock.mockResolvedValue({ success: true })
    downloadFinalizeMock.mockResolvedValue({ success: true, file_id: 'file-1' })
    getDownloadFileUrlMock.mockReturnValue('/api/download_file/file-1?format=zip')
    cleanTempFilesMock.mockResolvedValue({ success: true })
    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,original-image')

    const { useExportImport } = await import('@/composables/useExportImport')
    const Host = defineComponent({
      setup() {
        exportImport = useExportImport()
        return () => h('div')
      },
    })

    const wrapper = mount(Host)
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

    try {
      await exportImport?.downloadAllImages('zip')

      expect(vi.getTimerCount()).toBe(2)

      wrapper.unmount()

      expect(vi.getTimerCount()).toBe(1)
    } finally {
      clickSpy.mockRestore()
    }
  })

  it('revokes the current image object URL when the download click throws', async () => {
    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,aGVsbG8=')
    const createObjectUrlMock = vi.fn(() => 'blob:download-current')
    const revokeObjectUrlMock = vi.fn()
    const originalCreateObjectUrl = URL.createObjectURL
    const originalRevokeObjectUrl = URL.revokeObjectURL
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: createObjectUrlMock,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: revokeObjectUrlMock,
    })
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
      throw new Error('blocked download')
    })
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    try {
      const { useExportImport } = await import('@/composables/useExportImport')
      const { downloadCurrentImage } = useExportImport()

      downloadCurrentImage()

      expect(revokeObjectUrlMock).toHaveBeenCalledWith('blob:download-current')
      expect(document.body.querySelector('a')).toBeNull()
      expect(toastMock.error).toHaveBeenCalledWith('下载失败')
    } finally {
      errorSpy.mockRestore()
      clickSpy.mockRestore()
      Object.defineProperty(URL, 'createObjectURL', {
        configurable: true,
        value: originalCreateObjectUrl,
      })
      Object.defineProperty(URL, 'revokeObjectURL', {
        configurable: true,
        value: originalRevokeObjectUrl,
      })
    }
  })
})
