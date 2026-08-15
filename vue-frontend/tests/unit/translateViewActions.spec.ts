import { computed, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTranslateViewActions } from '@/views/useTranslateViewActions'

const mocks = vi.hoisted(() => ({
  clearChapterPages: vi.fn(),
  deletePage: vi.fn(),
  discardPageDocument: vi.fn(),
  flushPageDocument: vi.fn(),
  resetQuickWorkspace: vi.fn(),
  showToast: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  clearChapterPages: mocks.clearChapterPages,
  deletePage: mocks.deletePage,
  resetQuickWorkspace: mocks.resetQuickWorkspace,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  discardPageDocument: mocks.discardPageDocument,
  flushPageDocument: mocks.flushPageDocument,
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.showToast,
}))

function createOptions() {
  const image = { id: 'page-1', chapterId: 'chapter-1', fileName: 'page.png' }
  return {
    bubbleStore: {
      clearBubblesLocal: vi.fn(),
    },
    imageStore: {
      currentImage: image,
      currentImageIndex: 0,
      failedImageCount: 0,
      hasImages: true,
      images: [image],
      clearImages: vi.fn(),
    },
    settingsStore: {
      settings: {
        textStyle: {
          autoFontSize: false,
          fontSize: 18,
        },
      },
      updateTextStyle: vi.fn(),
    },
    translation: {
      translateCurrentImage: vi.fn(),
      translateAllImages: vi.fn(),
      executeHqTranslation: vi.fn(),
      executeProofreading: vi.fn(),
      removeTextOnly: vi.fn(),
      removeAllTexts: vi.fn(),
      translateSelectedImages: vi.fn(),
      removeTextSelection: vi.fn(),
      retryFailedImages: vi.fn(),
    },
    translateInit: {
      initializeBookChapterContext: vi.fn().mockResolvedValue(true),
      flushChapterWorkState: vi.fn().mockResolvedValue(true),
      isBookshelfMode: ref(true),
      switchImage: vi.fn(),
      goToPrevious: vi.fn(),
      goToNext: vi.fn(),
    },
    validateBeforeTranslation: vi.fn(() => true),
    currentImage: computed(() => image),
    hasImages: computed(() => true),
    hasFailedImages: computed(() => false),
    isEditMode: ref(false),
    confirmAction: vi.fn().mockResolvedValue(true),
  }
}

describe('useTranslateViewActions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.clearChapterPages.mockResolvedValue(1)
    mocks.deletePage.mockResolvedValue(undefined)
    mocks.discardPageDocument.mockReturnValue(true)
    mocks.flushPageDocument.mockResolvedValue(undefined)
    mocks.resetQuickWorkspace.mockResolvedValue({
      bookId: 'book-1',
      chapterId: 'chapter-1',
      title: '快速翻译',
    })
  })

  it('refreshes authoritative chapter metadata after upload', async () => {
    const options = createOptions()
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.handleUploadComplete()

    expect(options.translateInit.initializeBookChapterContext).toHaveBeenCalled()
    expect(options.translateInit.switchImage).not.toHaveBeenCalled()
  })

  it('uses backend deletion APIs after product confirmation', async () => {
    const options = createOptions()
    const browserConfirm = vi.spyOn(window, 'confirm')
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.deleteCurrentImage()
    await actions.clearAllImages()

    expect(options.confirmAction).toHaveBeenCalledWith({
      title: '删除当前图片',
      message: '确定要删除当前图片 (page.png) 吗？',
      confirmText: '删除',
      tone: 'danger',
    })
    expect(options.confirmAction).toHaveBeenCalledWith({
      title: '清空图片',
      message: '确定要从后端章节中删除所有图片和翻译结果吗？',
      confirmText: '清空',
      tone: 'danger',
    })
    expect(mocks.deletePage).toHaveBeenCalledWith('page-1')
    expect(mocks.discardPageDocument).toHaveBeenCalledWith('page-1')
    expect(options.imageStore.clearImages).toHaveBeenCalledTimes(2)
    expect(options.bubbleStore.clearBubblesLocal).toHaveBeenCalledTimes(2)
    expect(mocks.clearChapterPages).toHaveBeenCalledWith('chapter-1')
    expect(options.translateInit.initializeBookChapterContext).toHaveBeenCalledTimes(2)
    expect(browserConfirm).not.toHaveBeenCalled()
  })

  it('resets the fixed backend quick workspace when clearing quick translation', async () => {
    const options = createOptions()
    options.translateInit.isBookshelfMode.value = false
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.clearAllImages()

    expect(mocks.resetQuickWorkspace).toHaveBeenCalled()
  })

  it('does not delete a different page when the selection changes during confirmation', async () => {
    const options = createOptions()
    options.confirmAction.mockImplementation(async () => {
      options.imageStore.currentImage = {
        id: 'page-2',
        chapterId: 'chapter-1',
        fileName: 'page-2.png',
      }
      return true
    })
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.deleteCurrentImage()

    expect(mocks.deletePage).not.toHaveBeenCalled()
    expect(mocks.showToast).toHaveBeenCalledWith(
      '当前图片已切换，未执行删除',
      'warning',
    )
  })

  it('does not reinterpret an accepted deletion as failed when reloading fails', async () => {
    const options = createOptions()
    options.translateInit.initializeBookChapterContext.mockRejectedValue(
      new Error('reload failed'),
    )
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.deleteCurrentImage()

    expect(mocks.deletePage).toHaveBeenCalledWith('page-1')
    expect(mocks.showToast).toHaveBeenCalledWith('图片已删除', 'success')
    expect(options.imageStore.clearImages).toHaveBeenCalledOnce()
    expect(options.bubbleStore.clearBubblesLocal).toHaveBeenCalledOnce()
    expect(mocks.showToast).toHaveBeenCalledWith(
      '刷新后端章节失败：reload failed',
      'error',
    )
    expect(mocks.showToast).not.toHaveBeenCalledWith(
      expect.stringContaining('删除图片失败'),
      'error',
    )
  })

  it('reports a rejected deletion without clearing local document state', async () => {
    const options = createOptions()
    mocks.deletePage.mockRejectedValue(new Error('backend rejected'))
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.deleteCurrentImage()

    expect(mocks.discardPageDocument).not.toHaveBeenCalled()
    expect(options.translateInit.initializeBookChapterContext).not.toHaveBeenCalled()
    expect(mocks.showToast).toHaveBeenCalledWith(
      '删除图片失败：backend rejected',
      'error',
    )
  })

  it('flushes chapter settings and the page document before entering edit mode', async () => {
    const options = createOptions()
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.toggleEditMode()

    expect(options.translateInit.flushChapterWorkState).toHaveBeenCalled()
    expect(mocks.flushPageDocument).toHaveBeenCalledWith('page-1')
    expect(options.isEditMode.value).toBe(true)
  })

  it('does not enter edit mode without a current backend page', async () => {
    const options = createOptions()
    options.imageStore.currentImage = null as unknown as typeof options.imageStore.currentImage
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.toggleEditMode()

    expect(options.translateInit.flushChapterWorkState).not.toHaveBeenCalled()
    expect(mocks.flushPageDocument).not.toHaveBeenCalled()
    expect(options.isEditMode.value).toBe(false)
  })

  it('does not hijack shortcuts from form and editable controls', () => {
    const options = createOptions()
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )
    const select = document.createElement('select')
    const editable = document.createElement('div')
    const child = document.createElement('span')
    editable.setAttribute('contenteditable', 'plaintext-only')
    editable.append(child)

    for (const target of [select, child]) {
      const event = new KeyboardEvent('keydown', { altKey: true, key: 'ArrowRight' })
      Object.defineProperty(event, 'target', { value: target })
      actions.handleKeydown(event)
      expect(event.defaultPrevented).toBe(false)
    }
    expect(options.translateInit.goToNext).not.toHaveBeenCalled()
  })

  it('handles a targetless shortcut and respects the one-pixel font-size floor', () => {
    const options = createOptions()
    options.settingsStore.settings.textStyle.fontSize = 1
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )
    const event = new KeyboardEvent('keydown', { altKey: true, key: 'ArrowDown' })

    expect(() => actions.handleKeydown(event)).not.toThrow()
    expect(options.settingsStore.updateTextStyle).toHaveBeenCalledWith({ fontSize: 1 })
  })
})
