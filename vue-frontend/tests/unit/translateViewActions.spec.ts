import { computed, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTranslateViewActions } from '@/views/useTranslateViewActions'

const mocks = vi.hoisted(() => ({
  clearChapterPages: vi.fn(),
  deletePage: vi.fn(),
  flushPageDocument: vi.fn(),
  resetQuickWorkspace: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  clearChapterPages: mocks.clearChapterPages,
  deletePage: mocks.deletePage,
  resetQuickWorkspace: mocks.resetQuickWorkspace,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  flushPageDocument: mocks.flushPageDocument,
}))

function createOptions() {
  const image = { id: 'page-1', chapterId: 'chapter-1', fileName: 'page.png' }
  return {
    imageStore: {
      currentImage: image,
      currentImageIndex: 0,
      images: [image],
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
      initializeBookChapterContext: vi.fn().mockResolvedValue(undefined),
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
    mocks.flushPageDocument.mockResolvedValue(undefined)
  })

  it('refreshes authoritative chapter metadata after upload', async () => {
    const options = createOptions()
    const actions = useTranslateViewActions(
      options as unknown as Parameters<typeof useTranslateViewActions>[0],
    )

    await actions.handleUploadComplete(3)

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
})
