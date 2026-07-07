import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, h, ref } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const {
  reRenderFullImageMock,
  showToastMock,
  isBookshelfSessionInitializedMock,
  forceInitializeBookshelfSessionMock,
  saveBookshelfPageProgressMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  reRenderFullImageMock: vi.fn(),
  showToastMock: vi.fn(),
  isBookshelfSessionInitializedMock: vi.fn(),
  forceInitializeBookshelfSessionMock: vi.fn(),
  saveBookshelfPageProgressMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/composables/useImageViewer', () => ({
  useImageViewer: () => ({
    scale: ref(1),
    translateX: ref(0),
    translateY: ref(0),
    zoomAt: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    resetZoom: vi.fn(),
    startDrag: vi.fn(),
    drag: vi.fn(),
    endDrag: vi.fn(),
    getTransform: vi.fn(() => ({ scale: 1, translateX: 0, translateY: 0 })),
    setTransform: vi.fn(),
  }),
}))

vi.mock('@/composables/useEditRender', () => ({
  useEditRender: () => ({
    reRenderFullImage: reRenderFullImageMock,
  }),
}))

vi.mock('@/composables/useBubbleActions', () => ({
  useBubbleActions: () => ({
    isDrawingMode: ref(false),
    isDrawingBox: ref(false),
    currentDrawingRect: ref(null),
    isMiddleButtonDown: ref(false),
    handleBubbleSelect: vi.fn(),
    handleBubbleMultiSelect: vi.fn(),
    handleClearMultiSelect: vi.fn(),
    handleBubbleDragStart: vi.fn(),
    handleBubbleDragEnd: vi.fn(),
    handleBubbleResizeStart: vi.fn(),
    handleBubbleResizeEnd: vi.fn(),
    handleBubbleRotateStart: vi.fn(),
    handleBubbleRotateEnd: vi.fn(),
    toggleDrawingMode: vi.fn(),
    handleDrawBubble: vi.fn(),
    getDrawingRectStyle: vi.fn(() => ({})),
    handleBubbleUpdate: vi.fn(),
    deleteSelectedBubbles: vi.fn(),
    repairSelectedBubble: vi.fn(),
    handleOcrRecognize: vi.fn(),
  }),
}))

vi.mock('@/composables/useBrush', () => ({
  useBrush: () => ({
    brushMode: ref(null),
    brushSize: ref(20),
    mouseX: ref(0),
    mouseY: ref(0),
    isBrushKeyDown: ref(false),
    toggleBrushMode: vi.fn(),
    exitBrushMode: vi.fn(),
    startBrushPainting: vi.fn(),
    continueBrushPainting: vi.fn(),
    finishBrushPainting: vi.fn(),
    adjustBrushSize: vi.fn(),
  }),
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    translateWithCurrentBubbles: vi.fn(),
  }),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeDetection: vi.fn(),
  saveDetectionResultToImage: vi.fn(),
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

vi.mock('@/composables/translation/core/saveStep', () => ({
  isBookshelfSessionInitialized: isBookshelfSessionInitializedMock,
  forceInitializeBookshelfSession: forceInitializeBookshelfSessionMock,
  saveBookshelfPageProgress: saveBookshelfPageProgressMock,
}))

import EditWorkspace from '@/components/edit/EditWorkspace.vue'

enableAutoUnmount(afterEach)

let pinia: ReturnType<typeof createPinia>

const EditToolbarStub = defineComponent({
  name: 'EditToolbar',
  emits: ['apply-and-next'],
  setup(_props, { emit }) {
    return () =>
      h(
        'button',
        {
          class: 'apply-and-next-trigger',
          onClick: () => emit('apply-and-next'),
        },
        'apply-and-next'
      )
  },
})

describe('EditWorkspace applyAndNext', () => {
  it('keeps the edit workspace composable free of scaffold-style section narration', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/useEditWorkspace.ts'),
      'utf8',
    )

    expect(source).not.toContain('// ============================================================')
    expect(source).not.toContain('状态定义')
    expect(source).not.toContain('生命周期钩子')
    expect(source).not.toContain('其他方法')
    expect(source).toContain("import { deepClone } from '@/utils/deepClone'")
    expect(source).not.toContain('JSON.parse(JSON.stringify')

    for (const patchArtifact of [
      '\nfunction prepareForNavigation',
      '\nonErrorCaptured',
      '/**',
      'selectBubbleNew()',
      'watch(currentImageIndex) 会自动触发',
      '【业务契约 DualImageViewer】',
      '留5%边距',
    ]) {
      expect(source).not.toContain(patchArtifact)
    }
  })

  it('maps workspace shell colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditWorkspace.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).toContain('--edit-workspace-shell-background: var(--color-surface-inverse)')
  })

  it('keeps workspace root state classes under the edit-workspace owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditWorkspace.vue'),
      'utf8',
    )

    expect(source).toContain('`edit-workspace--layout-${layoutMode}`')
    expect(source).toContain("'edit-workspace--drawing-mode': isDrawingMode")
    expect(source).toContain("'edit-workspace--brush-mode-active': !!brushMode")
    expect(source).not.toContain('`layout-${layoutMode}`')
    expect(source).not.toContain("'drawing-mode': isDrawingMode")
    expect(source).not.toContain("'brush-mode-active': !!brushMode")
  })

  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)

    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2')

    reRenderFullImageMock.mockReset()
    showToastMock.mockReset()
    isBookshelfSessionInitializedMock.mockReset()
    forceInitializeBookshelfSessionMock.mockReset()
    saveBookshelfPageProgressMock.mockReset()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)

    vi.stubGlobal('confirm', vi.fn(() => true))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('does not navigate to the next image when re-rendering fails', async () => {
    reRenderFullImageMock.mockResolvedValue(false)

    const imageStore = useImageStore()
    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(reRenderFullImageMock).toHaveBeenCalledTimes(1)
    expect(goToNextSpy).not.toHaveBeenCalled()
    expect(saveBookshelfPageProgressMock).not.toHaveBeenCalled()
  })

  it('loads and applies existing bubbles without routine console logs', async () => {
    reRenderFullImageMock.mockResolvedValue(true)

    const imageStore = useImageStore()
    imageStore.updateCurrentImage({
      bubbleStates: [
        createBubbleState({
          coords: [10, 20, 110, 120],
          polygon: [],
          originalText: '原文',
          translatedText: '译文',
        }),
      ],
    })

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      const wrapper = mount(EditWorkspace, {
        props: {
          isEditModeActive: true,
        },
        global: {
          plugins: [pinia],
          stubs: {
            EditToolbar: EditToolbarStub,
            EditThumbnailPanel: true,
            BubbleOverlay: true,
            BubbleEditor: true,
          },
        },
      })

      await wrapper.find('.apply-and-next-trigger').trigger('click')
      await flushPromises()

      expect(logSpy).not.toHaveBeenCalled()
      expect(reRenderFullImageMock).toHaveBeenCalledTimes(1)
    } finally {
      logSpy.mockRestore()
    }
  })

  it('keeps the original behavior outside bookshelf mode', async () => {
    reRenderFullImageMock.mockResolvedValue(true)

    const imageStore = useImageStore()
    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(goToNextSpy).toHaveBeenCalledTimes(1)
    expect(isBookshelfSessionInitializedMock).not.toHaveBeenCalled()
    expect(forceInitializeBookshelfSessionMock).not.toHaveBeenCalled()
    expect(saveBookshelfPageProgressMock).not.toHaveBeenCalled()
  })

  it('persists the current page before navigating in bookshelf mode when the session is already initialized', async () => {
    reRenderFullImageMock.mockResolvedValue(true)
    isBookshelfSessionInitializedMock.mockResolvedValue(true)
    saveBookshelfPageProgressMock.mockResolvedValue(undefined)

    const imageStore = useImageStore()
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(isBookshelfSessionInitializedMock).toHaveBeenCalledTimes(1)
    expect(forceInitializeBookshelfSessionMock).not.toHaveBeenCalled()
    expect(saveBookshelfPageProgressMock).toHaveBeenCalledWith(0, 1)
    expect(goToNextSpy).toHaveBeenCalledTimes(1)
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
  })

  it('does not navigate when the uninitialized bookshelf session initialization is cancelled', async () => {
    reRenderFullImageMock.mockResolvedValue(true)
    isBookshelfSessionInitializedMock.mockResolvedValue(false)
    confirmProductActionMock.mockResolvedValueOnce(false)

    const imageStore = useImageStore()
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '初始化章节存档',
      message: '当前章节尚未初始化存档。首次使用“应用并下一张”需要先保存整章原图和基础元数据，是否继续？',
      confirmText: '继续',
      cancelText: '取消',
    })
    expect(window.confirm).not.toHaveBeenCalled()
    expect(forceInitializeBookshelfSessionMock).not.toHaveBeenCalled()
    expect(saveBookshelfPageProgressMock).not.toHaveBeenCalled()
    expect(goToNextSpy).not.toHaveBeenCalled()
  })

  it('initializes the bookshelf session on demand before saving the current page', async () => {
    reRenderFullImageMock.mockResolvedValue(true)
    isBookshelfSessionInitializedMock.mockResolvedValue(false)
    forceInitializeBookshelfSessionMock.mockResolvedValue(true)
    saveBookshelfPageProgressMock.mockResolvedValue(undefined)

    const imageStore = useImageStore()
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '初始化章节存档',
      message: '当前章节尚未初始化存档。首次使用“应用并下一张”需要先保存整章原图和基础元数据，是否继续？',
      confirmText: '继续',
      cancelText: '取消',
    })
    expect(window.confirm).not.toHaveBeenCalled()
    expect(forceInitializeBookshelfSessionMock).toHaveBeenCalledTimes(1)
    expect(saveBookshelfPageProgressMock).toHaveBeenCalledWith(0, 1)
    expect(goToNextSpy).toHaveBeenCalledTimes(1)
  })

  it('does not navigate when bookshelf session initialization fails', async () => {
    reRenderFullImageMock.mockResolvedValue(true)
    isBookshelfSessionInitializedMock.mockResolvedValue(false)
    forceInitializeBookshelfSessionMock.mockResolvedValue(false)

    const imageStore = useImageStore()
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(saveBookshelfPageProgressMock).not.toHaveBeenCalled()
    expect(goToNextSpy).not.toHaveBeenCalled()
  })

  it('does not navigate when persisting the current bookshelf page fails', async () => {
    reRenderFullImageMock.mockResolvedValue(true)
    isBookshelfSessionInitializedMock.mockResolvedValue(true)
    const saveError = new Error('保存失败')
    saveBookshelfPageProgressMock.mockRejectedValue(saveError)

    const imageStore = useImageStore()
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(goToNextSpy).not.toHaveBeenCalled()
    expect(showToastMock).toHaveBeenCalledWith('保存失败', 'error')
  })

  it('persists the last bookshelf page without navigating away', async () => {
    reRenderFullImageMock.mockResolvedValue(true)
    isBookshelfSessionInitializedMock.mockResolvedValue(true)
    saveBookshelfPageProgressMock.mockResolvedValue(undefined)

    const imageStore = useImageStore()
    imageStore.setCurrentImageIndex(1)

    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(saveBookshelfPageProgressMock).toHaveBeenCalledWith(1, 1)
    expect(goToNextSpy).not.toHaveBeenCalled()
  })
})
