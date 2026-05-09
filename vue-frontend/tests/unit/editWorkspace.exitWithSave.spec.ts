import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useSettingsStore } from '@/stores/settingsStore'

const { reRenderFullImageMock, showToastMock } = vi.hoisted(() => ({
  reRenderFullImageMock: vi.fn(),
  showToastMock: vi.fn(),
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
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

import EditWorkspace from '@/components/edit/EditWorkspace.vue'

let pinia: ReturnType<typeof createPinia>

const EditToolbarStub = defineComponent({
  name: 'EditToolbar',
  emits: ['exit-edit-mode'],
  setup(_props, { emit }) {
    return () =>
      h(
        'button',
        {
          class: 'exit-edit-mode-trigger',
          onClick: () => emit('exit-edit-mode'),
        },
        'exit-edit-mode'
      )
  },
})

describe('EditWorkspace exit with save', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)

    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2')

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(false)

    reRenderFullImageMock.mockReset()
    showToastMock.mockReset()
  })

  function mountWorkspace() {
    return mount(EditWorkspace, {
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
  }

  it('exits immediately outside bookshelf mode when the toolbar exit button is clicked', async () => {
    const wrapper = mountWorkspace()

    await wrapper.find('.exit-edit-mode-trigger').trigger('click')

    expect(wrapper.emitted('exit')).toHaveLength(1)
    expect(wrapper.text()).not.toContain('保存后退出')
  })

  it('exits immediately when bookshelf auto save is disabled', async () => {
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const saveChapterSessionSpy = vi.spyOn(sessionStore, 'saveChapterSession')
    const wrapper = mountWorkspace()

    await wrapper.find('.exit-edit-mode-trigger').trigger('click')

    expect(wrapper.emitted('exit')).toHaveLength(1)
    expect(saveChapterSessionSpy).not.toHaveBeenCalled()
    expect(wrapper.text()).not.toContain('保存后退出')
  })

  it('shows an exit confirmation dialog in bookshelf mode when auto save is enabled', async () => {
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(true)

    const wrapper = mountWorkspace()
    await wrapper.find('.exit-edit-mode-trigger').trigger('click')

    expect(wrapper.emitted('exit')).toBeUndefined()
    expect(wrapper.text()).toContain('直接退出')
    expect(wrapper.text()).toContain('保存后退出')
  })

  it('allows direct exit from the confirmation dialog without triggering a full chapter save', async () => {
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')
    const saveChapterSessionSpy = vi.spyOn(sessionStore, 'saveChapterSession')

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(true)

    const wrapper = mountWorkspace()
    await wrapper.find('.exit-edit-mode-trigger').trigger('click')
    await wrapper.find('[data-testid="exit-without-save-button"]').trigger('click')

    expect(saveChapterSessionSpy).not.toHaveBeenCalled()
    expect(wrapper.emitted('exit')).toHaveLength(1)
  })

  it('saves the chapter and exits only after save succeeds', async () => {
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')
    const saveChapterSessionSpy = vi
      .spyOn(sessionStore, 'saveChapterSession')
      .mockResolvedValue(true)

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(true)

    const wrapper = mountWorkspace()
    await wrapper.find('.exit-edit-mode-trigger').trigger('click')
    await wrapper.find('[data-testid="save-and-exit-button"]').trigger('click')
    await flushPromises()

    expect(saveChapterSessionSpy).toHaveBeenCalledWith('book-1', 'chapter-1')
    expect(wrapper.emitted('exit')).toHaveLength(1)
  })

  it('stays in edit mode and shows an error state when save-and-exit fails', async () => {
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')
    vi.spyOn(sessionStore, 'saveChapterSession').mockResolvedValue(false)

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(true)

    const wrapper = mountWorkspace()
    await wrapper.find('.exit-edit-mode-trigger').trigger('click')
    await wrapper.find('[data-testid="save-and-exit-button"]').trigger('click')
    await flushPromises()

    expect(wrapper.emitted('exit')).toBeUndefined()
    expect(wrapper.text()).toContain('保存失败')
    expect(wrapper.find('[data-testid="retry-save-and-exit-button"]').exists()).toBe(true)
  })

  it('keeps the Escape key as a direct exit path even when auto save is enabled', async () => {
    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')
    const saveChapterSessionSpy = vi.spyOn(sessionStore, 'saveChapterSession')

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(true)

    const wrapper = mountWorkspace()
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    await flushPromises()

    expect(saveChapterSessionSpy).not.toHaveBeenCalled()
    expect(wrapper.emitted('exit')).toHaveLength(1)
  })
})
