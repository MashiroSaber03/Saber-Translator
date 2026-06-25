import { enableAutoUnmount, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import TranslateView from '@/views/TranslateView.vue'

const {
  routeState,
  initializeAppMock,
  initializeBookChapterContextMock,
  initValidationMock,
  handleKeydownMock,
  saveCurrentSessionMock,
} = vi.hoisted(() => ({
  routeState: { query: {} as Record<string, string | undefined> },
  initializeAppMock: vi.fn(),
  initializeBookChapterContextMock: vi.fn(),
  initValidationMock: vi.fn(),
  handleKeydownMock: vi.fn(),
  saveCurrentSessionMock: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
}))

vi.mock('@/composables/useValidation', () => ({
  useValidation: () => ({
    validateBeforeTranslation: vi.fn(() => true),
    initValidation: initValidationMock,
  }),
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    progress: { value: {} },
    translateCurrentImage: vi.fn(),
    translateAllImages: vi.fn(),
    executeHqTranslation: vi.fn(),
    executeProofreading: vi.fn(),
    removeTextOnly: vi.fn(),
    removeAllTexts: vi.fn(),
    translateSelectedImages: vi.fn(),
    removeTextSelection: vi.fn(),
    retryFailedImages: vi.fn(),
  }),
}))

vi.mock('@/composables/useTranslateInit', () => ({
  useTranslateInit: () => ({
    currentBookTitle: { value: 'Book' },
    currentChapterTitle: { value: 'Chapter' },
    initializeApp: initializeAppMock,
    initializeBookChapterContext: initializeBookChapterContextMock,
  }),
}))

vi.mock('@/composables/useTextStyleSync', () => ({
  useTextStyleSync: () => ({
    handleTextStyleChanged: vi.fn(),
    handleAutoFontSizeChanged: vi.fn(),
    handleAutoTextColorChanged: vi.fn(),
    handleApplyToAll: vi.fn(),
  }),
}))

vi.mock('@/views/useTranslateViewActions', () => ({
  useTranslateViewActions: () => ({
    goToNext: vi.fn(),
    goToPrevious: vi.fn(),
    handleKeydown: handleKeydownMock,
    handleRetryFailed: vi.fn(),
    handleRunWorkflow: vi.fn(),
    handleUploadComplete: vi.fn(),
    loadChapterSession: vi.fn(),
    saveCurrentSession: saveCurrentSessionMock,
    selectImage: vi.fn(),
    toggleEditMode: vi.fn(),
  }),
}))

const AppShellStub = defineComponent({
  template: '<div class="app-shell-stub"><slot /></div>',
})

const AppHeaderStub = defineComponent({
  template: '<header class="app-header-stub"><slot name="header-links" /></header>',
})

const SidebarLayoutStub = defineComponent({
  template: '<div class="sidebar-layout-stub"><slot name="left" /><slot /><slot name="right" /></div>',
})

const RouterLinkStub = defineComponent({
  props: ['to'],
  template: '<a :href="to"><slot /></a>',
})

function mountTranslateView() {
  setActivePinia(createPinia())
  return mount(TranslateView, {
    global: {
      stubs: {
        AppShell: AppShellStub,
        AppHeader: AppHeaderStub,
        SidebarLayout: SidebarLayoutStub,
        RouterLink: RouterLinkStub,
        ImageUpload: true,
        SettingsSidebar: true,
        ImageResultDisplay: true,
        FirstTimeGuide: true,
        TranslationProgress: true,
        SponsorModal: true,
        ThumbnailSidebar: true,
        SettingsModal: true,
        BookGlossaryModal: true,
        BookNonTranslateModal: true,
        EditWorkspace: true,
        ProgressBar: true,
        WebImportModal: true,
        WebImportDisclaimer: true,
      },
    },
  })
}

describe('TranslateView', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    routeState.query = {}
    initializeAppMock.mockReset()
    initializeBookChapterContextMock.mockReset()
    initValidationMock.mockReset()
    handleKeydownMock.mockReset()
    saveCurrentSessionMock.mockReset()
    initializeAppMock.mockResolvedValue(undefined)
    initializeBookChapterContextMock.mockResolvedValue(undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('registers global keyboard handling before async initialization settles', () => {
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')
    initializeAppMock.mockReturnValue(new Promise(() => {}))

    const wrapper = mountTranslateView()

    expect(addEventListenerSpy).toHaveBeenCalledWith('keydown', handleKeydownMock)

    wrapper.unmount()
    expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', handleKeydownMock)

    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
  })

  it('exposes safe and named header actions', () => {
    routeState.query = { book: 'book-1', chapter: 'chapter-1' }

    const wrapper = mountTranslateView()

    expect(wrapper.get('.translate-header__back-link').attributes('aria-label')).toBe('返回书架')
    expect(wrapper.get('.translate-header__save-button').attributes('aria-label')).toBe('保存进度')
    expect(wrapper.get('.translate-header__theme-toggle').attributes('aria-label')).toBe('功能开发中')
    expect(wrapper.get('.translate-header__link--tutorial').attributes('rel')).toBe('noopener noreferrer')
    expect(wrapper.get('.translate-header__link--github').attributes('rel')).toBe('noopener noreferrer')
    expect(wrapper.get('.translate-header__link--donate').element.tagName).toBe('BUTTON')
    expect(wrapper.get('.translate-header__link--donate').attributes('href')).toBeUndefined()
  })
})
