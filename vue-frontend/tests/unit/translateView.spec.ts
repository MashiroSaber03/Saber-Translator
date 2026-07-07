import { enableAutoUnmount, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import TranslateView from '@/views/TranslateView.vue'
import { useSettingsStore } from '@/stores/settings'
import { useSessionStore } from '@/stores/sessionStore'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'

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

const ProductPageHeaderStub = defineComponent({
  props: {
    actionsLabel: {
      type: String,
      default: '页面操作',
    },
    navLabel: {
      type: String,
      default: '页面导航',
    },
    variant: {
      type: String,
      default: 'default',
    },
  },
  template: `
    <header class="product-page-header" :class="'product-page-header--' + variant">
      <slot name="meta" />
      <nav :aria-label="navLabel"><slot name="nav" /></nav>
      <div role="group" :aria-label="actionsLabel"><slot name="actions" /></div>
    </header>
  `,
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
        ProductPageHeader: ProductPageHeaderStub,
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
        UiProgressBar: true,
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

  it('exposes safe and named header actions', async () => {
    routeState.query = { book: 'book-1', chapter: 'chapter-1' }

    const wrapper = mountTranslateView()
    const settingsStore = useSettingsStore()

    expect(wrapper.get('.product-page-header--default').exists()).toBe(true)
    expect(wrapper.get('nav[aria-label="翻译页面导航"]').exists()).toBe(true)
    const actionGroup = wrapper.get('[role="group"][aria-label="翻译页面操作"]')
    expect(wrapper.get('.translate-header__back-link').attributes('aria-label')).toBe('返回书架')
    expect(wrapper.get('.translate-header__save-button').attributes('aria-label')).toBe('保存进度')
    expect(actionGroup.find('.translate-header__save-button').exists()).toBe(true)
    expect(actionGroup.find('.translate-header__settings-button').exists()).toBe(true)
    expect(actionGroup.find('.translate-header__link--donate').exists()).toBe(true)
    const themeToggle = wrapper.get('.translate-header__theme-toggle')
    expect(actionGroup.find('.translate-header__theme-toggle').exists()).toBe(true)
    expect(themeToggle.attributes('aria-label')).toBe('切换深色模式')
    await themeToggle.trigger('click')
    expect(settingsStore.theme).toBe('dark')
    expect(wrapper.get('.translate-header__link--tutorial').attributes('rel')).toBe('noopener noreferrer')
    const githubLink = wrapper.get('.translate-header__link--github')
    expect(githubLink.attributes('rel')).toBe('noopener noreferrer')
    expect(githubLink.getComponent(UiIcon).props('name')).toBe('github')
    expect(wrapper.get('.translate-header__link--donate').element.tagName).toBe('BUTTON')
    expect(wrapper.get('.translate-header__link--donate').attributes('href')).toBeUndefined()
  })

  it('keeps edit-mode shell visibility in template state instead of product header CSS reach-through', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/TranslateView.vue'), 'utf8')

    expect(source).toContain('<ProductPageHeader\n      v-show="!isEditMode"')
    expect(source).toContain('<SidebarLayout\n      v-show="!isEditMode"')
    expect(source).not.toContain('.translate-page.edit-mode-active .product-page-header')
  })

  it('keeps header actions free of DOM id hooks', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/TranslateView.vue'), 'utf8')

    expect(source).not.toContain('id="openSettingsBtn"')
    expect(source).not.toContain('github.jpg')
    expect(source).not.toContain('translate-header__github-icon')
  })

  it('renders chapter loading progress through the shared progress primitive', async () => {
    const wrapper = mountTranslateView()
    const sessionStore = useSessionStore()

    sessionStore.loadingProgress.current = 2
    sessionStore.loadingProgress.total = 5
    sessionStore.loadingProgress.message = '正在加载章节'
    await nextTick()

    const progress = wrapper.getComponent(UiProgressBar)
    expect(progress.props('value')).toBe(40)
    expect(progress.props('label')).toBe('正在加载章节')
  })

  it('keeps page owner tokens on semantic colors', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/TranslateView.vue'), 'utf8')
    const ownerTokenBlock = source.match(/\.translate-page\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(ownerTokenBlock).not.toMatch(/rgba?\(/)
    expect(ownerTokenBlock).not.toMatch(/#[0-9a-f]{3,8}\b/i)
  })

  it('keeps page-level helper hooks under the translate owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/TranslateView.vue'), 'utf8')

    expect(source).toContain('translate-bookshelf-mode-hint__text')
    expect(source).toContain('translate-upload-card__actions')
    expect(source).toContain('translate-upload-card__progress-label')
    expect(source).not.toContain('class="translate-upload-actions"')
    expect(source).not.toContain('.translate-upload-actions')
    expect(source).not.toContain('class="translate-loading-progress-label"')
    expect(source).not.toContain('.translate-loading-progress-label')
    expect(source).not.toMatch(/class="hint-text"/)
    expect(source).not.toMatch(/\.translate-bookshelf-mode-hint\s+\.hint-text/)
  })

  it('uses an owner modifier for settings highlight state', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/TranslateView.vue'), 'utf8')

    expect(source).toContain("'translate-header__settings-button--highlighted': isSettingsButtonHighlighted")
    expect(source).toContain('.translate-header__settings-button--highlighted')
    expect(source).not.toContain('{ highlight: isSettingsButtonHighlighted }')
    expect(source).not.toContain('.translate-header__settings-button.highlight')
  })
})
