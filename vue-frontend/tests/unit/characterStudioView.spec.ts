import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import CharacterStudioView from '@/views/CharacterStudioView.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { useCharacterStudioStore } from '@/stores/characterStudioStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'

const { confirmProductActionMock } = vi.hoisted(() => ({
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

const pushMock = vi.fn()
const replaceMock = vi.fn()

function createDeferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void
  const promise = new Promise<void>(resolvePromise => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: pushMock,
    replace: replaceMock,
  }),
}))

describe('CharacterStudioView workspace shell', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    pushMock.mockReset()
    replaceMock.mockReset()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
  })

  it('uses current product copy for the missing-book empty state', () => {
    const wrapper = mount(CharacterStudioView, {
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    const emptyState = wrapper.getComponent(ProductEmptyState)
    expect(emptyState.props()).toMatchObject({
      eyebrow: '缺少上下文',
      iconName: 'alert-triangle',
      title: '未检测到书籍参数',
      description:
        '请从漫画分析页进入角色工坊，或在 URL 中携带 `book` 参数。角色工坊需要当前书籍的分析上下文。',
    })
    expect(wrapper.text()).toContain('角色工坊需要当前书籍的分析上下文')
    expect(wrapper.text()).not.toContain('仍然依赖')
    expect(wrapper.find('.studio-empty-state').exists()).toBe(false)
    expect(wrapper.find('.empty-badge').exists()).toBe(false)
  })

  it('renders dedicated scroll containers for the two-pane workspace', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockResolvedValue(undefined)
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)
    studioStore.currentDocument = {
      id: 'doc_alpha',
      bookId: 'book-demo',
      origin: { type: 'manual', source_character: null },
      status: {
        is_favorite: false,
        frozen_sections: [],
        last_diagnostics: null,
        last_validated_at: null,
      },
      meta: {
        title: '阿尔法',
        tags: [],
      },
      identity: { name: '阿尔法', aliases: [], description: '', personality: '', scenario: '' },
      coreMessages: {
        first_message: '',
        message_example: '',
        alternate_greetings: [],
        system_prompt: '',
        post_history_instructions: '',
        creator_notes: '',
        character_version: '2.0.0',
      },
      lorebook: { name: '阿尔法世界书', entries: [] },
      regexScripts: [],
      stateTasks: [],
      exportArtifacts: {},
      revision: 1,
      avatarUrl: null,
      createdAt: '2026-05-15T00:00:00',
      updatedAt: '2026-05-15T00:00:00',
    }

    const wrapper = mount(CharacterStudioView, {
      props: {
        bookId: 'book-demo',
      },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    expect(wrapper.find('.product-split-workspace').exists()).toBe(true)
    expect(wrapper.find('[data-testid="editor-scroll"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="chat-scroll"]').exists()).toBe(true)
  })

  it('requires confirmation before deleting a role document', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()
    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockResolvedValue(undefined)
    studioStore.currentDocument = {
      id: 'doc-delete',
      origin: { type: 'manual', source_character: null },
      meta: { title: '待删除角色' },
    } as never
    studioStore.deleteCurrentDocument = vi.fn().mockResolvedValue(undefined)
    confirmProductActionMock.mockResolvedValueOnce(false).mockResolvedValueOnce(true)

    const wrapper = mount(CharacterStudioView, {
      props: { bookId: 'book-demo' },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div />' },
          CharacterStudioEditor: {
            emits: ['delete'],
            template:
              '<button class="delete-document" @click="$emit(\'delete\')">删除文档</button>',
          },
          CharacterStudioPreview: { template: '<div />' },
          StudioTopbar: { template: '<div />' },
        },
      },
    })

    await wrapper.get('.delete-document').trigger('click')
    await flushPromises()
    expect(studioStore.deleteCurrentDocument).not.toHaveBeenCalled()

    await wrapper.get('.delete-document').trigger('click')
    await flushPromises()
    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除角色文档',
      message: '确定删除“待删除角色”吗？角色设定、聊天会话与导出记录将一并删除，无法恢复。',
      confirmText: '删除',
      tone: 'danger',
    })
    expect(studioStore.deleteCurrentDocument).toHaveBeenCalledOnce()
  })

  it('maps view owner colors through semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/CharacterStudioView.vue'), 'utf8')
    const style = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(style).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('keeps Studio owner tokens explicit without styling primitive buttons from the view root', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/CharacterStudioView.vue'), 'utf8')
    const rootStyle = source.match(/\.studio-page \{(?<body>[\s\S]*?)\n\}/)

    expect(rootStyle?.groups?.body ?? '').toContain('--studio-surface-tint-muted:')
    expect(rootStyle?.groups?.body ?? '').toContain('--studio-shadow-floating:')
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
    expect(source).toContain('store.currentDocument?.avatarUrl')
    expect(source).not.toContain('getCharacterStudioAvatarUrl')
  })

  it('keeps the split-pane slot height contract aligned with current Studio child owners', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/CharacterStudioView.vue'), 'utf8')
    const style = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(style).toContain('.studio-page__workspace-slot-content > .studio-editor')
    expect(style).toContain('.studio-page__workspace-slot-content > .character-studio-preview')
    expect(style).not.toContain('.workspace-slot-content > .chat-shell')
  })

  it('keeps page-local shell hooks under the Studio page owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/CharacterStudioView.vue'), 'utf8')

    for (const currentHook of [
      'class="studio-page__missing-context-state"',
      'class="studio-page__workspace-root"',
      'class="studio-page__workspace-error-banner"',
      'class="studio-page__workspace-shell"',
      'class="studio-page__workspace-slot-content"',
      'class="studio-page__resource-dialog"',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const legacyHook of [
      'class="studio-missing-context-state"',
      'class="workspace-root"',
      'class="workspace-error-banner"',
      'class="workspace-shell"',
      'class="workspace-slot-content"',
      'class="resource-overlay"',
      'class="resource-dialog"',
    ]) {
      expect(source).not.toContain(legacyHook)
    }

    expect(source).not.toMatch(
      /\.(?:workspace-root|workspace-shell|workspace-slot-content|workspace-error-banner|resource-overlay|resource-dialog)\b/
    )
  })

  it('exposes the pane resizer as a keyboard-adjustable separator', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockResolvedValue(undefined)
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)

    const wrapper = mount(CharacterStudioView, {
      props: {
        bookId: 'book-demo',
      },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    const resizer = wrapper.find('[role="separator"]')
    expect(resizer.attributes('role')).toBe('separator')
    expect(resizer.attributes('aria-orientation')).toBe('vertical')
    expect(resizer.attributes('aria-valuemin')).toBe('35')
    expect(resizer.attributes('aria-valuemax')).toBe('70')
    expect(resizer.attributes('aria-valuenow')).toBe('52')

    await resizer.trigger('keydown', { key: 'ArrowRight' })

    expect(resizer.attributes('aria-valuenow')).toBe('54')
  })

  it('shows store error message in the workspace shell', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockResolvedValue(undefined)
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)
    studioStore.errorMessage = '导出失败：测试错误'

    const wrapper = mount(CharacterStudioView, {
      props: {
        bookId: 'book-demo',
      },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    expect(wrapper.text()).toContain('导出失败：测试错误')
    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('danger')
    expect(banner.props('ariaLive')).toBe('assertive')
    expect(wrapper.find('.workspace-error').exists()).toBe(false)
    expect(wrapper.find('.workspace-error__message').exists()).toBe(false)
  })

  it('renders the resource dialog shell when the resource panel is open', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockResolvedValue(undefined)
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)
    studioStore.resourcePanelOpen = true

    const wrapper = mount(CharacterStudioView, {
      props: {
        bookId: 'book-demo',
      },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    const dialog = document.body.querySelector('[data-testid="base-dialog-container"]')
    expect(dialog?.getAttribute('role')).toBe('dialog')
    expect(dialog?.getAttribute('aria-modal')).toBe('true')
    expect(dialog?.textContent).toContain('角色资源')
    expect(document.body.querySelector('[data-testid="resource-dialog"]')).not.toBeNull()
    expect(document.body.querySelector('[data-testid="base-dialog-close"]')).not.toBeNull()
    wrapper.unmount()
  })

  it('falls back to the first available document when requested docId cannot be opened', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockImplementation(async () => {
      studioStore.documents = [
        {
          id: 'doc_alpha',
          title: '阿尔法',
          origin: 'manual',
          source_character: null,
          updated_at: '2026-05-15T00:00:00',
          tags: [],
          is_favorite: false,
          has_avatar: false,
        },
      ]
    })
    studioStore.openDocument = vi
      .fn()
      .mockRejectedValueOnce(new Error('文档不存在'))
      .mockResolvedValueOnce(undefined)

    mount(CharacterStudioView, {
      props: {
        bookId: 'book-demo',
        docId: 'missing-doc',
      },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    await flushPromises()

    expect(studioStore.openDocument).toHaveBeenCalledTimes(2)
    expect(studioStore.openDocument).toHaveBeenNthCalledWith(1, 'missing-doc')
    expect(studioStore.openDocument).toHaveBeenNthCalledWith(2, 'doc_alpha')
    expect(replaceMock).toHaveBeenCalledWith({
      name: 'character-studio',
      query: { book: 'book-demo', doc: 'doc_alpha' },
    })
  })

  it('also falls back when a later route change points to an invalid document', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockImplementation(async () => {
      studioStore.documents = [
        {
          id: 'doc_alpha',
          title: '阿尔法',
          origin: 'manual',
          source_character: null,
          updated_at: '2026-05-15T00:00:00',
          tags: [],
          is_favorite: false,
          has_avatar: false,
        },
      ]
    })
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)

    const wrapper = mount(CharacterStudioView, {
      props: { bookId: 'book-demo', docId: 'doc_alpha' },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })
    await flushPromises()
    ;(studioStore.openDocument as ReturnType<typeof vi.fn>).mockReset()
    ;(studioStore.openDocument as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new Error('文档不存在'))
      .mockResolvedValueOnce(undefined)
    replaceMock.mockClear()

    await wrapper.setProps({ docId: 'missing-later' })
    await flushPromises()

    expect(studioStore.openDocument).toHaveBeenNthCalledWith(1, 'missing-later')
    expect(studioStore.openDocument).toHaveBeenNthCalledWith(2, 'doc_alpha')
    expect(replaceMock).toHaveBeenCalledWith({
      name: 'character-studio',
      query: { book: 'book-demo', doc: 'doc_alpha' },
    })
  })

  it('ignores stale workspace hydration after the route book changes', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    const alphaLoad = createDeferred()
    const betaLoad = createDeferred()

    bookshelfStore.books = [
      { id: 'book-alpha', title: '阿尔法书籍' },
      { id: 'book-beta', title: '贝塔书籍' },
    ] as typeof bookshelfStore.books
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockImplementation(async (bookId: string) => {
      await (bookId === 'book-alpha' ? alphaLoad.promise : betaLoad.promise)
      studioStore.documents = [
        {
          id: `${bookId}-doc`,
          title: `${bookId} 文档`,
          origin: 'manual',
          source_character: null,
          updated_at: '2026-05-15T00:00:00',
          tags: [],
          is_favorite: false,
          has_avatar: false,
        },
      ]
    })
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)

    const wrapper = mount(CharacterStudioView, {
      props: {
        bookId: 'book-alpha',
      },
      global: {
        stubs: {
          CharacterStudioSidebar: { template: '<div class="sidebar-stub">sidebar</div>' },
          CharacterStudioEditor: { template: '<div class="editor-stub">editor</div>' },
          CharacterStudioPreview: { template: '<div class="preview-stub">preview</div>' },
          StudioTopbar: { template: '<div class="topbar-stub">topbar</div>' },
        },
      },
    })

    await wrapper.setProps({ bookId: 'book-beta' })
    betaLoad.resolve()
    await flushPromises()

    expect(replaceMock).toHaveBeenCalledWith({
      name: 'character-studio',
      query: { book: 'book-beta', doc: 'book-beta-doc' },
    })

    replaceMock.mockClear()
    ;(studioStore.openDocument as ReturnType<typeof vi.fn>).mockClear()

    alphaLoad.resolve()
    await flushPromises()

    expect(studioStore.openDocument).not.toHaveBeenCalled()
    expect(replaceMock).not.toHaveBeenCalled()
  })
})
