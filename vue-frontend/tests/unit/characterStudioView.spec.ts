import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import CharacterStudioView from '@/views/CharacterStudioView.vue'
import { useCharacterStudioStore } from '@/stores/characterStudioStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'

const pushMock = vi.fn()
const replaceMock = vi.fn()

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
  })

  it('renders dedicated scroll containers for the two-pane workspace', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.fetchBooks = vi.fn().mockResolvedValue(undefined)
    studioStore.loadWorkspace = vi.fn().mockResolvedValue(undefined)
    studioStore.openDocument = vi.fn().mockResolvedValue(undefined)
    studioStore.currentDocument = {
      id: 'doc_alpha',
      bookId: 'book-demo',
      origin: { type: 'manual', source_character: null, source_pages: [] },
      status: { is_favorite: false, frozen_sections: [], last_validated_at: null },
      meta: { title: '阿尔法', tags: [], created_at: '2026-05-15T00:00:00', updated_at: '2026-05-15T00:00:00' },
      avatar: { mode: 'none', asset_path: null, source_page: null },
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
      chatPreset: { opening_mode: 'first_message' },
      grounding: { timeline_mode: '', sample_pages: [], relationships: [], key_moments: [] },
      exportArtifacts: {},
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

    expect(wrapper.find('[data-testid="editor-scroll"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="chat-scroll"]').exists()).toBe(true)
  })

  it('shows store error message in the workspace shell', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.fetchBooks = vi.fn().mockResolvedValue(undefined)
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
  })

  it('renders the resource dialog shell when the resource panel is open', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.fetchBooks = vi.fn().mockResolvedValue(undefined)
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

    expect(wrapper.find('[data-testid="resource-overlay"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="resource-dialog"]').exists()).toBe(true)
  })

  it('falls back to the first available document when requested docId cannot be opened', async () => {
    const studioStore = useCharacterStudioStore()
    const bookshelfStore = useBookshelfStore()

    bookshelfStore.books = [{ id: 'book-demo', title: '测试书籍' }] as typeof bookshelfStore.books
    bookshelfStore.fetchBooks = vi.fn().mockResolvedValue(undefined)
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
          sample_pages: [],
        },
      ]
    })
    studioStore.openDocument = vi.fn()
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
})
