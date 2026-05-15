import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import CharacterStudioView from '@/views/CharacterStudioView.vue'
import { useCharacterStudioStore } from '@/stores/characterStudioStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
}))

describe('CharacterStudioView workspace shell', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders dedicated scroll containers for the three-column workspace', async () => {
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
      previewState: { variables: {}, messages: [] },
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

    expect(wrapper.find('[data-testid="left-scroll"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="editor-scroll"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="right-scroll"]').exists()).toBe(true)
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
})
