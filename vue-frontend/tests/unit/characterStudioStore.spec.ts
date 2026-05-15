import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

const demoDocument = {
  id: 'doc_alpha',
  bookId: 'book-demo',
  origin: { type: 'manual', source_character: null, source_pages: [] },
  status: { is_favorite: false, frozen_sections: [], last_validated_at: null },
  meta: { title: '阿尔法', tags: ['主角'], created_at: '2026-05-15T00:00:00', updated_at: '2026-05-15T00:00:00' },
  avatar: { mode: 'none', asset_path: null, source_page: null },
  identity: { name: '阿尔法', aliases: [], description: '测试角色', personality: '沉稳', scenario: '测试场景' },
  coreMessages: {
    first_message: '我是阿尔法。',
    message_example: '<START>',
    alternate_greetings: [],
    system_prompt: '保持角色设定一致。',
    post_history_instructions: '',
    creator_notes: '',
    character_version: '2.0.0',
  },
  lorebook: { name: '阿尔法世界书', entries: [] },
  regexScripts: [],
  stateTasks: [],
  chatPreset: { opening_mode: 'first_message' },
  previewState: { variables: {}, messages: [] },
  grounding: { timeline_mode: 'enhanced', sample_pages: [1], relationships: [], key_moments: [] },
  exportArtifacts: {},
}

vi.mock('@/api/characterStudio', () => ({
  getCharacterStudioIndex: vi.fn().mockResolvedValue({
    success: true,
    book_id: 'book-demo',
    documents: [
      {
        id: 'doc_alpha',
        title: '阿尔法',
        origin: 'manual',
        source_character: null,
        updated_at: '2026-05-15T00:00:00',
        tags: ['主角'],
        is_favorite: false,
        has_avatar: false,
        sample_pages: [1],
      },
    ],
    candidates: [
      {
        name: '阿尔法',
        aliases: [],
        first_appearance: 1,
        description: '测试角色',
        arc: '成长',
        dialogue_count: 2,
        has_dialogues: true,
        sample_pages: [1],
        relationship_count: 0,
        key_moment_count: 0,
      },
    ],
    count: 1,
  }),
  getCharacterStudioDocument: vi.fn().mockResolvedValue({
    success: true,
    document: demoDocument,
  }),
}))

describe('characterStudioStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('loads index payload for a book', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')

    expect(store.bookId).toBe('book-demo')
    expect(store.documents).toHaveLength(1)
    expect(store.candidates).toHaveLength(1)
  })

  it('loads a document when selected', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    expect(store.currentDocument?.id).toBe('doc_alpha')
    expect(store.currentDocument?.identity.name).toBe('阿尔法')
  })
})
