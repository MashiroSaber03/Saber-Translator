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

const getCharacterStudioIndexMock = vi.fn().mockResolvedValue({
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
})

const getCharacterStudioDocumentMock = vi.fn().mockResolvedValue({
  success: true,
  document: demoDocument,
  preview_session: {
    doc_id: 'doc_alpha',
    messages: [
      { role: 'assistant', content: '已恢复的预览消息' },
    ],
    variables: { trust_score: 88 },
    log: [{ type: 'lorebook', comment: '恢复命中' }],
  },
})

const saveCharacterStudioDocumentMock = vi.fn().mockImplementation(async (_bookId: string, _docId: string, payload: Record<string, unknown>) => ({
  success: true,
  document: {
    ...demoDocument,
    ...payload,
    meta: {
      ...demoDocument.meta,
      ...((payload.meta as Record<string, unknown> | undefined) || {}),
      updated_at: new Date().toISOString(),
    },
  },
}))

vi.mock('@/api/characterStudio', () => ({
  getCharacterStudioIndex: getCharacterStudioIndexMock,
  getCharacterStudioDocument: getCharacterStudioDocumentMock,
  saveCharacterStudioDocument: saveCharacterStudioDocumentMock,
}))

describe('characterStudioStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useRealTimers()
    getCharacterStudioIndexMock.mockClear()
    getCharacterStudioDocumentMock.mockClear()
    saveCharacterStudioDocumentMock.mockClear()
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

  it('restores persisted preview session when opening a document', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    expect(store.previewSession?.messages[0]?.content).toBe('已恢复的预览消息')
    expect(store.previewSession?.variables.trust_score).toBe(88)
  })

  it('does not start autosave loop immediately after opening a document', async () => {
    vi.useFakeTimers()
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')
    await vi.advanceTimersByTimeAsync(2500)

    expect(saveCharacterStudioDocumentMock).not.toHaveBeenCalled()
  })

  it('autosaves user edits only once instead of re-saving server-updated document metadata', async () => {
    vi.useFakeTimers()
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    if (!store.currentDocument) {
      throw new Error('currentDocument missing in test setup')
    }

    store.updateCurrentDocument({
      ...store.currentDocument,
      identity: {
        ...store.currentDocument.identity,
        description: '新的角色描述',
      },
    })
    await vi.advanceTimersByTimeAsync(3000)

    expect(saveCharacterStudioDocumentMock).toHaveBeenCalledTimes(1)
  })

  it('manual save cancels any queued autosave request', async () => {
    vi.useFakeTimers()
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    if (!store.currentDocument) {
      throw new Error('currentDocument missing in test setup')
    }

    store.updateCurrentDocument({
      ...store.currentDocument,
      identity: {
        ...store.currentDocument.identity,
        description: '准备手动保存',
      },
    })

    await store.persistCurrentDocument()
    await vi.advanceTimersByTimeAsync(2000)

    expect(saveCharacterStudioDocumentMock).toHaveBeenCalledTimes(1)
  })
})
