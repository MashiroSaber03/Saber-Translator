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

const candidateDocument = {
  ...demoDocument,
  id: 'doc_candidate',
  origin: { type: 'analysis', source_character: '候选角色', source_pages: [] },
  meta: { ...demoDocument.meta, title: '候选角色', tags: [] },
  identity: { ...demoDocument.identity, name: '候选角色', aliases: [], description: '', personality: '', scenario: '' },
  coreMessages: { ...demoDocument.coreMessages, first_message: '', alternate_greetings: [] },
  lorebook: { name: '候选角色世界书', entries: [] },
  regexScripts: [],
  stateTasks: [],
  grounding: { timeline_mode: '', sample_pages: [], relationships: [], key_moments: [] },
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
      dialogue_count: 2,
      has_dialogues: true,
      sample_pages: [1],
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

const previewCharacterStudioChatMock = vi.fn()
const resetCharacterStudioPreviewMock = vi.fn()
const createCharacterStudioDocumentMock = vi.fn().mockResolvedValue({
  success: true,
  document: candidateDocument,
})
const generateCharacterStudioSectionMock = vi.fn()

vi.mock('@/api/characterStudio', () => ({
  createCharacterStudioDocument: createCharacterStudioDocumentMock,
  generateCharacterStudioSection: generateCharacterStudioSectionMock,
  getCharacterStudioIndex: getCharacterStudioIndexMock,
  getCharacterStudioDocument: getCharacterStudioDocumentMock,
  saveCharacterStudioDocument: saveCharacterStudioDocumentMock,
  previewCharacterStudioChat: previewCharacterStudioChatMock,
  resetCharacterStudioPreview: resetCharacterStudioPreviewMock,
}))

function createDeferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve
    reject = innerReject
  })
  return { promise, resolve, reject }
}

describe('characterStudioStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useRealTimers()
    getCharacterStudioIndexMock.mockClear()
    getCharacterStudioDocumentMock.mockClear()
    saveCharacterStudioDocumentMock.mockClear()
    createCharacterStudioDocumentMock.mockClear()
    generateCharacterStudioSectionMock.mockReset()
    previewCharacterStudioChatMock.mockReset()
    resetCharacterStudioPreviewMock.mockReset()
    getCharacterStudioIndexMock.mockResolvedValue({
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
          dialogue_count: 2,
          has_dialogues: true,
          sample_pages: [1],
        },
      ],
      count: 1,
    })
    getCharacterStudioDocumentMock.mockResolvedValue({
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

  it('clears stale document state when loading a different book workspace', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    getCharacterStudioIndexMock.mockResolvedValueOnce({
      success: true,
      book_id: 'book-other',
      documents: [],
      candidates: [],
      count: 0,
      has_timeline: false,
    })

    await store.loadWorkspace('book-other')

    expect(store.bookId).toBe('book-other')
    expect(store.currentDocument).toBeNull()
    expect(store.previewSession).toBeNull()
    expect(store.diagnostics).toBeNull()
    expect(store.agentMessages).toEqual([])
    expect(store.pendingAgentPatch).toBeNull()
  })

  it('ignores stale preview replies that resolve after a preview reset', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const deferredPreview = createDeferred<{
      success: boolean
      doc_id: string
      messages: Array<{ role: 'user' | 'assistant'; content: string }>
      variables: Record<string, unknown>
      log: Array<Record<string, unknown>>
    }>()

    previewCharacterStudioChatMock.mockReturnValueOnce(deferredPreview.promise)
    resetCharacterStudioPreviewMock.mockResolvedValueOnce({
      success: true,
      doc_id: 'doc_alpha',
      messages: [],
      variables: {},
      log: [],
    })

    const previewPromise = store.sendPreviewMessage('你好')
    await Promise.resolve()
    await store.resetPreview()

    deferredPreview.resolve({
      success: true,
      doc_id: 'doc_alpha',
      messages: [{ role: 'assistant', content: '这是一条过期回复' }],
      variables: { trust_score: 42 },
      log: [{ type: 'task', name: '过期任务' }],
    })

    await previewPromise

    expect(store.previewSession?.messages).toEqual([])
    expect(store.previewSession?.variables).toEqual({})
    expect(store.previewSession?.log).toEqual([])
  })

  it('creates a candidate document without prefilled card content', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioIndexMock.mockResolvedValue({
      success: true,
      book_id: 'book-demo',
      documents: [
        {
          id: 'doc_candidate',
          title: '候选角色',
          origin: 'analysis',
          source_character: '候选角色',
          updated_at: '2026-05-15T00:00:00',
          tags: [],
          is_favorite: false,
          has_avatar: false,
          sample_pages: [],
        },
      ],
      candidates: [],
      count: 1,
    })
    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: candidateDocument,
      preview_session: {
        doc_id: 'doc_candidate',
        messages: [],
        variables: {},
        log: [],
      },
    })

    await store.loadWorkspace('book-demo')
    await store.createDocumentFromCandidate('候选角色')

    expect(store.currentDocument?.identity.name).toBe('候选角色')
    expect(store.currentDocument?.identity.description).toBe('')
    expect(store.currentDocument?.coreMessages.first_message).toBe('')
    expect(store.currentDocument?.lorebook.entries).toEqual([])
  })

  it('shows dedicated progress copy for full card generation', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    generateCharacterStudioSectionMock.mockImplementationOnce(async () => new Promise(() => {}))
    void store.generateSection('full')
    await Promise.resolve()

    expect(store.activeActionLabel).toBe('正在补全整张角色卡')
  })

  it('preserves backend validation messages when section generation fails', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const { ApiClientError } = await import('@/api/client')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    generateCharacterStudioSectionMock.mockRejectedValueOnce(new ApiClientError({
      code: 'ERR_BAD_REQUEST',
      message: 'AI 生成结果缺少 identity。',
      status: 400,
      details: { section: 'full' },
    }))

    await expect(store.generateSection('full')).rejects.toThrow('AI 生成结果缺少 identity。')
    expect(store.errorMessage).toBe('AI 生成结果缺少 identity。')
  })

  it('keeps document title in sync when an agent patch changes identity.name', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      set: {
        'identity.name': '新名字',
      },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.identity.name).toBe('新名字')
    expect(store.currentDocument?.meta.title).toBe('新名字')
  })
})
