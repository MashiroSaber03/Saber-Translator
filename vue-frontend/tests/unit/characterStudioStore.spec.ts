import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { CharacterStudioChatSession } from '@/types/characterStudio'

function cloneDocument<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

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
  grounding: { timeline_mode: 'enhanced', sample_pages: [1], relationships: [], key_moments: [] },
  exportArtifacts: {},
}

const structuredDocument = {
  ...cloneDocument(demoDocument),
  lorebook: {
    name: '阿尔法世界书',
    entries: [
      {
        id: 'entry_root',
        comment: '根条目',
        keys: ['阿尔法'],
        secondary_keys: [],
        content: '根条目内容',
        enabled: true,
        constant: false,
        selective: true,
        priority: 100,
        position: 'before_char',
        depth: 4,
        probability: 100,
        prevent_recursion: true,
        use_regex: false,
        match_persona_description: true,
        match_character_description: true,
        match_character_personality: true,
        match_character_depth_prompt: true,
        match_scenario: true,
        children: [
          {
            id: 'entry_child',
            comment: '子条目',
            keys: ['测试'],
            secondary_keys: [],
            content: '子条目内容',
            enabled: true,
            constant: false,
            selective: true,
            priority: 80,
            position: 'before_char',
            depth: 3,
            probability: 100,
            prevent_recursion: true,
            use_regex: false,
            match_persona_description: true,
            match_character_description: true,
            match_character_personality: true,
            match_character_depth_prompt: true,
            match_scenario: true,
            children: [],
          },
        ],
      },
    ],
  },
  regexScripts: [
    {
      id: 'regex_alpha',
      scriptName: '旧脚本',
      findRegex: '旧内容',
      replaceString: '新内容',
      placement: [2],
      markdownOnly: false,
      promptOnly: false,
      runOnEdit: true,
      disabled: false,
    },
  ],
  stateTasks: [
    {
      id: 'task_alpha',
      name: '初始化任务',
      triggerTiming: 'initialization',
      interval: 0,
      commands: '<<taskjs>>\nawait STscript(\'/setvar key=trust_score 20\');\n<</taskjs>>',
      disabled: false,
    },
  ],
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

const createCharacterStudioDocumentMock = vi.fn().mockResolvedValue({
  success: true,
  document: candidateDocument,
})
const generateCharacterStudioSectionMock = vi.fn()
const getCharacterStudioChatStateMock = vi.fn()
const createCharacterStudioChatSessionMock = vi.fn()
const switchCharacterStudioChatSessionMock = vi.fn()
const streamCharacterStudioChatMessageMock = vi.fn()
const editCharacterStudioChatMessageMock = vi.fn()
const deleteCharacterStudioChatMessageMock = vi.fn()
const regenerateCharacterStudioChatMessageMock = vi.fn()
const summarizeCharacterStudioChatSessionMock = vi.fn()
const exportCharacterStudioChatSessionMock = vi.fn()
const importCharacterStudioChatSessionMock = vi.fn()
const getCharacterStudioChatPromptPreviewMock = vi.fn()

const demoChatSession: CharacterStudioChatSession = {
  session_id: 'chat_alpha',
  doc_id: 'doc_alpha',
  title: '新对话',
  created_at: '2026-05-15T00:00:00',
  updated_at: '2026-05-15T00:00:00',
  archived_at: null,
  greeting_source: { type: 'first_message', index: 0 },
  summary_blocks: [],
  messages: [
    {
      message_id: 'msg_opening',
      role: 'assistant',
      content: '我是阿尔法。',
      attachments: [],
      runtime_log: [],
      variables_snapshot: { trust_score: 20 },
      generation_meta: {},
      created_at: '2026-05-15T00:00:00',
      updated_at: '2026-05-15T00:00:00',
    },
  ],
  variables: { trust_score: 20 },
  _runtime: {},
  last_prompt_preview: '',
}

vi.mock('@/api/characterStudio', () => ({
  createCharacterStudioDocument: createCharacterStudioDocumentMock,
  createCharacterStudioChatSession: createCharacterStudioChatSessionMock,
  switchCharacterStudioChatSession: switchCharacterStudioChatSessionMock,
  getCharacterStudioChatState: getCharacterStudioChatStateMock,
  streamCharacterStudioChatMessage: streamCharacterStudioChatMessageMock,
  editCharacterStudioChatMessage: editCharacterStudioChatMessageMock,
  deleteCharacterStudioChatMessage: deleteCharacterStudioChatMessageMock,
  regenerateCharacterStudioChatMessage: regenerateCharacterStudioChatMessageMock,
  summarizeCharacterStudioChatSession: summarizeCharacterStudioChatSessionMock,
  exportCharacterStudioChatSession: exportCharacterStudioChatSessionMock,
  importCharacterStudioChatSession: importCharacterStudioChatSessionMock,
  getCharacterStudioChatPromptPreview: getCharacterStudioChatPromptPreviewMock,
  generateCharacterStudioSection: generateCharacterStudioSectionMock,
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
    createCharacterStudioDocumentMock.mockClear()
    generateCharacterStudioSectionMock.mockReset()
    getCharacterStudioChatStateMock.mockReset()
    createCharacterStudioChatSessionMock.mockReset()
    switchCharacterStudioChatSessionMock.mockReset()
    streamCharacterStudioChatMessageMock.mockReset()
    editCharacterStudioChatMessageMock.mockReset()
    deleteCharacterStudioChatMessageMock.mockReset()
    regenerateCharacterStudioChatMessageMock.mockReset()
    summarizeCharacterStudioChatSessionMock.mockReset()
    exportCharacterStudioChatSessionMock.mockReset()
    importCharacterStudioChatSessionMock.mockReset()
    getCharacterStudioChatPromptPreviewMock.mockReset()
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
    })
    getCharacterStudioChatStateMock.mockResolvedValue({
      success: true,
      doc_id: 'doc_alpha',
      active_session: demoChatSession,
      archived_sessions: [],
      available_greetings: [],
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

  it('loads active chat session when opening a document', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    expect(store.activeChatSession?.session_id).toBe('chat_alpha')
    expect(store.activeChatSession?.messages[0]?.content).toBe('我是阿尔法。')
    expect(store.activeChatSession?.variables.trust_score).toBe(20)
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
    expect(store.activeChatSession).toBeNull()
    expect(store.archivedChatSessions).toEqual([])
    expect(store.diagnostics).toBeNull()
    expect(store.agentMessages).toEqual([])
    expect(store.pendingAgentPatch).toBeNull()
  })

  it('creates a fresh active session when starting a new conversation', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    createCharacterStudioChatSessionMock.mockResolvedValueOnce({
      success: true,
      doc_id: 'doc_alpha',
      active_session: {
        ...demoChatSession,
        session_id: 'chat_beta',
        messages: [
          {
            ...demoChatSession.messages[0],
            message_id: 'msg_beta',
            content: '新的开场白',
          },
        ],
      },
      archived_sessions: [
        {
          session_id: 'chat_alpha',
          title: '新对话',
          message_count: 1,
          updated_at: '2026-05-15T00:00:00',
        },
      ],
      available_greetings: [],
    })

    await store.createChatSession()

    expect(store.activeChatSession?.session_id).toBe('chat_beta')
    expect(store.activeChatSession?.messages[0]?.content).toBe('新的开场白')
    expect(store.archivedChatSessions[0]?.session_id).toBe('chat_alpha')
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

  it('updates a worldbook root entry by id via agent patch', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      worldbook_update: {
        id: 'entry_root',
        changes: {
          content: '更新后的根条目内容',
          priority: 250,
        },
      },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.content).toBe('更新后的根条目内容')
    expect(store.currentDocument?.lorebook.entries[0]?.priority).toBe(250)
    expect(store.pendingAgentPatch).toBeNull()
  })

  it('updates and deletes nested worldbook entries by id via agent patch', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      worldbook_update: {
        id: 'entry_child',
        changes: {
          content: '更新后的子条目内容',
          keys: ['测试', '支线'],
        },
      },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.children[0]?.content).toBe('更新后的子条目内容')
    expect(store.currentDocument?.lorebook.entries[0]?.children[0]?.keys).toEqual(['测试', '支线'])

    store.pendingAgentPatch = {
      worldbook_delete: {
        id: 'entry_child',
      },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.children).toEqual([])
  })

  it('updates and deletes regex and task entries by id via agent patch', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      regex_update: {
        id: 'regex_alpha',
        changes: {
          replaceString: '更新后的替换内容',
          placement: [1, 2],
        },
      },
      task_update: {
        id: 'task_alpha',
        changes: {
          interval: 3,
          commands: '<<taskjs>>\nawait STscript(\'/setvar key=trust_score 40\');\n<</taskjs>>',
        },
      },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.regexScripts[0]?.replaceString).toBe('更新后的替换内容')
    expect(store.currentDocument?.regexScripts[0]?.placement).toEqual([1, 2])
    expect(store.currentDocument?.stateTasks[0]?.interval).toBe(3)
    expect(store.currentDocument?.stateTasks[0]?.commands).toContain('trust_score 40')

    store.pendingAgentPatch = {
      regex_delete: { id: 'regex_alpha' },
      task_delete: { id: 'task_alpha' },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.regexScripts).toEqual([])
    expect(store.currentDocument?.stateTasks).toEqual([])
  })

  it('keeps pending patch and document state unchanged when patch target id does not exist', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const before = cloneDocument(store.currentDocument)

    const missingPatch = {
      regex_update: {
        id: 'regex_missing',
        changes: {
          replaceString: '不会生效',
        },
      },
    }

    store.pendingAgentPatch = missingPatch as any
    store.applyPendingPatch()

    expect(store.currentDocument).toEqual(before)
    expect(store.pendingAgentPatch).toEqual(missingPatch)
    expect(store.errorMessage).toContain('regex_missing')
  })

  it('rejects unsupported patch top-level fields instead of silently ignoring them', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const unsupportedPatch = {
      worldbook_move: {
        id: 'entry_root',
      },
    }

    const before = cloneDocument(store.currentDocument)
    store.pendingAgentPatch = unsupportedPatch as any
    store.applyPendingPatch()

    expect(store.currentDocument).toEqual(before)
    expect(store.pendingAgentPatch).toEqual(unsupportedPatch)
    expect(store.errorMessage).toContain('不支持的 patch 顶层字段')
  })

  it('rejects set paths that try to mutate collection entries directly', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const before = cloneDocument(store.currentDocument)
    const invalidSetPatch = {
      set: {
        'regexScripts.0.disabled': true,
      },
    }

    store.pendingAgentPatch = invalidSetPatch as any
    store.applyPendingPatch()

    expect(store.currentDocument).toEqual(before)
    expect(store.pendingAgentPatch).toEqual(invalidSetPatch)
    expect(store.errorMessage).toContain('set 不允许直接修改集合字段')
  })

  it('rejects regex placements outside the runtime-supported range', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const invalidRegexPatch = {
      regex_update: {
        id: 'regex_alpha',
        changes: {
          placement: [3],
        },
      },
    }

    const before = cloneDocument(store.currentDocument)
    store.pendingAgentPatch = invalidRegexPatch as any
    store.applyPendingPatch()

    expect(store.currentDocument).toEqual(before)
    expect(store.pendingAgentPatch).toEqual(invalidRegexPatch)
    expect(store.errorMessage).toContain('只能使用 1 或 2')
  })

  it('rejects unsupported lorebook positions so prompt rules stay aligned with runtime behavior', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const invalidWorldbookPatch = {
      worldbook_update: {
        id: 'entry_root',
        changes: {
          position: 'top_an',
        },
      },
    }

    const before = cloneDocument(store.currentDocument)
    store.pendingAgentPatch = invalidWorldbookPatch as any
    store.applyPendingPatch()

    expect(store.currentDocument).toEqual(before)
    expect(store.pendingAgentPatch).toEqual(invalidWorldbookPatch)
    expect(store.errorMessage).toContain('before_char、at_depth、after_char')
  })

  it('skips frozen section operations while applying other valid patch ops', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    const frozenDocument = cloneDocument(structuredDocument)
    frozenDocument.status.frozen_sections = ['lorebook']

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: frozenDocument,
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      worldbook_update: {
        id: 'entry_root',
        changes: {
          content: '不应更新',
        },
      },
      regex_update: {
        id: 'regex_alpha',
        changes: {
          disabled: true,
        },
      },
    } as any

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.content).toBe('根条目内容')
    expect(store.currentDocument?.regexScripts[0]?.disabled).toBe(true)
  })

  it('can undo a v2 agent patch after applying it', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: cloneDocument(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      task_delete: {
        id: 'task_alpha',
      },
    } as any

    store.applyPendingPatch()
    expect(store.currentDocument?.stateTasks).toEqual([])

    store.undoLastPatch()
    expect(store.currentDocument?.stateTasks[0]?.id).toBe('task_alpha')
  })

  it('clears a no-op frozen patch without creating undo state or autosaving', async () => {
    vi.useFakeTimers()
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    const frozenDocument = cloneDocument(structuredDocument)
    frozenDocument.status.frozen_sections = ['lorebook']

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: frozenDocument,
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      worldbook_update: {
        id: 'entry_root',
        changes: {
          content: '不会生效',
        },
      },
    } as any

    store.applyPendingPatch()
    await vi.advanceTimersByTimeAsync(1500)

    expect(store.pendingAgentPatch).toBeNull()
    expect(store.canUndoPatch).toBe(false)
    expect(saveCharacterStudioDocumentMock).not.toHaveBeenCalled()
  })
})
