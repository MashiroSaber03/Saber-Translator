import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { CharacterStudioChatStreamEvent } from '@/api/characterStudio'
import type { CharacterStudioAgentPatchV2, CharacterStudioChatSession, CharacterStudioDocument } from '@/types/characterStudio'
import { buildCharacterStudioGreetingOptions } from '@/utils/characterStudioGreetings'
import { deepClone } from '@/utils/deepClone'

const demoDocument = {
  id: 'doc_alpha',
  bookId: 'book-demo',
  origin: { type: 'manual', source_character: null },
  status: { is_favorite: false, frozen_sections: [], last_validated_at: null },
  meta: { title: '阿尔法', tags: ['主角'], created_at: '2026-05-15T00:00:00', updated_at: '2026-05-15T00:00:00' },
  avatar: { asset_path: null },
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

const structuredDocument: CharacterStudioDocument = {
  ...deepClone(demoDocument),
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
      scriptName: '初始脚本',
      findRegex: '初始内容',
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

const candidateDocument: CharacterStudioDocument = {
  ...demoDocument,
  id: 'doc_candidate',
  origin: { type: 'analysis', source_character: '候选角色' },
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
const deleteCharacterStudioChatSessionMock = vi.fn()
const abortCharacterStudioChatOperationMock = vi.fn()
const streamCharacterStudioChatMessageMock = vi.fn()
const editCharacterStudioChatMessageMock = vi.fn()
const deleteCharacterStudioChatMessageMock = vi.fn()
const regenerateCharacterStudioChatMessageMock = vi.fn()
const summarizeCharacterStudioChatSessionMock = vi.fn()
const exportCharacterStudioChatSessionMock = vi.fn()
const importCharacterStudioChatSessionMock = vi.fn()
const getCharacterStudioChatPromptPreviewMock = vi.fn()
const importWorldbookIntoCharacterStudioDocumentMock = vi.fn()

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

const conversationChatSession: CharacterStudioChatSession = {
  ...deepClone(demoChatSession),
  messages: [
    {
      ...deepClone(demoChatSession.messages[0]!),
      message_id: 'msg_opening',
      content: '我是阿尔法。',
      generation_meta: { kind: 'opening' },
    },
    {
      message_id: 'msg_user_1',
      role: 'user',
      content: '今天情况怎么样？',
      attachments: [],
      runtime_log: [],
      variables_snapshot: { trust_score: 20 },
      generation_meta: { original_content: '今天情况怎么样？' },
      created_at: '2026-05-15T00:01:00',
      updated_at: '2026-05-15T00:01:00',
    },
    {
      message_id: 'msg_assistant_1',
      role: 'assistant',
      content: '局势暂时稳定，但还需要继续观察。',
      attachments: [],
      runtime_log: [],
      variables_snapshot: { trust_score: 20 },
      generation_meta: {},
      created_at: '2026-05-15T00:01:05',
      updated_at: '2026-05-15T00:01:05',
    },
  ],
}

vi.mock('@/api/characterStudio', () => ({
  createCharacterStudioDocument: createCharacterStudioDocumentMock,
  createCharacterStudioChatSession: createCharacterStudioChatSessionMock,
  switchCharacterStudioChatSession: switchCharacterStudioChatSessionMock,
  deleteCharacterStudioChatSession: deleteCharacterStudioChatSessionMock,
  abortCharacterStudioChatOperation: abortCharacterStudioChatOperationMock,
  getCharacterStudioChatState: getCharacterStudioChatStateMock,
  streamCharacterStudioChatMessage: streamCharacterStudioChatMessageMock,
  editCharacterStudioChatMessage: editCharacterStudioChatMessageMock,
  deleteCharacterStudioChatMessage: deleteCharacterStudioChatMessageMock,
  regenerateCharacterStudioChatMessage: regenerateCharacterStudioChatMessageMock,
  summarizeCharacterStudioChatSession: summarizeCharacterStudioChatSessionMock,
  exportCharacterStudioChatSession: exportCharacterStudioChatSessionMock,
  importCharacterStudioChatSession: importCharacterStudioChatSessionMock,
  getCharacterStudioChatPromptPreview: getCharacterStudioChatPromptPreviewMock,
  importWorldbookIntoCharacterStudioDocument: importWorldbookIntoCharacterStudioDocumentMock,
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
    deleteCharacterStudioChatSessionMock.mockReset()
    abortCharacterStudioChatOperationMock.mockReset()
    streamCharacterStudioChatMessageMock.mockReset()
    editCharacterStudioChatMessageMock.mockReset()
    deleteCharacterStudioChatMessageMock.mockReset()
    regenerateCharacterStudioChatMessageMock.mockReset()
    summarizeCharacterStudioChatSessionMock.mockReset()
    exportCharacterStudioChatSessionMock.mockReset()
    importCharacterStudioChatSessionMock.mockReset()
    getCharacterStudioChatPromptPreviewMock.mockReset()
    importWorldbookIntoCharacterStudioDocumentMock.mockReset()
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

  it('keeps agent patch cloning on the shared clone helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioPatch.ts'), 'utf8')

    expect(source).toContain("import { deepClone } from '@/utils/deepClone'")
    expect(source).not.toContain('function cloneDocument')
    expect(source).not.toContain('function cloneValue')
    expect(source).not.toContain('JSON.parse(JSON.stringify')
  })

  it('keeps agent patch dynamic writes behind a named document boundary', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioPatch.ts'), 'utf8')

    expect(source).not.toContain('setByPath(nextDocument as unknown as Record<string, unknown>, path, value)')
    expect(source).toContain('type MutableCharacterStudioDocument')
  })

  it('keeps dynamic patch path traversal behind a named record helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioPatch.ts'), 'utf8')

    expect(source).toContain('function ensurePathRecord')
    expect(source).not.toContain('current = current[key] as Record<string, unknown>')
  })

  it('keeps store snapshots on the shared clone helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioStore.ts'), 'utf8')

    expect(source).toContain("import { deepClone } from '@/utils/deepClone'")
    expect(source).not.toContain('function cloneDocument')
    expect(source).not.toContain('JSON.parse(JSON.stringify')
  })

  it('saves current documents without generic record payload casts', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioStore.ts'), 'utf8')
    const apiSource = readFileSync(resolve(process.cwd(), 'src/api/characterStudio.ts'), 'utf8')

    expect(storeSource).not.toContain('currentDocument.value as unknown as Record<string, unknown>')
    expect(apiSource).toContain('payload: CharacterStudioDocument')
  })

  it('keeps export download transport behind a Studio export helper', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioStore.ts'), 'utf8')
    const exportSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioExports.ts'), 'utf8')

    expect(storeSource).toContain("from '@/stores/characterStudioExports'")
    expect(storeSource).not.toContain('downloadCharacterStudioExport')
    expect(storeSource).not.toContain('downloadCharacterStudioWorldbook')
    expect(storeSource).not.toContain('exportCharacterStudioChatSession')
    expect(storeSource).not.toContain("from '@/utils/browserDownload'")
    expect(exportSource).toContain("import { triggerBlobDownload } from '@/utils/browserDownload'")
  })

  it('keeps busy action copy behind a Studio activity helper', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioStore.ts'), 'utf8')
    const activitySource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioActivity.ts'), 'utf8')

    expect(storeSource).toContain("from '@/stores/characterStudioActivity'")
    expect(storeSource).toContain('getCharacterStudioActionLabel')
    expect(storeSource).toContain('hasCharacterStudioBusyAction')
    expect(storeSource).not.toContain('正在加载角色工坊')
    expect(storeSource).not.toContain('正在生成聊天回复')
    expect(storeSource).not.toContain('正在导出 V3 JSON')
    expect(activitySource).toContain('export function getCharacterStudioActionLabel')
  })

  it('keeps agent output parsing behind a Studio agent output helper', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioStore.ts'), 'utf8')
    const outputSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioAgentOutput.ts'), 'utf8')

    expect(storeSource).toContain("from '@/stores/characterStudioAgentOutput'")
    expect(storeSource).toContain('parseCharacterStudioAgentOutput')
    expect(storeSource).not.toContain('```json:patch')
    expect(storeSource).not.toContain('```html')
    expect(storeSource).not.toContain('content.match')
    expect(outputSource).toContain('export function parseCharacterStudioAgentOutput')
  })

  it('keeps chat stream message mutations behind a Studio chat session helper', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioStore.ts'), 'utf8')
    const chatWorkflowSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudio/useCharacterStudioChat.ts'), 'utf8')
    const chatSessionSource = readFileSync(resolve(process.cwd(), 'src/stores/characterStudioChatSession.ts'), 'utf8')

    expect(storeSource).toContain("from './characterStudio/useCharacterStudioChat'")
    expect(chatWorkflowSource).toContain("from '@/stores/characterStudioChatSession'")
    expect(chatWorkflowSource).toContain('applyAssistantStreamContent')
    expect(chatWorkflowSource).toContain('applyAssistantRuntimeState')
    expect(chatWorkflowSource).toContain('findRegenerationUserMessageIndex')
    expect(chatWorkflowSource).not.toContain('lastMessage.content = event.content')
    expect(chatWorkflowSource).not.toContain('lastMessage.runtime_log = event.runtime_log')
    expect(chatWorkflowSource).not.toContain('messages.findIndex(item => item.message_id === messageId)')
    expect(chatSessionSource).toContain('export function applyAssistantStreamContent')
    expect(chatSessionSource).toContain('export function applyAssistantRuntimeState')
    expect(chatSessionSource).toContain('export function findRegenerationUserMessageIndex')
  })

  it('updates assistant stream messages through the Studio chat session helper', async () => {
    const {
      applyAssistantRuntimeState,
      applyAssistantStreamContent,
      findRegenerationUserMessageIndex,
    } = await import('@/stores/characterStudioChatSession')
    const session = deepClone(conversationChatSession)

    expect(applyAssistantStreamContent(session, '新的流式内容')).toBe(true)
    expect(session.messages.at(-1)?.content).toBe('新的流式内容')

    const runtimeLog = [{ stage: '变量更新' }]
    const variables = { trust_score: 35 }
    expect(applyAssistantRuntimeState(session, runtimeLog, variables)).toBe(true)
    expect(session.messages.at(-1)?.runtime_log).toEqual([{ stage: '变量更新' }])
    expect(session.messages.at(-1)?.variables_snapshot).toEqual({ trust_score: 35 })
    runtimeLog[0]!.stage = '外部复用'
    variables.trust_score = 99
    expect(session.messages.at(-1)?.runtime_log).toEqual([{ stage: '变量更新' }])
    expect(session.messages.at(-1)?.variables_snapshot).toEqual({ trust_score: 35 })
    expect(findRegenerationUserMessageIndex(session.messages, 'msg_assistant_1')).toBe(1)
    expect(findRegenerationUserMessageIndex(session.messages, 'msg_user_1')).toBe(1)
    expect(findRegenerationUserMessageIndex(session.messages, 'missing')).toBe(-1)
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

  it('restores persisted diagnostics and invalidates them on document edits', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    const diagnosedDocument = deepClone(demoDocument) as CharacterStudioDocument
    diagnosedDocument.status.last_diagnostics = {
      valid: true,
      errors: [],
      warnings: ['待确认'],
      checks: { document: true },
    }
    diagnosedDocument.status.last_validated_at = '2026-07-01T00:00:00Z'
    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: diagnosedDocument,
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')
    expect(store.diagnostics).toEqual(diagnosedDocument.status.last_diagnostics)

    store.updateCurrentDocument({
      ...store.currentDocument!,
      identity: {
        ...store.currentDocument!.identity,
        description: '诊断后发生编辑',
      },
    })
    expect(store.diagnostics).toBeNull()
    expect(store.currentDocument?.status.last_diagnostics).toBeNull()
    expect(store.currentDocument?.status.last_validated_at).toBeNull()
  })

  it('ignores stale workspace responses after a newer book load starts', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    let resolveFirst!: (value: Awaited<ReturnType<typeof getCharacterStudioIndexMock>>) => void

    getCharacterStudioIndexMock
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce({
        success: true,
        book_id: 'book-beta',
        documents: [{
          id: 'doc_beta',
          title: '贝塔',
          origin: 'manual',
          source_character: null,
          updated_at: '2026-05-16T00:00:00',
          tags: [],
          is_favorite: false,
          has_avatar: false,
          sample_pages: [],
        }],
        candidates: [],
        count: 1,
      })

    const firstLoad = store.loadWorkspace('book-alpha')
    const secondLoad = store.loadWorkspace('book-beta')
    await secondLoad

    resolveFirst({
      success: true,
      book_id: 'book-alpha',
      documents: [{
        id: 'doc_alpha',
        title: '阿尔法',
        origin: 'manual',
        source_character: null,
        updated_at: '2026-05-15T00:00:00',
        tags: [],
        is_favorite: false,
        has_avatar: false,
        sample_pages: [],
      }],
      candidates: [],
      count: 1,
    })
    await firstLoad

    expect(store.bookId).toBe('book-beta')
    expect(store.documents.map(item => item.id)).toEqual(['doc_beta'])
  })

  it('ignores stale document responses after a newer document open starts', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    let resolveFirst!: (value: Awaited<ReturnType<typeof getCharacterStudioDocumentMock>>) => void
    const betaDocument: CharacterStudioDocument = {
      ...deepClone(demoDocument),
      id: 'doc_beta',
      meta: { ...demoDocument.meta, title: '贝塔' },
      identity: { ...demoDocument.identity, name: '贝塔' },
    }

    getCharacterStudioDocumentMock
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce({
        success: true,
        document: betaDocument,
      })

    await store.loadWorkspace('book-demo')
    const firstOpen = store.openDocument('doc_alpha')
    const secondOpen = store.openDocument('doc_beta')
    await secondOpen

    resolveFirst({
      success: true,
      document: deepClone(demoDocument),
    })
    await firstOpen

    expect(store.currentDocument?.id).toBe('doc_beta')
    expect(store.currentDocument?.identity.name).toBe('贝塔')
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

  it('uses the backend-regenerated session returned by a durable user-message edit', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioChatStateMock.mockResolvedValueOnce({
      success: true,
      doc_id: 'doc_alpha',
      active_session: deepClone(conversationChatSession),
      archived_sessions: [],
      available_greetings: [],
    })

    editCharacterStudioChatMessageMock.mockResolvedValueOnce({
      success: true,
      session: {
        ...deepClone(conversationChatSession),
        messages: [
          deepClone(conversationChatSession.messages[0]!),
          {
            ...deepClone(conversationChatSession.messages[1]!),
            content: '编辑后的用户消息',
            generation_meta: { original_content: '编辑后的用户消息' },
          },
          {
            ...deepClone(conversationChatSession.messages[2]!),
            message_id: 'msg_assistant_regenerated',
            content: '新的回答',
          },
        ],
      },
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')
    await store.editChatMessage('msg_user_1', '编辑后的用户消息')

    expect(editCharacterStudioChatMessageMock).toHaveBeenCalledTimes(1)
    expect(regenerateCharacterStudioChatMessageMock).not.toHaveBeenCalled()
    expect(store.activeChatSession?.messages.map(item => item.content)).toEqual([
      '我是阿尔法。',
      '编辑后的用户消息',
      '新的回答',
    ])
  })

  it('aborts the durable chat operation before disconnecting the local stream', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    const abortedSession = {
      ...deepClone(demoChatSession),
      revision: 3,
      generation: 2,
      messages: [
        ...deepClone(demoChatSession.messages),
        {
          ...deepClone(demoChatSession.messages[0]!),
          message_id: 'msg_user_abort',
          role: 'user' as const,
          content: '保留这条用户消息',
        },
      ],
    }
    abortCharacterStudioChatOperationMock.mockResolvedValueOnce(abortedSession)
    streamCharacterStudioChatMessageMock.mockImplementationOnce(async (
      _bookId: string,
      _docId: string,
      options: {
        onAccepted?: (operationId: string) => void
        signal: AbortSignal
      },
    ) => new Promise<void>((_resolve, reject) => {
      options.onAccepted?.('chat-op-abort')
      options.signal.addEventListener('abort', () => reject(new Error('aborted')))
    }))

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')
    const sendPromise = store.sendChatMessage('保留这条用户消息')
    await Promise.resolve()

    expect(store.activeChatOperationId).toBe('chat-op-abort')
    await store.abortActiveChatOperation()
    await sendPromise

    expect(abortCharacterStudioChatOperationMock).toHaveBeenCalledWith(
      'chat_alpha',
      'chat-op-abort',
    )
    expect(store.isChatStreaming).toBe(false)
    expect(store.activeChatOperationId).toBeNull()
    expect(store.activeChatSession?.messages.at(-1)?.content).toBe('保留这条用户消息')
  })

  it('permanently deletes an archived session with its current revision', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    getCharacterStudioChatStateMock.mockResolvedValueOnce({
      success: true,
      doc_id: 'doc_alpha',
      active_session: deepClone(demoChatSession),
      archived_sessions: [{
        session_id: 'chat_archived',
        title: '旧会话',
        updated_at: '2026-05-14T00:00:00',
        message_count: 3,
        revision: 7,
        generation: 1,
      }],
      available_greetings: [],
    })
    deleteCharacterStudioChatSessionMock.mockResolvedValueOnce({
      success: true,
      doc_id: 'doc_alpha',
      active_session: deepClone(demoChatSession),
      archived_sessions: [],
      available_greetings: [],
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')
    await store.deleteArchivedChatSession('chat_archived', 7)

    expect(deleteCharacterStudioChatSessionMock).toHaveBeenCalledWith(
      'book-demo',
      'doc_alpha',
      'chat_archived',
      7,
    )
    expect(store.archivedChatSessions).toEqual([])
  })

  it('rehydrates chat state after full generation so opening and greetings refresh immediately', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioChatStateMock
      .mockResolvedValueOnce({
        success: true,
        doc_id: 'doc_alpha',
        active_session: {
          ...demoChatSession,
          messages: [],
        },
        archived_sessions: [],
        available_greetings: [],
      })
      .mockResolvedValueOnce({
        success: true,
        doc_id: 'doc_alpha',
        active_session: {
          ...demoChatSession,
          messages: [
            {
              ...demoChatSession.messages[0],
              content: '新的默认开场白',
            },
          ],
        },
        archived_sessions: [],
        available_greetings: [
          {
            greeting_id: 'first_message',
            label: '主问候',
            content: '新的默认开场白',
            source: { type: 'first_message', index: 0 },
          },
          {
            greeting_id: 'alternate_1',
            label: '备用问候 1',
            content: '备用问候',
            source: { type: 'alternate_greetings', index: 0 },
          },
        ],
      })

    generateCharacterStudioSectionMock.mockResolvedValueOnce({
      success: true,
      document: {
        ...deepClone(demoDocument),
        coreMessages: {
          ...deepClone(demoDocument.coreMessages),
          first_message: '新的默认开场白',
          alternate_greetings: ['备用问候'],
        },
      },
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')
    await store.generateSection('full')

    expect(getCharacterStudioChatStateMock).toHaveBeenCalledTimes(2)
    expect(store.activeChatSession?.messages[0]?.content).toBe('新的默认开场白')
    expect(buildCharacterStudioGreetingOptions(store.currentDocument)).toHaveLength(2)
  })

  it('defers chat rehydrate until streaming finishes when document save happens mid-chat', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    let resolveStream: (() => void) | null = null
    streamCharacterStudioChatMessageMock.mockImplementationOnce(async () => new Promise<void>(resolve => {
      resolveStream = resolve
    }))

    getCharacterStudioChatStateMock
      .mockResolvedValueOnce({
        success: true,
        doc_id: 'doc_alpha',
        active_session: deepClone(demoChatSession),
        archived_sessions: [],
        available_greetings: [
          {
            greeting_id: 'first_message',
            label: '主问候',
            content: '我是阿尔法。',
            source: { type: 'first_message', index: 0 },
          },
        ],
      })
      .mockResolvedValueOnce({
        success: true,
        doc_id: 'doc_alpha',
        active_session: {
          ...deepClone(demoChatSession),
          messages: [
            {
              ...deepClone(demoChatSession.messages[0]!),
              content: '保存后同步的新开场',
            },
          ],
        },
        archived_sessions: [],
        available_greetings: [
          {
            greeting_id: 'first_message',
            label: '主问候',
            content: '保存后同步的新开场',
            source: { type: 'first_message', index: 0 },
          },
        ],
      })

    saveCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: {
        ...deepClone(demoDocument),
        coreMessages: {
          ...deepClone(demoDocument.coreMessages),
          first_message: '保存后同步的新开场',
        },
      },
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const sendPromise = store.sendChatMessage('先别刷新我')
    await Promise.resolve()

    expect(store.isChatStreaming).toBe(true)

    await store.persistCurrentDocument()

    expect(getCharacterStudioChatStateMock).toHaveBeenCalledTimes(1)

    if (resolveStream) {
      resolveStream()
    }
    await sendPromise

    expect(getCharacterStudioChatStateMock).toHaveBeenCalledTimes(2)
    expect(store.activeChatSession?.messages[0]?.content).toBe('保存后同步的新开场')
  })

  it('freezes optimistic chat message variable snapshots while sending', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    const loadedSession = deepClone(demoChatSession)
    loadedSession.variables = {
      trust_score: 20,
      mood: { intensity: 1 },
    }
    let resolveStream: (() => void) | null = null

    getCharacterStudioChatStateMock.mockResolvedValueOnce({
      success: true,
      doc_id: 'doc_alpha',
      active_session: loadedSession,
      archived_sessions: [],
      available_greetings: [],
    })
    streamCharacterStudioChatMessageMock.mockImplementationOnce(async () => new Promise<void>(resolve => {
      resolveStream = resolve
    }))

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const sourceVariables = store.activeChatSession?.variables as {
      trust_score: number
      mood: { intensity: number }
    }
    const sendPromise = store.sendChatMessage('记录当前变量')
    await Promise.resolve()

    sourceVariables.trust_score = 77
    sourceVariables.mood.intensity = 9

    const optimisticMessages = store.activeChatSession?.messages.slice(-2) || []
    expect(optimisticMessages[0]?.variables_snapshot).toEqual({
      trust_score: 20,
      mood: { intensity: 1 },
    })
    expect(optimisticMessages[1]?.variables_snapshot).toEqual({
      trust_score: 20,
      mood: { intensity: 1 },
    })

    resolveStream?.()
    await sendPromise
  })

  it('releases optimistic attachment URLs when workspace reset aborts streaming chat', async () => {
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation(file => `blob:${(file as File).name}`)
    const revokeObjectURLSpy = vi
      .spyOn(URL, 'revokeObjectURL')
      .mockImplementation(() => {})
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    streamCharacterStudioChatMessageMock.mockImplementationOnce(async (
      _bookId: string,
      _docId: string,
      options: { signal: AbortSignal },
    ) => new Promise<void>((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new Error('aborted')))
    }))

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const file = new File(['image'], 'panel.png', { type: 'image/png' })
    const sendPromise = store.sendChatMessage('看这张图', [file])
    await Promise.resolve()

    expect(createObjectURLSpy).toHaveBeenCalledWith(file)

    getCharacterStudioIndexMock.mockResolvedValueOnce({
      success: true,
      book_id: 'book-other',
      documents: [],
      candidates: [],
      count: 0,
    })

    await store.loadWorkspace('book-other')
    await sendPromise

    expect(store.activeChatSession).toBeNull()
    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:panel.png')

    createObjectURLSpy.mockRestore()
    revokeObjectURLSpy.mockRestore()
  })

  it('releases optimistic attachment URLs when a new send supersedes active streaming chat', async () => {
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation(file => `blob:${(file as File).name}`)
    const revokeObjectURLSpy = vi
      .spyOn(URL, 'revokeObjectURL')
      .mockImplementation(() => {})
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    let secondSend: Promise<void> | null = null

    streamCharacterStudioChatMessageMock
      .mockImplementationOnce(async (
        _bookId: string,
        _docId: string,
        options: { signal: AbortSignal },
      ) => new Promise<void>((_resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new Error('aborted')))
      }))
      .mockImplementationOnce(async (
        _bookId: string,
        _docId: string,
        options: { signal: AbortSignal },
      ) => new Promise<void>((_resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new Error('aborted')))
      }))

    try {
      await store.loadWorkspace('book-demo')
      await store.openDocument('doc_alpha')

      const file = new File(['image'], 'superseded.png', { type: 'image/png' })
      const firstSend = store.sendChatMessage('第一条带图消息', [file])
      await Promise.resolve()

      secondSend = store.sendChatMessage('第二条消息')
      await Promise.resolve()
      await firstSend

      expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:superseded.png')
    } finally {
      if (secondSend) {
        getCharacterStudioIndexMock.mockResolvedValueOnce({
          success: true,
          book_id: 'book-other',
          documents: [],
          candidates: [],
          count: 0,
        })
        await store.loadWorkspace('book-other')
        await secondSend
      }
      createObjectURLSpy.mockRestore()
      revokeObjectURLSpy.mockRestore()
    }
  })

  it('ignores late chat stream state events after the workspace changes', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()
    let emitStaleEvent: ((event: CharacterStudioChatStreamEvent) => void) | null = null
    let resolveStream: (() => void) | null = null

    streamCharacterStudioChatMessageMock.mockImplementationOnce(async (
      _bookId: string,
      _docId: string,
      options: {
        onEvent: (event: CharacterStudioChatStreamEvent) => void
        signal: AbortSignal
      },
    ) => new Promise<void>(resolve => {
      emitStaleEvent = options.onEvent
      resolveStream = resolve
      options.signal.addEventListener('abort', () => {})
    }))

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const sendPromise = store.sendChatMessage('这条流会过期')
    await Promise.resolve()

    getCharacterStudioIndexMock.mockResolvedValueOnce({
      success: true,
      book_id: 'book-other',
      documents: [],
      candidates: [],
      count: 0,
    })

    await store.loadWorkspace('book-other')

    expect(store.bookId).toBe('book-other')
    expect(store.activeChatSession).toBeNull()

    emitStaleEvent?.({
      type: 'state',
      session: {
        ...deepClone(demoChatSession),
        messages: [
          {
            ...deepClone(demoChatSession.messages[0]!),
            content: '不应该写回的新状态',
          },
        ],
      },
    })
    resolveStream?.()
    await sendPromise

    expect(store.bookId).toBe('book-other')
    expect(store.activeChatSession).toBeNull()
  })

  it('updates locally derived greeting options and clears stale diagnostics/prompt preview on document edits', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.diagnostics = {
      valid: true,
      errors: [],
      warnings: [],
      checks: {},
    }
    store.chatPromptPreview = '过期提示词缓存'
    store.chatPromptPreviewError = '过期错误'

    if (!store.currentDocument) {
      throw new Error('currentDocument missing in test setup')
    }

    store.updateCurrentDocument({
      ...store.currentDocument,
      coreMessages: {
        ...store.currentDocument.coreMessages,
        first_message: '本地立即可见的主问候',
        alternate_greetings: ['备用问候 A'],
      },
    })

    expect(buildCharacterStudioGreetingOptions(store.currentDocument).map(item => item.content)).toEqual([
      '本地立即可见的主问候',
      '备用问候 A',
    ])
    expect(store.diagnostics).toBeNull()
    expect(store.chatPromptPreview).toBe('')
    expect(store.chatPromptPreviewError).toBe('')
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
    }

    store.applyPendingPatch()

    expect(store.currentDocument?.identity.name).toBe('新名字')
    expect(store.currentDocument?.meta.title).toBe('新名字')
  })

  it('updates a worldbook root entry by id via agent patch', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: deepClone(structuredDocument),
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
    }

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
      document: deepClone(structuredDocument),
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
    }

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.children[0]?.content).toBe('更新后的子条目内容')
    expect(store.currentDocument?.lorebook.entries[0]?.children[0]?.keys).toEqual(['测试', '支线'])

    store.pendingAgentPatch = {
      worldbook_delete: {
        id: 'entry_child',
      },
    }

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.children).toEqual([])
  })

  it('updates and deletes regex and task entries by id via agent patch', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: deepClone(structuredDocument),
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
    }

    store.applyPendingPatch()

    expect(store.currentDocument?.regexScripts[0]?.replaceString).toBe('更新后的替换内容')
    expect(store.currentDocument?.regexScripts[0]?.placement).toEqual([1, 2])
    expect(store.currentDocument?.stateTasks[0]?.interval).toBe(3)
    expect(store.currentDocument?.stateTasks[0]?.commands).toContain('trust_score 40')

    store.pendingAgentPatch = {
      regex_delete: { id: 'regex_alpha' },
      task_delete: { id: 'task_alpha' },
    }

    store.applyPendingPatch()

    expect(store.currentDocument?.regexScripts).toEqual([])
    expect(store.currentDocument?.stateTasks).toEqual([])
  })

  it('keeps pending patch and document state unchanged when patch target id does not exist', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: deepClone(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const before = deepClone(store.currentDocument)

    const missingPatch: CharacterStudioAgentPatchV2 = {
      regex_update: {
        id: 'regex_missing',
        changes: {
          replaceString: '不会生效',
        },
      },
    }

    store.pendingAgentPatch = missingPatch
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
      document: deepClone(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const unsupportedPatch = {
      worldbook_move: {
        id: 'entry_root',
      },
    }

    const before = deepClone(store.currentDocument)
    store.pendingAgentPatch = unsupportedPatch as unknown as CharacterStudioAgentPatchV2
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
      document: deepClone(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const before = deepClone(store.currentDocument)
    const invalidSetPatch: CharacterStudioAgentPatchV2 = {
      set: {
        'regexScripts.0.disabled': true,
      },
    }

    store.pendingAgentPatch = invalidSetPatch
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
      document: deepClone(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const invalidRegexPatch: CharacterStudioAgentPatchV2 = {
      regex_update: {
        id: 'regex_alpha',
        changes: {
          placement: [3],
        },
      },
    }

    const before = deepClone(store.currentDocument)
    store.pendingAgentPatch = invalidRegexPatch
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
      document: deepClone(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    const invalidWorldbookPatch: CharacterStudioAgentPatchV2 = {
      worldbook_update: {
        id: 'entry_root',
        changes: {
          position: 'top_an',
        },
      },
    }

    const before = deepClone(store.currentDocument)
    store.pendingAgentPatch = invalidWorldbookPatch
    store.applyPendingPatch()

    expect(store.currentDocument).toEqual(before)
    expect(store.pendingAgentPatch).toEqual(invalidWorldbookPatch)
    expect(store.errorMessage).toContain('before_char、at_depth、after_char')
  })

  it('skips frozen section operations while applying other valid patch ops', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    const frozenDocument = deepClone(structuredDocument)
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
    }

    store.applyPendingPatch()

    expect(store.currentDocument?.lorebook.entries[0]?.content).toBe('根条目内容')
    expect(store.currentDocument?.regexScripts[0]?.disabled).toBe(true)
  })

  it('can undo a v2 agent patch after applying it', async () => {
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    getCharacterStudioDocumentMock.mockResolvedValueOnce({
      success: true,
      document: deepClone(structuredDocument),
    })

    await store.loadWorkspace('book-demo')
    await store.openDocument('doc_alpha')

    store.pendingAgentPatch = {
      task_delete: {
        id: 'task_alpha',
      },
    }

    store.applyPendingPatch()
    expect(store.currentDocument?.stateTasks).toEqual([])

    store.diagnostics = {
      valid: true,
      errors: [],
      warnings: [],
      checks: {},
    }
    store.chatPromptPreview = '过期提示词缓存'
    store.chatPromptPreviewError = '过期错误'

    store.undoLastPatch()
    expect(store.currentDocument?.stateTasks[0]?.id).toBe('task_alpha')
    expect(store.diagnostics).toBeNull()
    expect(store.chatPromptPreview).toBe('')
    expect(store.chatPromptPreviewError).toBe('')
  })

  it('clears a no-op frozen patch without creating undo state or autosaving', async () => {
    vi.useFakeTimers()
    const { useCharacterStudioStore } = await import('@/stores/characterStudioStore')
    const store = useCharacterStudioStore()

    const frozenDocument = deepClone(structuredDocument)
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
    }

    store.applyPendingPatch()
    await vi.advanceTimersByTimeAsync(1500)

    expect(store.pendingAgentPatch).toBeNull()
    expect(store.canUndoPatch).toBe(false)
    expect(saveCharacterStudioDocumentMock).not.toHaveBeenCalled()
  })
})
