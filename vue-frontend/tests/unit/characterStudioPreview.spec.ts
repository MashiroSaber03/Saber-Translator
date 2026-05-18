import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import CharacterStudioPreview from '@/components/insight/studio/CharacterStudioPreview.vue'
import type { CharacterStudioAgentPatchV2, CharacterStudioChatSession, CharacterStudioDocument } from '@/types/characterStudio'

const documentStub: CharacterStudioDocument = {
  id: 'doc-alpha',
  bookId: 'book-demo',
  origin: {
    type: 'manual',
    source_character: null,
    source_pages: [],
  },
  status: {
    is_favorite: false,
    frozen_sections: [],
    last_validated_at: null,
  },
  meta: {
    title: '阿尔法',
    tags: [],
    created_at: '2026-05-15T00:00:00',
    updated_at: '2026-05-15T00:00:00',
  },
  avatar: {
    mode: 'none',
    asset_path: null,
    source_page: null,
  },
  identity: {
    name: '阿尔法',
    aliases: [],
    description: '测试角色',
    personality: '沉稳',
    scenario: '测试场景',
  },
  coreMessages: {
    first_message: '你好，我是阿尔法。',
    message_example: '<START>',
    alternate_greetings: [],
    system_prompt: '',
    post_history_instructions: '',
    creator_notes: '',
    character_version: '2.0.0',
  },
  lorebook: {
    name: '阿尔法世界书',
    entries: [],
  },
  regexScripts: [],
  stateTasks: [],
  chatPreset: {
    opening_mode: 'first_message',
  },
  grounding: {
    timeline_mode: 'enhanced',
    sample_pages: [],
    relationships: [],
    key_moments: [],
  },
  exportArtifacts: {},
}

const documentWithPatchTargets: CharacterStudioDocument = {
  ...documentStub,
  lorebook: {
    name: '阿尔法世界书',
    entries: [
      {
        id: 'entry_root',
        comment: '世界观设定',
        keys: ['学院'],
        secondary_keys: [],
        content: '旧内容',
        enabled: true,
        constant: false,
        selective: true,
        priority: 100,
        position: 'before_char',
        depth: 4,
        children: [
          {
            id: 'entry_child',
            comment: '支线事件',
            keys: ['祭典'],
            secondary_keys: [],
            content: '子条目',
            enabled: true,
            constant: false,
            selective: true,
            priority: 90,
            position: 'before_char',
            depth: 3,
            children: [],
          },
        ],
      },
    ],
  },
  regexScripts: [
    {
      id: 'regex_alpha',
      scriptName: '隐藏状态块',
      findRegex: '<state>[\\s\\S]*?</state>',
      replaceString: '',
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
      name: '初始化状态',
      triggerTiming: 'initialization',
      interval: 0,
      commands: '<<taskjs>>\n<</taskjs>>',
      disabled: false,
    },
  ],
}

const sessionStub: CharacterStudioChatSession = {
  session_id: 'chat-alpha',
  doc_id: 'doc-alpha',
  title: '新对话',
  created_at: '2026-05-15T00:00:00',
  updated_at: '2026-05-15T00:00:00',
  archived_at: null,
  greeting_source: { type: 'first_message', index: 0 },
  summary_blocks: [],
  messages: [
    {
      message_id: 'msg-open',
      role: 'assistant',
      content: '你好，我是阿尔法。',
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

function mountPreview(overrides: Record<string, unknown> = {}) {
  return mount(CharacterStudioPreview, {
    props: {
      bookId: 'book-demo',
      document: documentStub,
      session: sessionStub,
      archivedSessions: [
        {
          session_id: 'chat-archived',
          title: '旧会话',
          message_count: 5,
          updated_at: '2026-05-15T01:00:00',
          archived_at: '2026-05-15T01:00:00',
          last_message_excerpt: '上一次聊到这里',
        },
      ],
      promptPreview: '',
      promptPreviewError: '',
      activeTab: 'chat',
      chatLoading: false,
      chatStreaming: false,
      chatMutating: false,
      chatSummarizing: false,
      chatExporting: false,
      chatImporting: false,
      chatPromptLoading: false,
      agentBusy: false,
      agentMessages: [],
      pendingPatch: null,
      canUndoPatch: false,
      agentHtmlPreview: '',
      ...overrides,
    },
  })
}

const summaryPatch: CharacterStudioAgentPatchV2 = {
  set: {
    'identity.name': '新阿尔法',
  },
  greeting_add: '今晚继续推进计划。',
  worldbook_update: {
    id: 'entry_root',
    changes: {
      content: '新的世界观摘要',
      priority: 250,
    },
  },
  worldbook_delete: {
    id: 'entry_child',
  },
  regex_add: {
    scriptName: '战斗提示',
    findRegex: '战斗开始',
    replaceString: '<div>提示</div>',
  },
  regex_update: {
    id: 'regex_alpha',
    changes: {
      disabled: true,
      placement: [1, 2],
    },
  },
  task_delete: {
    id: 'task_alpha',
  },
}

describe('CharacterStudioPreview workspace', () => {
  it('renders the chat tab with a compact toolbar instead of the old workspace intro banner', () => {
    const wrapper = mountPreview()

    expect(wrapper.find('.workspace-head').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('聊天工作区')
    expect(wrapper.text()).not.toContain('在同一个区域里完成继续聊天、卡片助手修卡和命中调试。')
    expect(wrapper.text()).not.toContain('切换开场白并新建会话')
    expect(wrapper.text()).toContain('新对话')
    expect(wrapper.text()).toContain('查看提示词')
  })

  it('renders chat / assistant / runtime tabs without old native selectors', () => {
    const wrapper = mountPreview()

    expect(wrapper.text()).toContain('聊天')
    expect(wrapper.text()).toContain('卡片助手')
    expect(wrapper.text()).toContain('运行日志')
    expect(wrapper.find('select').exists()).toBe(false)
  })

  it('keeps undo patch available after patch is applied', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
      canUndoPatch: true,
    })

    const undoButton = wrapper.findAll('button').find(button => button.text().includes('撤销 patch'))
    expect(undoButton).toBeDefined()
    expect((undoButton!.element as HTMLButtonElement).disabled).toBe(false)
  })

  it('uses a full-height assistant workspace instead of the old fixed compact message panel', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
    })

    expect(wrapper.find('.assistant-main').exists()).toBe(true)
    expect(wrapper.find('.messages-panel.compact').exists()).toBe(false)
  })

  it('renders the assistant composer in the same compact style as the chat composer', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
    })

    expect(wrapper.find('.assistant-composer .composer-main').exists()).toBe(true)
    expect(wrapper.get('.assistant-composer textarea').attributes('rows')).toBe('1')

    const sendButton = wrapper.get('[data-testid="assistant-send-trigger"]')
    expect(sendButton.attributes('aria-label')).toBe('发送给助手')
    expect(sendButton.text()).toBe('↗')
    expect(wrapper.text()).not.toContain('发送给助手')
  })

  it('uses a full-height runtime workspace container instead of leaving a loose empty block', () => {
    const wrapper = mountPreview({
      activeTab: 'runtime',
    })

    expect(wrapper.find('.runtime-main').exists()).toBe(true)
    expect(wrapper.find('.runtime-empty-panel').exists()).toBe(true)
  })

  it('disables send button while a chat reply is still being generated', () => {
    const wrapper = mountPreview({
      chatStreaming: true,
    })

    const sendButton = wrapper.get('[data-testid="chat-send-trigger"]')
    expect(sendButton.attributes('aria-label')).toBe('回复生成中...')
    expect((sendButton.element as HTMLButtonElement).disabled).toBe(true)
  })

  it('opens session list panel from the current session button', async () => {
    const wrapper = mountPreview()

    await wrapper.get('[data-testid="session-list-trigger"]').trigger('click')

    expect(wrapper.text()).toContain('旧会话')
    expect(wrapper.text()).toContain('上一次聊到这里')
  })

  it('renders a compact chat composer with icon-only upload and send buttons', () => {
    const wrapper = mountPreview()

    expect(wrapper.find('.composer-main').exists()).toBe(true)
    expect(wrapper.get('.chat-composer-input').attributes('rows')).toBe('1')

    const uploadButton = wrapper.get('[data-testid="chat-upload-trigger"]')
    const sendButton = wrapper.get('[data-testid="chat-send-trigger"]')

    expect(uploadButton.attributes('aria-label')).toBe('添加图片')
    expect(sendButton.attributes('aria-label')).toBe('发送消息')
    expect(uploadButton.text()).toBe('+')
    expect(sendButton.text()).toBe('↗')
    expect(wrapper.text()).not.toContain('添加图片')
    expect(wrapper.text()).not.toContain('发送消息')
  })

  it('opens greeting picker modal and shows greeting content cards', async () => {
    const wrapper = mountPreview({
      document: {
        ...documentStub,
        coreMessages: {
          ...documentStub.coreMessages,
          alternate_greetings: ['今天也一起推进计划吧。'],
        },
      },
    })

    await wrapper.get('[data-testid="greeting-picker-trigger"]').trigger('click')
    await flushPromises()

    expect(document.body.textContent).toContain('重选开场白')
    expect(document.body.textContent).toContain('今天也一起推进计划吧。')
  })

  it('falls back to document-derived greetings when chat-state greetings are still empty', () => {
    const wrapper = mountPreview({
      document: {
        ...documentStub,
        coreMessages: {
          ...documentStub.coreMessages,
          first_message: '新的主问候',
          alternate_greetings: ['新的备用问候'],
        },
      },
      session: {
        ...sessionStub,
        greeting_source: { type: 'first_message', index: 0 },
        messages: [],
      },
    })

    const trigger = wrapper.get('[data-testid="greeting-picker-trigger"]')
    expect((trigger.element as HTMLButtonElement).disabled).toBe(false)
    expect(wrapper.text()).toContain('主问候')
  })

  it('opens prompt preview modal and shows empty state when no prompt is available', async () => {
    const wrapper = mountPreview()

    await wrapper.get('[data-testid="prompt-preview-trigger"]').trigger('click')
    await flushPromises()

    expect(document.body.textContent).toContain('本轮提示词预览')
    expect(document.body.textContent).toContain('请先发送至少一条消息后再查看本轮提示词')
  })

  it('renders a grouped human-readable patch summary instead of only raw json', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
      document: documentWithPatchTargets,
      pendingPatch: summaryPatch,
    })

    expect(wrapper.text()).toContain('待应用 Patch')
    expect(wrapper.text()).toContain('字段更新')
    expect(wrapper.text()).toContain('问候语')
    expect(wrapper.text()).toContain('世界书')
    expect(wrapper.text()).toContain('正则')
    expect(wrapper.text()).toContain('状态任务')
    expect(wrapper.text()).toContain('identity.name → 新阿尔法')
    expect(wrapper.text()).toContain('追加备用问候语：今晚继续推进计划。')
    expect(wrapper.text()).toContain('更新「世界观设定」')
    expect(wrapper.text()).toContain('删除「支线事件」')
    expect(wrapper.text()).toContain('新增「战斗提示」')
    expect(wrapper.text()).toContain('更新「隐藏状态块」')
    expect(wrapper.text()).toContain('删除「初始化状态」')
    expect(wrapper.find('.patch-summary').text()).not.toContain('"worldbook_update"')
  })

  it('keeps raw patch json available in a collapsible details block', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
      document: documentWithPatchTargets,
      pendingPatch: summaryPatch,
    })

    const details = wrapper.find('details.patch-raw-details')
    expect(details.exists()).toBe(true)
    expect(details.text()).toContain('查看原始 JSON')
    expect(details.text()).toContain('"worldbook_update"')
    expect(details.text()).toContain('"regex_update"')
  })
})
