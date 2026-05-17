import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import CharacterStudioPreview from '@/components/insight/studio/CharacterStudioPreview.vue'
import type { CharacterStudioChatSession, CharacterStudioDocument } from '@/types/characterStudio'

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
      availableGreetings: [
        {
          greeting_id: 'first_message',
          label: '主问候',
          content: '你好，我是阿尔法。',
          source: { type: 'first_message', index: 0 },
        },
        {
          greeting_id: 'alternate_1',
          label: '备用问候 1',
          content: '今天也一起推进计划吧。',
          source: { type: 'alternate_greetings', index: 0 },
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

  it('disables send button while a chat reply is still being generated', () => {
    const wrapper = mountPreview({
      chatStreaming: true,
    })

    const sendButton = wrapper.findAll('button').find(button => button.text().includes('回复生成中'))
    expect(sendButton).toBeDefined()
    expect((sendButton!.element as HTMLButtonElement).disabled).toBe(true)
  })

  it('opens session list panel from the current session button', async () => {
    const wrapper = mountPreview()

    await wrapper.get('[data-testid="session-list-trigger"]').trigger('click')

    expect(wrapper.text()).toContain('旧会话')
    expect(wrapper.text()).toContain('上一次聊到这里')
  })

  it('opens greeting picker modal and shows greeting content cards', async () => {
    const wrapper = mountPreview()

    await wrapper.get('[data-testid="greeting-picker-trigger"]').trigger('click')
    await flushPromises()

    expect(document.body.textContent).toContain('重选开场白')
    expect(document.body.textContent).toContain('今天也一起推进计划吧。')
  })

  it('opens prompt preview modal and shows empty state when no prompt is available', async () => {
    const wrapper = mountPreview()

    await wrapper.get('[data-testid="prompt-preview-trigger"]').trigger('click')
    await flushPromises()

    expect(document.body.textContent).toContain('本轮提示词预览')
    expect(document.body.textContent).toContain('请先发送至少一条消息后再查看本轮提示词')
  })
})
