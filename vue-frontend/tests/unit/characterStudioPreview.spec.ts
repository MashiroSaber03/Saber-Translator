import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import CharacterStudioPreview from '@/components/insight/studio/CharacterStudioPreview.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { CharacterStudioAgentPatchV2, CharacterStudioChatSession, CharacterStudioDocument } from '@/types/characterStudio'

enableAutoUnmount(afterEach)

afterEach(() => {
  document.body.innerHTML = ''
  document.body.style.overflow = ''
})

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
        content: '初始内容',
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

const conversationSessionStub: CharacterStudioChatSession = {
  ...sessionStub,
  messages: [
    {
      message_id: 'msg-open',
      role: 'assistant',
      content: '你好，我是阿尔法。',
      attachments: [],
      runtime_log: [],
      variables_snapshot: { trust_score: 20 },
      generation_meta: { kind: 'opening' },
      created_at: '2026-05-15T00:00:00',
      updated_at: '2026-05-15T00:00:00',
    },
    {
      message_id: 'msg-user-1',
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
      message_id: 'msg-assistant-1',
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

const attachmentSessionStub: CharacterStudioChatSession = {
  ...conversationSessionStub,
  messages: [
    {
      ...conversationSessionStub.messages[1]!,
      attachments: [
        {
          attachment_id: 'attachment-preview',
          filename: 'scene.png',
          mime_type: 'image/png',
          asset_path: 'attachments/scene.png',
          created_at: '2026-05-15T00:01:00',
        },
      ],
    },
  ],
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
          title: '归档会话',
          message_count: 5,
          revision: 7,
          generation: 1,
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
  it('maps preview owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreview.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toContain('--character-studio-preview-card-background')
    expect(source).not.toContain('--character-studio-preview-primary-action-background')
    expect(source).not.toContain('--character-studio-preview-primary-action-shadow')
  })

  it('keeps Studio preview visual styling local without parent token warehouses', () => {
    const previewSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreview.vue'),
      'utf8',
    )
    const sessionToolbarSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/SessionToolbar.vue'),
      'utf8',
    )
    const composerSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/ChatComposer.vue'),
      'utf8',
    )
    const messageSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/MessageList.vue'),
      'utf8',
    )
    const agentSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/AgentWorkspace.vue'),
      'utf8',
    )
    const runtimeSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/RuntimeWorkspace.vue'),
      'utf8',
    )
    const modalsSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreviewModals.vue'),
      'utf8',
    )

    expect(previewSource).not.toMatch(/--character-studio-preview-/)
    expect(sessionToolbarSource).not.toMatch(/--character-studio-preview-/)
    expect(composerSource).not.toMatch(/--character-studio-preview-/)
    expect(messageSource).not.toMatch(/--character-studio-preview-/)
    expect(agentSource).not.toMatch(/--character-studio-preview-/)
    expect(runtimeSource).not.toMatch(/--character-studio-preview-/)
    expect(modalsSource).not.toMatch(/--character-studio-preview-/)

    const childTokenWarehousePattern = /--(?:agent-workspace|runtime-workspace|session-toolbar|studio-chat-composer|studio-message-list|studio-preview-workspace-header)-/
    for (const [file, source] of [
      ['AgentWorkspace.vue', agentSource],
      ['ChatComposer.vue', composerSource],
      ['MessageList.vue', messageSource],
      ['RuntimeWorkspace.vue', runtimeSource],
      ['SessionToolbar.vue', sessionToolbarSource],
    ] as const) {
      expect(source, file).not.toMatch(childTokenWarehousePattern)
    }
  })

  it('renders the chat tab with the compact toolbar contract', () => {
    const wrapper = mountPreview()

    expect(wrapper.find('.workspace-head').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('聊天工作区')
    expect(wrapper.text()).not.toContain('在同一个区域里完成继续聊天、卡片助手修卡和命中调试。')
    expect(wrapper.text()).not.toContain('切换开场白并新建会话')
    expect(wrapper.text()).toContain('新对话')
    expect(wrapper.text()).toContain('查看提示词')
  })

  it('uses a product action row for session toolbar actions', () => {
    const wrapper = mountPreview()

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('ariaLabel')).toBe('聊天会话操作')
    expect(actionRow.props('justify')).toBe('start')
    expect(actionRow.props('variant')).toBe('toolbar')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/SessionToolbar.vue'),
      'utf8',
    )
    expect(source).not.toContain('toolbar-buttons')
    expect(source).not.toContain('action-ghost')
    expect(source).not.toContain('document.querySelector')
    expect(source).not.toMatch(/\.session-actions\s*\{[\s\S]*--ui-button-/)
  })

  it('keeps SessionToolbar internal hooks owned by the session toolbar owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/SessionToolbar.vue'),
      'utf8',
    )

    expect(source).toContain('class="session-toolbar"')
    expect(source).toContain('session-toolbar__triggers')
    expect(source).toContain('session-toolbar__trigger')
    expect(source).toContain('session-toolbar__session-list')
    expect(source).toContain('session-toolbar__session-item')
    expect(source).toContain('session-toolbar__trigger-title')
    expect(source).toContain('session-toolbar__session-title')
    expect(source).toContain('session-toolbar__session-excerpt')
    expect(source).toContain('session-toolbar__actions')
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))
    for (const legacyClass of [
      'session-triggers',
      'trigger-stack',
      'trigger-stack-wide',
      'session-trigger',
      'session-trigger-inline',
      'trigger-copy',
      'trigger-copy-inline',
      'trigger-tag',
      'trigger-meta',
      'trigger-arrow',
      'session-list-panel',
      'session-list-item',
      'item-main',
      'item-meta',
      'item-badge',
      'session-actions',
    ]) {
      expect(classTokens).not.toContain(legacyClass)
    }
    expect(source).not.toMatch(/\.(?:session-triggers|trigger-stack|trigger-stack-wide|session-trigger|trigger-copy|trigger-tag|trigger-meta|trigger-arrow|session-list-panel|session-list-item|item-main|item-meta|item-badge|session-actions)\b/)
    expect(source).not.toContain('.session-toolbar__trigger-copy strong')
    expect(source).not.toContain('.session-toolbar__session-item-main strong')
    expect(source).not.toContain('.session-toolbar__session-item-main p')
  })

  it('exposes the session selector popup relationship to assistive technology', async () => {
    const wrapper = mountPreview()
    const trigger = wrapper.get('[data-testid="session-list-trigger"]')

    expect(trigger.attributes('aria-haspopup')).toBe('menu')
    expect(trigger.attributes('aria-controls')).toBe('studio-session-list-panel')
    expect(trigger.attributes('aria-expanded')).toBe('false')

    await trigger.trigger('click')

    expect(trigger.attributes('aria-expanded')).toBe('true')
    const panel = wrapper.get('#studio-session-list-panel')
    expect(panel.attributes('role')).toBe('menu')
    expect(panel.attributes('aria-label')).toBe('聊天会话列表')
    expect(panel.findAll('[role="menuitem"]').length).toBeGreaterThan(0)
  })

  it('sizes the session selector popup from the toolbar container instead of the viewport', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/SessionToolbar.vue'),
      'utf8',
    )
    const sessionListStyle = source.match(/\.session-toolbar__session-list \{(?<body>[\s\S]*?)\n\}/)
      ?.groups?.body ?? ''

    expect(sessionListStyle).toContain('width: min(460px, 100%)')
    expect(sessionListStyle).toContain('max-width: 100%')
    expect(sessionListStyle).not.toContain('100vw')
  })

  it('renders Studio chat messages through product message bubbles and action rows', () => {
    const wrapper = mountPreview({
      session: conversationSessionStub,
    })

    const messageBubbles = wrapper.findAllComponents(ProductMessageBubble)
    expect(messageBubbles).toHaveLength(conversationSessionStub.messages.length)
    expect(messageBubbles.map(bubble => bubble.props('role'))).toEqual(
      conversationSessionStub.messages.map(message => message.role),
    )
    expect(messageBubbles.every(bubble => bubble.props('appearance') === 'reading')).toBe(true)
    expect(wrapper.findAllComponents(ProductActionRow).length).toBeGreaterThanOrEqual(3)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/MessageList.vue'),
      'utf8',
    )

    expect(source).toContain('ProductMessageBubble')
    expect(source).toContain('ProductActionRow')
    expect(source).toContain('variant="toolbar"')
    expect(source).not.toContain('message-card')
    expect(source).not.toContain('action-ghost')
    expect(source).not.toContain('action-primary')
    expect(source).not.toMatch(/<UiButton\b(?=[^>]*variant="toolbar")/)
    expect(source).not.toMatch(/\.(?:message-actions|editor-actions)\s*\{[\s\S]*--ui-button-/)
  })

  it('renders chat / assistant / runtime tabs through public tab controls', () => {
    const wrapper = mountPreview()

    expect(wrapper.text()).toContain('聊天')
    expect(wrapper.text()).toContain('卡片助手')
    expect(wrapper.text()).toContain('运行日志')
    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs).toHaveLength(3)
    expect(tabs[0]?.attributes('aria-selected')).toBe('true')
    expect(wrapper.getComponent(ProductSegmentedTabs).props('layout')).toBe('wrap')
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

  it('uses a full-height assistant workspace with a scrollable message panel', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
    })

    expect(wrapper.find('.agent-workspace__main').exists()).toBe(true)
    expect(wrapper.find('.agent-workspace__messages--compact').exists()).toBe(false)
  })

  it('renders assistant workspace messages and actions through product primitives', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
      agentMessages: [
        { role: 'user', content: '帮我检查角色卡。' },
        { role: 'assistant', content: '可以，建议补充世界书。' },
      ],
    })

    const bubbles = wrapper.findAllComponents(ProductMessageBubble)
    expect(bubbles).toHaveLength(2)
    expect(bubbles.map(bubble => bubble.props('role'))).toEqual(['user', 'assistant'])
    expect(wrapper.findAllComponents(ProductActionRow).length).toBeGreaterThanOrEqual(2)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/AgentWorkspace.vue'),
      'utf8',
    )
    expect(source).toContain('ProductMessageBubble')
    expect(source).toContain('ProductActionRow')
    expect(source).toContain('variant="toolbar"')
    expect(source).not.toContain('message-card')
    expect(source).not.toContain('action-ghost')
    expect(source).not.toContain('action-primary')
    expect(source).not.toMatch(/<UiButton\b(?=[^>]*variant="toolbar")/)
    expect(source).not.toMatch(/\.assistant-actions\s*\{[\s\S]*--ui-button-/)
  })

  it('renders the assistant composer in the same compact style as the chat composer', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
    })

    expect(wrapper.find('.agent-workspace__composer .agent-workspace__composer-main').exists()).toBe(true)
    expect(wrapper.get('.agent-workspace__composer textarea').attributes('rows')).toBe('1')

    const sendButton = wrapper.get('[data-testid="assistant-send-trigger"]')
    expect(sendButton.attributes('aria-label')).toBe('发送给助手')
    expect(sendButton.getComponent(UiIcon).props('name')).toBe('send')
    expect(sendButton.text()).not.toContain('↗')
    expect(wrapper.text()).not.toContain('发送给助手')
  })

  it('uses a full-height runtime workspace container instead of leaving a loose empty block', () => {
    const wrapper = mountPreview({
      activeTab: 'runtime',
    })

    expect(wrapper.find('.runtime-workspace__main').exists()).toBe(true)
    expect(wrapper.find('.runtime-workspace__empty-panel').exists()).toBe(true)
  })

  it('keeps preview grids responsive inside resizable panes', () => {
    const runtimeSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/RuntimeWorkspace.vue'),
      'utf8',
    )
    const composerSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/ChatComposer.vue'),
      'utf8',
    )
    const messageSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/MessageList.vue'),
      'utf8',
    )

    expect(runtimeSource).toContain('repeat(auto-fit, minmax(min(100%, 280px), 1fr))')
    expect(runtimeSource).not.toContain('repeat(2, minmax(0, 1fr))')
    expect(composerSource).toContain('repeat(auto-fill, minmax(min(100%, 180px), 1fr))')
    expect(composerSource).not.toContain('repeat(auto-fill, minmax(180px, 1fr))')
    expect(messageSource).toContain('repeat(auto-fill, minmax(min(100%, 110px), 1fr))')
    expect(messageSource).not.toContain('repeat(auto-fill, minmax(110px, 1fr))')
  })

  it('keeps preview text and layout contracts safe for narrow split panes', () => {
    const agentSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/AgentWorkspace.vue'),
      'utf8',
    )
    const composerSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/ChatComposer.vue'),
      'utf8',
    )
    const messageSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/MessageList.vue'),
      'utf8',
    )

    expect(agentSource).toContain('(event: \'update:agentInput\', value: string): void')
    expect(agentSource).toMatch(/\.agent-workspace__message-text \{[\s\S]*overflow-wrap: anywhere/)
    expect(agentSource).toMatch(/\.agent-workspace__patch-summary-list \{[\s\S]*overflow-wrap: anywhere/)
    expect(messageSource).toMatch(/\.studio-message-list__body \{[\s\S]*overflow-wrap: anywhere/)
    expect(composerSource).not.toMatch(/\.studio-chat-composer__pending-files \{[\s\S]*flex-wrap:/)
  })

  it('uses a shared Studio preview workspace panel shell', () => {
    const workspaceFiles = [
      'ChatWorkspace.vue',
      'AgentWorkspace.vue',
      'RuntimeWorkspace.vue',
    ]

    for (const file of workspaceFiles) {
      const source = readFileSync(
        resolve(process.cwd(), `src/components/insight/studio/preview/${file}`),
        'utf8',
      )

      expect(source).toContain('StudioPreviewWorkspacePanel')
      expect(source).not.toMatch(/<section class="workspace-card/)
      expect(source).not.toMatch(/\.workspace-card\s*\{/)
    }
  })

  it('uses Studio-owned preview shell and workspace headers', () => {
    const previewSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreview.vue'),
      'utf8',
    )
    const agentSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/AgentWorkspace.vue'),
      'utf8',
    )
    const runtimeSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/RuntimeWorkspace.vue'),
      'utf8',
    )

    expect(previewSource).toContain('class="character-studio-preview"')
    expect(previewSource).not.toContain('chat-shell')
    expect(agentSource).toContain('StudioPreviewWorkspaceHeader')
    expect(runtimeSource).toContain('StudioPreviewWorkspaceHeader')
    expect(agentSource).not.toContain('assistant-head')
    expect(runtimeSource).not.toContain('assistant-head')
  })

  it('keeps Studio preview child hooks owner-prefixed', () => {
    const previewChildFiles = [
      'AgentWorkspace.vue',
      'ChatComposer.vue',
      'MessageList.vue',
      'RuntimeWorkspace.vue',
    ]
    const legacyHookPattern = /\.(?:messages-panel|composer-card|composer-main|composer-actions|compact-actions|message-role|message-body|editor-row|editor-actions|attachment-grid|attachment-card|attachment-frame|attachment-info|pending-files|pending-image-card|pending-image-thumb|pending-image-copy|pending-remove|prompt-preview-card|html-preview-card|patch-summary|patch-summary-section|patch-summary-head|patch-summary-list|patch-raw-details|preview-frame|log-list|log-item)\b/

    for (const file of previewChildFiles) {
      const source = readFileSync(
        resolve(process.cwd(), `src/components/insight/studio/preview/${file}`),
        'utf8',
      )

      expect(source, file).not.toMatch(legacyHookPattern)
    }

    const agentSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/AgentWorkspace.vue'),
      'utf8',
    )
    const composerSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/ChatComposer.vue'),
      'utf8',
    )
    const messageSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/MessageList.vue'),
      'utf8',
    )
    const runtimeSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/RuntimeWorkspace.vue'),
      'utf8',
    )
    const modalSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreviewModals.vue'),
      'utf8',
    )

    expect(agentSource).toContain('agent-workspace__messages')
    expect(agentSource).toContain('agent-workspace__patch-card-title')
    expect(agentSource).toContain('agent-workspace__patch-summary-title')
    expect(agentSource).toContain('agent-workspace__patch-summary-count')
    expect(agentSource).toContain('agent-workspace__patch-raw-summary')
    expect(agentSource).toContain('agent-workspace__patch-raw-json')
    expect(agentSource).not.toContain('.agent-workspace__patch-card h4')
    expect(agentSource).not.toContain('.agent-workspace__html-preview-card h4')
    expect(agentSource).not.toContain('.agent-workspace__patch-summary-head strong')
    expect(agentSource).not.toContain('.agent-workspace__patch-summary-head span')
    expect(agentSource).not.toContain('.agent-workspace__patch-raw-details summary')
    expect(agentSource).not.toContain('.agent-workspace__patch-card pre')
    expect(composerSource).toContain('studio-chat-composer__pending-files')
    expect(composerSource).toContain('studio-chat-composer__pending-image')
    expect(composerSource).toContain('studio-chat-composer__pending-name')
    expect(composerSource).toContain('studio-chat-composer__pending-type')
    expect(composerSource).not.toContain('.studio-chat-composer__pending-thumb img')
    expect(composerSource).not.toContain('.studio-chat-composer__pending-copy strong')
    expect(composerSource).not.toContain('.studio-chat-composer__pending-copy span')
    expect(messageSource).toContain('studio-message-list__attachment-card')
    expect(messageSource).toContain('studio-message-list__attachment-image')
    expect(messageSource).toContain('studio-message-list__attachment-name')
    expect(messageSource).toContain('studio-message-list__attachment-type')
    expect(messageSource).not.toContain('.studio-message-list__attachment-card img')
    expect(messageSource).not.toContain('.studio-message-list__attachment-info strong')
    expect(messageSource).not.toContain('.studio-message-list__attachment-info span')
    expect(runtimeSource).toContain('runtime-workspace__log-list')
    expect(runtimeSource).toContain('runtime-workspace__card-title')
    expect(runtimeSource).toContain('runtime-workspace__card-code')
    expect(runtimeSource).not.toContain('.runtime-workspace__card h5')
    expect(runtimeSource).not.toContain('.runtime-workspace__card pre')
    expect(modalSource).toContain('character-studio-preview-modals__copy-text')
    expect(modalSource).toContain('character-studio-preview-modals__prompt-preview')
    expect(modalSource).toContain('character-studio-preview-modals__image-preview')
    expect(modalSource).toContain('character-studio-preview-modals__image')
    expect(modalSource).not.toContain('.character-studio-preview-modals__copy p')
    expect(modalSource).not.toContain('.character-studio-preview-modals__prompt-body pre')
    expect(modalSource).not.toContain('.character-studio-preview-modals__image-body img')
  })

  it('renders preview empty states through compact product empty states', () => {
    for (const file of [
      'src/components/insight/studio/preview/ChatWorkspace.vue',
      'src/components/insight/studio/preview/MessageList.vue',
      'src/components/insight/studio/preview/AgentWorkspace.vue',
      'src/components/insight/studio/preview/RuntimeWorkspace.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain('ProductEmptyState')
      expect(source, file).not.toContain('empty-copy')
    }

    const noDocument = mountPreview({
      document: null,
      session: null,
    })
    expect(noDocument.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'users',
      role: 'note',
      size: 'compact',
      title: '选择角色文档后可开始聊天',
    })

    const noSession = mountPreview({
      session: null,
    })
    expect(noSession.getComponent(ProductEmptyState).props('title')).toBe('当前还没有聊天会话')

    const emptySession = mountPreview({
      session: {
        ...sessionStub,
        messages: [],
      },
    })
    expect(emptySession.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'message',
      size: 'compact',
      title: '当前会话还没有消息',
    })

    const assistant = mountPreview({
      activeTab: 'assistant',
      agentMessages: [],
    })
    expect(assistant.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'sparkles',
      size: 'compact',
      title: '还没有与卡片助手对话',
    })

    const runtime = mountPreview({
      activeTab: 'runtime',
    })
    expect(runtime.getComponent(ProductEmptyState).props()).toMatchObject({
      iconName: 'bar-chart',
      size: 'compact',
      title: '发送消息后查看运行结果',
    })
  })

  it('offers a backend abort action while a chat reply is still being generated', async () => {
    const wrapper = mountPreview({
      chatStreaming: true,
      chatAbortable: true,
    })

    expect(wrapper.find('[data-testid="chat-send-trigger"]').exists()).toBe(false)
    const abortButton = wrapper.get('[data-testid="chat-abort-trigger"]')
    expect(abortButton.attributes('aria-label')).toBe('中止本次生成')
    expect(abortButton.getComponent(UiIcon).props('name')).toBe('square')
    expect((abortButton.element as HTMLButtonElement).disabled).toBe(false)

    await abortButton.trigger('click')
    expect(wrapper.emitted('abort-chat')).toHaveLength(1)
  })

  it('opens session list panel from the current session button', async () => {
    const wrapper = mountPreview()
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/SessionToolbar.vue'),
      'utf8',
    )

    expect(source).not.toContain('>▾<')
    expect(wrapper.findAll('.session-toolbar__trigger-arrow').map(arrow => arrow.getComponent(UiIcon).props('name'))).toEqual([
      'chevron-down',
      'chevron-down',
    ])

    await wrapper.get('[data-testid="session-list-trigger"]').trigger('click')

    expect(wrapper.text()).toContain('归档会话')
    expect(wrapper.text()).toContain('上一次聊到这里')
  })

  it('exposes permanent deletion for archived sessions without activating them', async () => {
    const wrapper = mountPreview()
    await wrapper.get('[data-testid="session-list-trigger"]').trigger('click')

    await wrapper.get('[aria-label="永久删除归档会话：归档会话"]').trigger('click')

    expect(wrapper.emitted('delete-session')?.[0]?.[0]).toMatchObject({
      session_id: 'chat-archived',
      revision: 7,
    })
    expect(wrapper.emitted('switch-chat-session')).toBeUndefined()
  })

  it('renders the empty archived-session list through product status feedback', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/SessionToolbar.vue'),
      'utf8',
    )

    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('session-list-empty')

    const wrapper = mountPreview({ archivedSessions: [] })

    await wrapper.get('[data-testid="session-list-trigger"]').trigger('click')

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      iconName: 'message',
      role: 'note',
      tone: 'neutral',
      title: '暂无归档会话',
    })
    expect(wrapper.text()).toContain('还没有归档会话')
  })

  it('renders a compact chat composer with icon-only upload and send buttons', () => {
    const wrapper = mountPreview()

    expect(wrapper.find('.studio-chat-composer__main').exists()).toBe(true)
    expect(wrapper.get('.studio-chat-composer__input').attributes('rows')).toBe('1')

    const uploadButton = wrapper.get('[data-testid="chat-upload-trigger"]')
    const sendButton = wrapper.get('[data-testid="chat-send-trigger"]')

    expect(uploadButton.attributes('aria-label')).toBe('添加图片')
    expect(sendButton.attributes('aria-label')).toBe('发送消息')
    expect(uploadButton.getComponent(UiIcon).props('name')).toBe('plus')
    expect(sendButton.getComponent(UiIcon).props('name')).toBe('send')
    expect(uploadButton.text()).not.toContain('+')
    expect(sendButton.text()).not.toContain('↗')
    expect(wrapper.text()).not.toContain('添加图片')
    expect(wrapper.text()).not.toContain('发送消息')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/preview/ChatComposer.vue'),
      'utf8',
    )
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('action-ghost')
    expect(source).not.toContain('action-primary')
  })

  it('labels preview textareas when no visible field label is present', async () => {
    const chatWrapper = mountPreview()
    expect(chatWrapper.get('.studio-chat-composer__input').attributes('aria-label')).toBe('聊天消息内容')

    const assistantWrapper = mountPreview({
      activeTab: 'assistant',
    })
    expect(assistantWrapper.get('.agent-workspace__composer-input').attributes('aria-label')).toBe('卡片助手消息内容')

    const messageWrapper = mountPreview({
      session: conversationSessionStub,
    })
    const userCard = messageWrapper.findAll('[data-testid="studio-chat-message"]').find(card => card.text().includes('今天情况怎么样？'))
    expect(userCard).toBeDefined()

    await userCard!.find('button').trigger('click')

    expect(userCard!.get('textarea').attributes('aria-label')).toBe('编辑聊天消息内容')
  })

  it('does not assert shared icon-button primitives through internal class names', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'tests/unit/characterStudioPreview.spec.ts'),
      'utf8',
    )
    const iconButtonClassPrefix = 'ui-' + 'icon-button--'

    expect(source).not.toContain(iconButtonClassPrefix)
  })

  it('keeps preview composer and attachment actions on product primitives', () => {
    const chatWrapper = mountPreview({
      session: attachmentSessionStub,
    })

    const uploadButton = chatWrapper.get('[data-testid="chat-upload-trigger"]')
    const sendButton = chatWrapper.get('[data-testid="chat-send-trigger"]')

    expect(uploadButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '添加图片',
      size: 'lg',
      variant: 'soft',
    })
    expect(sendButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '发送消息',
      size: 'lg',
      variant: 'primary',
    })
    expect(chatWrapper.findAllComponents(UiIconButton).length).toBeGreaterThanOrEqual(2)
    expect(chatWrapper.findAllComponents(ProductRecordCard).length).toBeGreaterThanOrEqual(1)

    const assistantWrapper = mountPreview({
      activeTab: 'assistant',
      document: documentStub,
      agentInput: '帮我检查角色卡',
    })
    const assistantSendButton = assistantWrapper.get('[data-testid="assistant-send-trigger"]')
    expect(assistantSendButton.getComponent(UiIconButton).props()).toMatchObject({
      label: '发送给助手',
      size: 'lg',
      variant: 'primary',
    })

    for (const file of [
      'src/components/insight/studio/preview/ChatComposer.vue',
      'src/components/insight/studio/preview/AgentWorkspace.vue',
      'src/components/insight/studio/preview/MessageList.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('--ui-button-')
    }
  })

  it('uses the studio textarea variant instead of preview-local primitive skins', () => {
    for (const file of [
      'src/components/insight/studio/preview/ChatComposer.vue',
      'src/components/insight/studio/preview/AgentWorkspace.vue',
      'src/components/insight/studio/preview/MessageList.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain('variant="studio"')
      expect(source, file).not.toMatch(/--ui-textarea-/)
    }
  })

  it('uses the typed file-input primitive boundary for chat attachments and session imports', () => {
    for (const file of [
      'src/components/insight/studio/preview/ChatComposer.vue',
      'src/components/insight/studio/preview/SessionToolbar.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain('@files-change=')
      expect(source, file).not.toContain('ref<HTMLInputElement')
      expect(source, file).not.toContain('event.target as HTMLInputElement')
      expect(source, file).not.toContain('target.files')
      expect(source, file).not.toContain("target.value = ''")
    }
  })

  it('shows user-message editing as an edit-and-regenerate action and uses clearer rollback labels', async () => {
    const wrapper = mountPreview({
      session: conversationSessionStub,
    })

    const userCard = wrapper.findAll('[data-testid="studio-chat-message"]').find(card => card.text().includes('今天情况怎么样？'))
    expect(userCard).toBeDefined()
    expect(userCard!.text()).toContain('编辑')
    expect(userCard!.text()).toContain('从这里回退')
    expect(userCard!.text()).not.toContain('重新生成')

    await userCard!.find('button').trigger('click')

    expect(userCard!.text()).toContain('保存并重新生成')
    expect(wrapper.text()).not.toContain('删除')
    expect(wrapper.text()).not.toContain('重生')
  })

  it('shows regenerate on assistant replies instead of allowing direct editing', () => {
    const wrapper = mountPreview({
      session: conversationSessionStub,
    })

    const assistantCard = wrapper.findAll('[data-testid="studio-chat-message"]').find(card => card.text().includes('局势暂时稳定'))
    expect(assistantCard).toBeDefined()
    expect(assistantCard!.text()).toContain('重新生成')
    expect(assistantCard!.text()).toContain('从这里回退')
    expect(assistantCard!.text()).not.toContain('编辑')
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

  it('uses product action and choice primitives for preview modals', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreviewModals.vue'),
      'utf8',
    )

    expect(source).toContain('ProductActionRow')
    expect(source).toContain('ProductChoiceCardGrid')
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('action-ghost')
    expect(source).not.toContain('action-primary')

    for (const oldClass of [
      'modal-copy',
      'greeting-grid',
      'modal-actions',
      'prompt-preview-body',
      'prompt-tools',
      'image-preview-body',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }

    for (const ownerClass of [
      'character-studio-preview-modals__copy',
      'character-studio-preview-modals__greeting-grid',
      'character-studio-preview-modals__actions',
      'character-studio-preview-modals__prompt-body',
      'character-studio-preview-modals__prompt-tools',
      'character-studio-preview-modals__image-preview',
      'character-studio-preview-modals__image',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })

  it('renders preview modal feedback through product status banners', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/studio/CharacterStudioPreviewModals.vue'),
      'utf8',
    )

    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('modal-empty')
    expect(source).not.toContain('modal-loading')

    const wrapper = mountPreview()

    await wrapper.get('[data-testid="prompt-preview-trigger"]').trigger('click')
    await flushPromises()

    expect(document.body.querySelector('.product-status-banner')).not.toBeNull()
    expect(document.body.textContent).toContain('请先发送至少一条消息后再查看本轮提示词')
    expect(wrapper.findComponent(ProductStatusBanner).exists()).toBe(true)
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
    expect(wrapper.find('.agent-workspace__patch-summary').text()).not.toContain('"worldbook_update"')
  })

  it('keeps raw patch json available in a collapsible details block', () => {
    const wrapper = mountPreview({
      activeTab: 'assistant',
      document: documentWithPatchTargets,
      pendingPatch: summaryPatch,
    })

    const details = wrapper.find('details.agent-workspace__patch-raw-details')
    expect(details.exists()).toBe(true)
    expect(details.text()).toContain('查看原始 JSON')
    expect(details.text()).toContain('"worldbook_update"')
    expect(details.text()).toContain('"regex_update"')
  })
})
