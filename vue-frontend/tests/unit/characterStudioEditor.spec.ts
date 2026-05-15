import { beforeEach, describe, expect, it } from 'vitest'
import { defineComponent, h, ref } from 'vue'
import { mount } from '@vue/test-utils'
import CharacterStudioEditor from '@/components/insight/studio/CharacterStudioEditor.vue'
import type { CharacterStudioDocument } from '@/types/characterStudio'

function buildDocument(): CharacterStudioDocument {
  return {
    id: 'doc-alpha',
    bookId: 'book-demo',
    origin: {
      type: 'analysis',
      source_character: '上杉风太郎',
      source_pages: [1, 2, 3],
    },
    status: {
      is_favorite: false,
      frozen_sections: [],
      last_validated_at: null,
    },
    meta: {
      title: '上杉风太郎',
      tags: ['主角', '分析生成'],
      created_at: '2026-05-15T00:00:00',
      updated_at: '2026-05-15T00:00:00',
    },
    avatar: {
      mode: 'none',
      asset_path: null,
      source_page: null,
    },
    identity: {
      name: '上杉风太郎',
      aliases: ['风太郎'],
      description: '一个认真但嘴硬的学生。',
      personality: '冷静，略带防备心。',
      scenario: '当前处于学园日常阶段。',
    },
    coreMessages: {
      first_message: '我是上杉风太郎。',
      message_example: '<START>\n{{user}}: 你好\n{{char}}: 你好。',
      alternate_greetings: ['今天也要继续努力。'],
      system_prompt: '保持角色稳定。',
      post_history_instructions: '保持叙事连续。',
      creator_notes: '测试备注',
      character_version: '2.0.0',
    },
    lorebook: {
      name: '风太郎世界书',
      entries: [],
    },
    regexScripts: [],
    stateTasks: [],
    chatPreset: {
      opening_mode: 'first_message',
    },
    previewState: {
      variables: {},
      messages: [],
    },
    grounding: {
      timeline_mode: 'enhanced',
      sample_pages: [1, 3],
      relationships: [],
      key_moments: [],
    },
    exportArtifacts: {},
  }
}

describe('CharacterStudioEditor tabs', () => {
  let document: CharacterStudioDocument

  beforeEach(() => {
    document = buildDocument()
  })

  function mountHarness() {
    return mount(defineComponent({
      components: { CharacterStudioEditor },
      setup() {
        const currentDocument = ref<CharacterStudioDocument | null>(document)
        const activeTab = ref<'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'>('overview')
        const activeScriptTab = ref<'regex' | 'tasks'>('regex')
        return () => h(CharacterStudioEditor, {
          document: currentDocument.value,
          avatarUrl: '',
          saving: false,
          diagnostics: null,
          activeTab: activeTab.value,
          activeScriptTab: activeScriptTab.value,
          pendingState: {
            generatingSection: null,
            validating: false,
            importingWorldbook: false,
            deleting: false,
            saving: false,
            downloadingFormat: null,
          },
          'onUpdate:document': (value: CharacterStudioDocument | null) => { currentDocument.value = value },
          'onUpdate:activeTab': (value: typeof activeTab.value) => { activeTab.value = value },
          'onUpdate:activeScriptTab': (value: typeof activeScriptTab.value) => { activeScriptTab.value = value },
        })
      },
    }), {
      global: {
        stubs: {
          LorebookTreeEditor: {
            template: '<div class="lorebook-stub">世界书树编辑器</div>',
          },
        },
      },
    })
  }

  it('shows chinese section tabs and defaults to 概览', () => {
    const wrapper = mountHarness()

    expect(wrapper.text()).toContain('概览')
    expect(wrapper.text()).toContain('角色设定')
    expect(wrapper.text()).toContain('问候语')
    expect(wrapper.text()).toContain('脚本任务')
    expect(wrapper.text()).toContain('来源摘要')
  })

  it('preserves edited data when switching tabs', async () => {
    const wrapper = mountHarness()

    await wrapper.find('[data-tab="character"]').trigger('click')
    const description = wrapper.find('textarea')
    await description.setValue('新的角色简介')

    await wrapper.find('[data-tab="greetings"]').trigger('click')
    await wrapper.find('[data-tab="character"]').trigger('click')

    const currentValue = wrapper.find('textarea').element as HTMLTextAreaElement
    expect(currentValue.value).toBe('新的角色简介')
  })

  it('shows persisted review summary when latest review exists', () => {
    document.exportArtifacts = {
      last_review: {
        summary: '建议补强世界书和备用问候。',
        issues: ['世界书覆盖面不足'],
        suggestions: ['增加 2-3 条候选问候语'],
      },
    }

    const wrapper = mountHarness()

    expect(wrapper.text()).toContain('建议补强世界书和备用问候。')
    expect(wrapper.text()).toContain('世界书覆盖面不足')
  })

  it('shows loading copy for section generation and validation actions', async () => {
    const wrapper = mount(defineComponent({
      components: { CharacterStudioEditor },
      setup() {
        return () => h(CharacterStudioEditor, {
          document,
          avatarUrl: '',
          saving: false,
          diagnostics: null,
          activeTab: 'overview',
          activeScriptTab: 'regex',
          pendingState: {
            generatingSection: 'identity',
            validating: true,
            importingWorldbook: false,
            deleting: false,
            saving: false,
            downloadingFormat: null,
          },
        })
      },
    }), {
      global: {
        stubs: {
          LorebookTreeEditor: {
            template: '<div class="lorebook-stub">世界书树编辑器</div>',
          },
        },
      },
    })

    expect(wrapper.text()).toContain('补全中...')
    expect(wrapper.text()).toContain('诊断中...')
  })

  it('renders freeze settings as aligned rows with separate label and control cells', () => {
    const wrapper = mountHarness()

    const freezeItems = wrapper.findAll('.freeze-item')
    expect(freezeItems.length).toBeGreaterThan(0)
    expect(freezeItems[0]?.find('.freeze-item-label').exists()).toBe(true)
    expect(freezeItems[0]?.find('.freeze-item-control').exists()).toBe(true)
  })
})
