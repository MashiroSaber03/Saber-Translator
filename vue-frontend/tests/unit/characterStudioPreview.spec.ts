import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import CharacterStudioPreview from '@/components/insight/studio/CharacterStudioPreview.vue'
import type { CharacterStudioDocument, PreviewSessionState } from '@/types/characterStudio'

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
  previewState: {
    variables: {},
    messages: [],
  },
  grounding: {
    timeline_mode: 'enhanced',
    sample_pages: [],
    relationships: [],
    key_moments: [],
  },
  exportArtifacts: {},
}

const sessionStub: PreviewSessionState = {
  doc_id: 'doc-alpha',
  messages: [
    { role: 'assistant', content: '你好，我是阿尔法。' },
  ],
  variables: { trust_score: 20 },
  log: [],
}

describe('CharacterStudioPreview layout state', () => {
  it('emits toggle event for collapsible runtime sidebar', async () => {
    const wrapper = mount(CharacterStudioPreview, {
      props: {
        document: documentStub,
        session: sessionStub,
        previewing: false,
        agentBusy: false,
        agentMessages: [],
        pendingPatch: null,
        agentHtmlPreview: '',
        collapsed: false,
        canUndoPatch: false,
      },
    })

    await wrapper.find('[data-testid="toggle-preview"]').trigger('click')

    expect(wrapper.emitted('toggle-collapsed')).toBeTruthy()
  })

  it('keeps undo patch available after patch is applied', () => {
    const wrapper = mount(CharacterStudioPreview, {
      props: {
        document: documentStub,
        session: sessionStub,
        previewing: false,
        agentBusy: false,
        agentMessages: [],
        pendingPatch: null,
        agentHtmlPreview: '',
        collapsed: false,
        canUndoPatch: true,
      },
    })

    const buttons = wrapper.findAll('button')
    const undoButton = buttons.find(button => button.text().includes('撤销 patch'))
    expect(undoButton).toBeDefined()
    expect((undoButton!.element as HTMLButtonElement).disabled).toBe(false)
  })
})
