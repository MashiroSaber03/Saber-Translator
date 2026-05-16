import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import CharacterStudioSidebar from '@/components/insight/studio/CharacterStudioSidebar.vue'

describe('CharacterStudioSidebar pending feedback', () => {
  it('shows loading copy for manual create, import, document open, and candidate creation', () => {
    const wrapper = mount(CharacterStudioSidebar, {
      props: {
        documents: [
          {
            id: 'doc_alpha',
            title: '阿尔法',
            origin: 'manual',
            source_character: null,
            updated_at: '2026-05-15T00:00:00',
            tags: [],
            is_favorite: false,
            has_avatar: false,
            sample_pages: [],
          },
        ],
        candidates: [
          {
            name: '候选角色',
            aliases: [],
            first_appearance: 1,
            dialogue_count: 2,
            has_dialogues: true,
            sample_pages: [1],
          },
        ],
        search: '',
        currentDocumentId: 'doc_alpha',
        hasTimeline: true,
        workspaceLoading: false,
        creatingManual: true,
        importingFile: true,
        openingDocumentId: 'doc_alpha',
        creatingCandidateName: '候选角色',
      },
    })

    expect(wrapper.text()).toContain('新建中...')
    expect(wrapper.text()).toContain('导入中...')
    expect(wrapper.text()).toContain('打开中...')
    expect(wrapper.text()).toContain('创建中...')
    expect(wrapper.text()).toContain('候选仅预填角色名')
  })
})
