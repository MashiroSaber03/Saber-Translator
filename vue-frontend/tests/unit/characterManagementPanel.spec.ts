import { mount } from '@vue/test-utils'
import { defineComponent, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import CharacterManagementPanel from '@/components/insight/continuation/CharacterManagementPanel.vue'

const characterDetailPanelStub = defineComponent({
  props: {
    character: {
      type: Object,
      default: null,
    },
  },
  template: '<div class="character-detail-stub">{{ character?.name || "empty" }}</div>',
})

function createState() {
  return {
    characters: ref([
      {
        name: '主角',
        aliases: [],
        description: 'desc',
        forms: [],
        reference_image: '',
        enabled: true,
      },
    ]),
    getCharacterImageUrl: vi.fn().mockReturnValue(''),
    getFormImageUrl: vi.fn().mockReturnValue(''),
    showMessage: vi.fn(),
  }
}

describe('CharacterManagementPanel', () => {
  it('uses button semantics for selectable character tiles', async () => {
    const wrapper = mount(CharacterManagementPanel, {
      props: {
        bookId: 'book-1',
        characterManagement: {},
        state: createState(),
      },
      global: {
        stubs: {
          CharacterDetailPanel: characterDetailPanelStub,
          AddCharacterDialog: true,
          EditCharacterDialog: true,
          AddFormDialog: true,
          EditFormDialog: true,
          OrthographicDialog: true,
        },
      },
    })

    const tile = wrapper.find('.character-tile')
    expect(tile.element.tagName).toBe('BUTTON')
    expect(tile.attributes('type')).toBe('button')
    expect(tile.attributes('aria-pressed')).toBe('false')

    await tile.trigger('click')

    expect(tile.attributes('aria-pressed')).toBe('true')
    expect(wrapper.find('.character-detail-stub').text()).toBe('主角')
  })
})
