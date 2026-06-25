import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import CharacterDetailPanel from '@/components/insight/continuation/CharacterDetailPanel.vue'
import FormTile from '@/components/insight/continuation/FormTile.vue'

describe('continuation character controls', () => {
  it('exposes explicit names for character detail actions', () => {
    const wrapper = mount(CharacterDetailPanel, {
      props: {
        character: {
          name: 'Saber',
          aliases: [],
          description: '骑士王',
          forms: [],
          reference_image: '',
          enabled: true,
        },
        avatarUrl: '',
        getFormImageUrl: vi.fn(),
      },
      global: {
        stubs: {
          FormTile: true,
        },
      },
    })

    const toggle = wrapper.get('button[aria-label="启用角色 Saber"]')
    expect(toggle.attributes('aria-pressed')).toBe('true')
    expect(wrapper.find('button[aria-label="编辑角色 Saber"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除角色 Saber"]').exists()).toBe(true)
  })

  it('exposes explicit names for form-tile controls', () => {
    const wrapper = mount(FormTile, {
      props: {
        form: {
          form_id: 'form_1',
          form_name: '常服',
          description: '日常服装',
          reference_image: '/tmp/form.png',
          enabled: true,
        },
        characterName: 'Saber',
        formImageUrl: '/tmp/form.png',
      },
    })

    expect(wrapper.find('input[type="file"]').attributes('aria-label')).toBe('上传 Saber 常服 参考图')
    const toggle = wrapper.get('button[aria-label="启用 Saber 常服"]')
    expect(toggle.attributes('aria-pressed')).toBe('true')
    expect(wrapper.find('button[aria-label="生成 Saber 常服 三视图"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除 Saber 常服 参考图"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="编辑 Saber 常服"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除 Saber 常服"]').exists()).toBe(true)
  })
})
