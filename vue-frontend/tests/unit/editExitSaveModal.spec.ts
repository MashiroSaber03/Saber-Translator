import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { describe, expect, it } from 'vitest'

import EditExitSaveModal from '@/components/edit/EditExitSaveModal.vue'

describe('EditExitSaveModal', () => {
  it('exposes save progress through progressbar semantics', () => {
    const wrapper = mount(EditExitSaveModal, {
      props: {
        state: 'saving',
        message: '正在保存',
        error: '',
        progressPercent: 40,
        hasProgress: true,
        current: 2,
        total: 5,
      },
      global: {
        stubs: {
          BaseModal: defineComponent({
            setup(_props, { slots }) {
              return () => h('div', [
                slots.default?.(),
                slots.footer?.(),
              ])
            },
          }),
        },
      },
    })

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('5')
    expect(progressbar.attributes('aria-valuenow')).toBe('2')
    expect(progressbar.attributes('aria-label')).toBe('退出编辑保存进度')
  })
})
