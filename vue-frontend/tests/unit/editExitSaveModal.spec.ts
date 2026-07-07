import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, h } from 'vue'
import { describe, expect, it } from 'vitest'

import EditExitSaveModal from '@/components/edit/EditExitSaveModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'

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

  it('uses product dialog actions and semantic owner tokens', () => {
    const wrapper = mount(EditExitSaveModal, {
      props: {
        state: 'confirm',
        message: '',
        error: '',
        progressPercent: 0,
        hasProgress: false,
        current: 0,
        total: 0,
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

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('退出编辑保存操作')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditExitSaveModal.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toContain('EditExitSaveModal.global.styles.css')
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('exit-save-dialog-btn--')
    expect(source).not.toContain('--ui-button-')
    expect(source).not.toContain('footer-gap=')
    expect(source).not.toContain('footer-justify=')
  })

  it('uses typed BaseModal placement and inverse chrome instead of global container skinning', () => {
    const wrapper = mount(EditExitSaveModal, {
      props: {
        state: 'confirm',
        message: '',
        error: '',
        progressPercent: 0,
        hasProgress: false,
        current: 0,
        total: 0,
      },
      global: {
        stubs: {
          BaseModal: defineComponent({
            props: [
              'placement',
              'backdrop',
              'chromeVariant',
              'dividerVariant',
            ],
            setup(props, { slots }) {
              return () => h('div', {
                'data-placement': props.placement,
                'data-backdrop': props.backdrop,
                'data-chrome': props.chromeVariant,
                'data-divider': props.dividerVariant,
              }, [
                slots.default?.(),
                slots.footer?.(),
              ])
            },
          }),
        },
      },
    })

    const modal = wrapper.get('[data-placement]')
    expect(modal.attributes('data-placement')).toBe('top-end')
    expect(modal.attributes('data-backdrop')).toBe('strong')
    expect(modal.attributes('data-chrome')).toBe('inverse')
    expect(modal.attributes('data-divider')).toBe('none')
  })
})
