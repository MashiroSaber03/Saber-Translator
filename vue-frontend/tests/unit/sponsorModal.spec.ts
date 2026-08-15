import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import SponsorModal from '@/components/bookshelf/SponsorModal.vue'

const BaseModalStub = defineComponent({
  props: {
    title: {
      type: String,
      default: '',
    },
  },
  emits: ['close'],
  template: `
    <section class="base-modal-stub">
      <h2>{{ title }}</h2>
      <slot />
      <footer><slot name="footer" /></footer>
    </section>
  `,
})

describe('SponsorModal', () => {
  it('uses current product copy without decorative emoji', () => {
    const wrapper = mount(SponsorModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    expect(wrapper.text()).toContain('如果这个项目对你有帮助，欢迎支持作者继续维护。')
    expect(wrapper.text()).toContain('感谢您的支持！')
    expect(wrapper.text()).not.toContain('☕')
    expect(wrapper.text()).not.toContain('🙏')
  })

  it('renders the close action through the product dialog action row', () => {
    const wrapper = mount(SponsorModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)

    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('赞助弹窗操作')
  })

  it('uses current semantic tokens without old-token fallback chains', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/SponsorModal.vue'),
      'utf8',
    )

    expect(source).toContain('var(--color-border-muted)')
    expect(source).toContain('var(--color-text-supporting)')
    expect(source).not.toContain('color-border-subtle')
    expect(source).not.toContain('color-text-secondary')
    expect(source).toContain('flex-wrap: wrap')
  })

  it('keeps sponsor modal hooks under the sponsor-modal owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/SponsorModal.vue'),
      'utf8',
    )

    for (const oldClass of [
      'sponsor-content',
      'sponsor-message',
      'qr-codes',
      'qr-item',
      'qr-image',
      'qr-label',
      'sponsor-thanks',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toMatch(/\.sponsor-modal__[^{]+ img\b/)

    for (const ownerClass of [
      'sponsor-modal__content',
      'sponsor-modal__message',
      'sponsor-modal__qr-codes',
      'sponsor-modal__qr-item',
      'sponsor-modal__qr-image',
      'sponsor-modal__qr-img',
      'sponsor-modal__qr-label',
      'sponsor-modal__thanks',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
