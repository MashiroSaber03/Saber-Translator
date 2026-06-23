import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AppHeader from '@/components/common/AppHeader.vue'
import CollapsiblePanel from '@/components/common/CollapsiblePanel.vue'
import ToastNotification from '@/components/common/ToastNotification.vue'
import { toastService } from '@/utils/toast'

vi.mock('vue-router', () => ({
  useRoute: () => ({ path: '/translate' }),
}))

const routerLinkStub = {
  props: ['to'],
  template: '<a :href="typeof to === \'string\' ? to : \'#\'"><slot /></a>',
}

describe('common component accessibility contracts', () => {
  beforeEach(() => {
    toastService.clearAll()
  })

  afterEach(() => {
    toastService.clearAll()
    document.body.innerHTML = ''
  })

  it('keeps AppHeader default external links safe and icon actions named', () => {
    const wrapper = mount(AppHeader, {
      props: {
        showSettingsButton: true,
      },
      global: {
        stubs: {
          RouterLink: routerLinkStub,
        },
      },
    })

    expect(wrapper.get('.app-header__settings-button').attributes('aria-label')).toBe('打开设置')
    expect(wrapper.get('.app-header__theme-toggle').attributes('aria-label')).toBe('功能开发中')
    expect(wrapper.get('.app-header__link--tutorial').attributes('rel')).toBe('noopener noreferrer')
    expect(wrapper.get('.app-header__link--github').attributes('rel')).toBe('noopener noreferrer')
  })

  it('names toast close buttons', async () => {
    mount(ToastNotification, {
      attachTo: document.body,
    })

    toastService.addToast('Hello', 'info', 0)
    await Promise.resolve()

    const closeButton = document.body.querySelector('.vue-toast-close')
    expect(closeButton?.getAttribute('aria-label')).toBe('关闭通知')
  })

  it('sanitizes html toast messages before rendering', async () => {
    mount(ToastNotification, {
      attachTo: document.body,
    })

    toastService.showGeneralMessage(
      '<strong>完成</strong><img src=x onerror="alert(1)"><a href="javascript:alert(1)" onclick="alert(1)">链接</a>',
      'info',
      true,
      0
    )
    await nextTick()

    const message = document.body.querySelector('.vue-toast-message span')
    expect(message?.innerHTML).toContain('<strong>完成</strong>')
    expect(message?.querySelector('img')).toBeNull()
    expect(message?.querySelector('a')?.getAttribute('href')).toBeNull()
    expect(message?.querySelector('a')?.getAttribute('onclick')).toBeNull()
  })

  it('uses a real button for collapsible panel toggles', async () => {
    const wrapper = mount(CollapsiblePanel, {
      props: {
        title: 'Settings',
        defaultExpanded: true,
      },
      slots: {
        default: '<p>Body</p>',
      },
    })

    const trigger = wrapper.get('.collapsible-header')
    expect(trigger.element.tagName).toBe('BUTTON')
    expect(trigger.attributes('aria-expanded')).toBe('true')

    await trigger.trigger('click')

    expect(trigger.attributes('aria-expanded')).toBe('false')
  })
})
