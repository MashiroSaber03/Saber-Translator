import { enableAutoUnmount, mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import FirstTimeGuide from '@/components/translate/FirstTimeGuide.vue'
import ToastNotification from '@/components/common/ToastNotification.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { toastService } from '@/utils/toast'

enableAutoUnmount(afterEach)

describe('common component accessibility contracts', () => {
  beforeEach(() => {
    toastService.clearAll()
  })

  afterEach(() => {
    toastService.clearAll()
    document.body.innerHTML = ''
  })

  it('names toast close buttons', async () => {
    const wrapper = mount(ToastNotification, {
      attachTo: document.body,
    })

    toastService.addToast('Hello', 'info', 0)
    await Promise.resolve()

    const closeButton = document.body.querySelector('.vue-toast-close')
    expect(closeButton?.getAttribute('aria-label')).toBe('关闭通知')
    expect(closeButton?.querySelector('.ui-icon')).not.toBeNull()
    expect(closeButton?.textContent).not.toContain('×')
    expect(wrapper.getComponent(UiIconButton).props('label')).toBe('关闭通知')
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

  it('keeps the toast service off the component instance API', () => {
    const toastSource = readFileSync(
      resolve(process.cwd(), 'src/components/common/ToastNotification.vue'),
      'utf8'
    )

    expect(toastSource).not.toContain('defineExpose')
    expect(toastSource).not.toContain('addToast:')
    expect(toastSource).not.toContain('showGeneralMessage:')
  })

  it('renders first-use setup actions through the product dialog action row', async () => {
    localStorage.removeItem('saber_translator_dismiss_setup_reminder')

    mount(FirstTimeGuide, {
      attachTo: document.body,
    })
    await nextTick()

    const actionRow = document.body.querySelector('[role="group"][aria-label="首次使用设置操作"]')
    expect(actionRow).not.toBeNull()
    const dialog = document.body.querySelector('[role="dialog"]')
    expect(dialog?.textContent).toContain('立即配置')
    expect(dialog?.textContent).toContain('稍后配置')
  })

  it('keeps first-use guide state behind the translate owner helper boundary', () => {
    const guideSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/FirstTimeGuide.vue'),
      'utf8'
    )

    expect(guideSource).toContain('@/components/translate/firstTimeGuideState')
    expect(guideSource).not.toContain('defineExpose')
    expect(guideSource).not.toContain('resetGuideState')
    expect(guideSource).not.toMatch(/var\(--color-border-muted,\s*var\(--color-border-(?:default|subtle)\)\)/)
  })

  it('keeps feature-only setup and sidebar disclosure components out of common', () => {
    const textStyleSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/TextStyleSection.vue'),
      'utf8'
    )
    const pageSelectionSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/PageSelectionSection.vue'),
      'utf8'
    )
    const translateViewSource = readFileSync(
      resolve(process.cwd(), 'src/views/TranslateView.vue'),
      'utf8'
    )

    expect(textStyleSource).toContain('@/components/product/ProductCollapsibleSection.vue')
    expect(pageSelectionSource).toContain('@/components/product/ProductCollapsibleSection.vue')
    expect(textStyleSource).not.toContain('@/components/common/CollapsiblePanel.vue')
    expect(pageSelectionSource).not.toContain('@/components/common/CollapsiblePanel.vue')
    expect(translateViewSource).toContain('@/components/translate/FirstTimeGuide.vue')
    expect(existsSync(resolve(process.cwd(), 'src/components/common/FirstTimeGuide.vue'))).toBe(false)
    expect(existsSync(resolve(process.cwd(), 'src/components/common/CollapsiblePanel.vue'))).toBe(false)
  })
})
