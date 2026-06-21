import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import AppShell from '@/components/ui/AppShell.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

describe('UI primitives architecture contracts', () => {
  it('renders named button variants through the styled button base instead of the unstyled escape hatch', () => {
    const variants = ['link', 'toolbar', 'card-action', 'tab', 'plain-danger'] as const

    for (const variant of variants) {
      const wrapper = mount(UiButton, {
        props: { variant },
        slots: { default: variant },
      })

      expect(wrapper.classes()).toContain('ui-button')
      expect(wrapper.classes()).toContain(`ui-button--${variant}`)
      expect(wrapper.classes()).not.toContain('ui-button-unstyled')
    }
  })

  it('supports loading state and tone classes without changing the public button element', () => {
    const wrapper = mount(UiButton, {
      props: { loading: true, tone: 'danger', icon: true, block: true },
      slots: { default: 'Delete' },
    })

    expect(wrapper.attributes('disabled')).toBeDefined()
    expect(wrapper.attributes('aria-busy')).toBe('true')
    expect(wrapper.classes()).toEqual(expect.arrayContaining([
      'ui-button',
      'ui-button--tone-danger',
      'ui-button--loading',
      'ui-button--icon',
      'ui-button--block',
    ]))
  })

  it('passes through input attributes and emits numeric values for number inputs', async () => {
    const wrapper = mount(UiInput, {
      props: {
        modelValue: 1,
        type: 'number',
        min: 0,
        max: 10,
        step: 1,
        size: 'sm',
        error: true,
        readonly: true,
        disabled: true,
      },
    })

    const input = wrapper.get('input')
    expect(input.attributes('type')).toBe('number')
    expect(input.attributes('min')).toBe('0')
    expect(input.attributes('max')).toBe('10')
    expect(input.attributes('step')).toBe('1')
    expect(input.attributes('readonly')).toBeDefined()
    expect(input.attributes('disabled')).toBeDefined()
    expect(input.classes()).toContain('ui-input--sm')
    expect(input.classes()).toContain('ui-input--error')

    await wrapper.setProps({ disabled: false, readonly: false })
    await input.setValue('7')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([7])
  })

  it('supports textarea readonly disabled size and error states', () => {
    const wrapper = mount(UiTextarea, {
      props: {
        modelValue: 'hello',
        rows: 8,
        size: 'lg',
        error: true,
        readonly: true,
        disabled: true,
      },
    })

    const textarea = wrapper.get('textarea')
    expect(textarea.attributes('rows')).toBe('8')
    expect(textarea.attributes('readonly')).toBeDefined()
    expect(textarea.attributes('disabled')).toBeDefined()
    expect(textarea.classes()).toContain('ui-textarea--lg')
    expect(textarea.classes()).toContain('ui-textarea--error')
  })

  it('defers textarea model and input handlers until IME composition commits', async () => {
    const onInput = vi.fn()
    const wrapper = mount(UiTextarea, {
      props: { modelValue: '' },
      attrs: { onInput },
    })
    const textarea = wrapper.get('textarea')

    await textarea.trigger('compositionstart')
    textarea.element.value = 'ni'
    await textarea.trigger('input')

    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    expect(onInput).not.toHaveBeenCalled()

    textarea.element.value = '你'
    await textarea.trigger('compositionend')

    expect(wrapper.emitted('update:modelValue')).toEqual([['你']])
    expect(onInput).toHaveBeenCalledTimes(1)
    expect((onInput.mock.calls[0]?.[0] as Event).type).toBe('input')
  })

  it('defers text input model and input handlers until IME composition commits', async () => {
    const onInput = vi.fn()
    const wrapper = mount(UiInput, {
      props: { modelValue: '', type: 'text' },
      attrs: { onInput },
    })
    const input = wrapper.get('input')

    await input.trigger('compositionstart')
    input.element.value = 'hao'
    await input.trigger('input')

    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    expect(onInput).not.toHaveBeenCalled()

    input.element.value = '好'
    await input.trigger('compositionend')

    expect(wrapper.emitted('update:modelValue')).toEqual([['好']])
    expect(onInput).toHaveBeenCalledTimes(1)
    expect((onInput.mock.calls[0]?.[0] as Event).type).toBe('input')
  })

  it('connects field labels descriptions and errors to controls', () => {
    const wrapper = mount(UiField, {
      props: {
        label: 'API Key',
        description: 'Used for requests',
        error: 'Required',
        controlId: 'api-key',
        required: true,
      },
      slots: { default: '<input id="api-key">' },
    })

    expect(wrapper.get('label').attributes('for')).toBe('api-key')
    expect(wrapper.get('label').text()).toContain('API Key')
    expect(wrapper.text()).toContain('Used for requests')
    expect(wrapper.text()).toContain('Required')
    expect(wrapper.classes()).toContain('ui-field--invalid')
  })

  it('exposes shell slots and layout variables for page-level composition', () => {
    const shell = mount(AppShell, {
      props: { variant: 'reader', contentClass: 'reader-content' },
      slots: {
        header: '<header>Header</header>',
        default: '<main>Main</main>',
        overlay: '<aside>Overlay</aside>',
      },
    })
    expect(shell.classes()).toContain('ui-app-shell--reader')
    expect(shell.find('.ui-app-shell__header').text()).toBe('Header')
    expect(shell.find('.reader-content').text()).toBe('Main')
    expect(shell.find('.ui-app-shell__overlay').text()).toBe('Overlay')

    const layout = mount(SidebarLayout, {
      props: {
        leftWidth: '300px',
        rightWidth: '220px',
        gap: '12px',
        mode: 'fixed',
        collapsed: 'left',
      },
      slots: {
        left: '<aside>Left</aside>',
        default: '<main>Center</main>',
        right: '<aside>Right</aside>',
      },
    })
    expect(layout.classes()).toEqual(expect.arrayContaining([
      'ui-sidebar-layout--fixed',
      'ui-sidebar-layout--left-collapsed',
    ]))
    expect(layout.attributes('style')).toContain('--ui-sidebar-left-width: 300px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-right-width: 220px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-gap: 12px;')

    const fallbackLayout = mount(SidebarLayout, {
      slots: {
        default: '<aside class="fallback-left">Left</aside><main class="fallback-main">Center</main>',
      },
    })
    expect(fallbackLayout.find('.ui-sidebar-layout__main').exists()).toBe(false)
    expect(fallbackLayout.find('.fallback-left').exists()).toBe(true)
    expect(fallbackLayout.find('.fallback-main').exists()).toBe(true)
  })

  it('keeps page layout algorithms in shell primitives instead of page CSS', () => {
    const shell = mount(AppShell, {
      props: {
        variant: 'studio',
        chrome: 'fixed',
        viewportMode: 'locked',
        contentScroll: 'content',
        headerHeight: '72px',
        headerOffset: '72px',
        contentPadding: '16px',
        scrollMode: 'content',
        fullHeight: true,
      },
      slots: {
        header: '<header>Header</header>',
        default: '<main>Main</main>',
      },
    })

    expect(shell.classes()).toEqual(expect.arrayContaining([
      'ui-app-shell--studio',
      'ui-app-shell--chrome-fixed',
      'ui-app-shell--viewport-locked',
      'ui-app-shell--full-height',
      'ui-app-shell--scroll-content',
    ]))
    expect(shell.attributes('style')).toContain('--ui-app-shell-header-height: 72px;')
    expect(shell.attributes('style')).toContain('--ui-app-shell-header-offset: 72px;')
    expect(shell.attributes('style')).toContain('--ui-app-shell-content-padding: 16px;')

    const layout = mount(SidebarLayout, {
      props: {
        height: 'calc(100vh - 72px)',
        leftWidth: '320px',
        rightWidth: '240px',
        leftInset: '320px',
        rightInset: '240px',
        contentInset: '20px',
        gap: '20px',
        mobileMode: 'drawer',
        paneScroll: true,
        scrollMode: 'panes',
        sidebars: 'sticky',
        sidebarTop: '72px',
      },
      slots: {
        left: '<aside>Left</aside>',
        default: '<main>Center</main>',
        right: '<aside>Right</aside>',
      },
    })

    expect(layout.classes()).toEqual(expect.arrayContaining([
      'ui-sidebar-layout--scroll-panes',
      'ui-sidebar-layout--sidebars-sticky',
      'ui-sidebar-layout--mobile-drawer',
      'ui-sidebar-layout--pane-scroll',
    ]))
    expect(layout.attributes('style')).toContain('--ui-sidebar-height: calc(100vh - 72px);')
    expect(layout.attributes('style')).toContain('--ui-sidebar-top: 72px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-left-inset: 320px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-right-inset: 240px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-content-inset: 20px;')
  })

  it('provides select checkbox and file input primitives for form migration', async () => {
    const select = mount(UiSelect, {
      props: {
        modelValue: 'b',
        options: [
          { label: 'Option A', value: 'a' },
          { label: 'Option B', value: 'b' },
        ],
        size: 'sm',
        error: true,
      },
    })
    expect(select.get('select').classes()).toContain('ui-select--sm')
    expect(select.get('select').classes()).toContain('ui-select--error')
    await select.get('select').setValue('a')
    expect(select.emitted('update:modelValue')?.at(-1)).toEqual(['a'])

    const checkbox = mount(UiCheckbox, {
      props: { modelValue: true, label: 'Enabled', description: 'Turn it on' },
    })
    expect(checkbox.get('input').attributes('type')).toBe('checkbox')
    expect(checkbox.get('input').element.checked).toBe(true)
    expect(checkbox.text()).toContain('Enabled')
    expect(checkbox.text()).toContain('Turn it on')
    await checkbox.get('input').setValue(false)
    expect(checkbox.emitted('update:modelValue')?.at(-1)).toEqual([false])

    const fileInput = mount(UiFileInput, {
      props: { accept: '.json', multiple: true, hidden: true },
    })
    expect(fileInput.get('input').attributes('type')).toBe('file')
    expect(fileInput.get('input').attributes('accept')).toBe('.json')
    expect(fileInput.get('input').attributes('multiple')).toBeDefined()
    expect(fileInput.get('input').attributes('hidden')).toBeDefined()
  })
})
