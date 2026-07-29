import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import AppShell from '@/components/ui/AppShell.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiColorSwatchGroup from '@/components/ui/UiColorSwatchGroup.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

describe('UI primitives architecture contracts', () => {
  it('owns shared select option contracts in the UI primitive layer', () => {
    const selectTypesSource = readFileSync(
      resolve(process.cwd(), 'src/components/ui/selectTypes.ts'),
      'utf8',
    )

    expect(selectTypesSource).toContain('export type UiSelectValue = string | number')
    expect(selectTypesSource).toContain('export interface UiSelectOption')
    expect(selectTypesSource).toContain('export interface UiSelectGroup')

    for (const file of [
      'src/components/ui/UiSelect.vue',
      'src/components/ui/UiCombobox.vue',
      'src/components/ui/UiModelPicker.vue',
      'src/components/translate/settings-sidebar/TextStyleSection.vue',
      'src/components/translate/settings-sidebar/WorkflowSection.vue',
      'src/components/translate/web-import/WebImportBasicSettingsPanel.vue',
      'src/components/translate/web-import/WebImportSettingsPanel.vue',
      'src/components/insight/settings/InsightModelProviderSection.vue',
      'src/components/product/ProductBookSelector.vue',
      'src/components/settings/HqTranslationSettings.vue',
      'src/components/settings/OcrSettings.vue',
      'src/components/settings/TranslationSettings.vue',
      'tests/unit/settingsSidebar.bookConstraints.spec.ts',
    ]) {
      const content = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(content, file).toContain("from '@/components/ui/selectTypes'")
      expect(content, file).not.toMatch(/type Select(Value|Option)\s*=/)
      expect(content, file).not.toMatch(/interface SelectOption/)
    }
  })

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

  it('owns link-button styling as a public primitive contract', () => {
    const wrapper = mount(UiButton, {
      props: {
        variant: 'link',
        size: 'xs',
      },
      slots: { default: '展开' },
    })
    const buttonSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiButton.vue'), 'utf8')

    expect(wrapper.classes()).toEqual(expect.arrayContaining([
      'ui-button',
      'ui-button--link',
      'ui-button--xs',
      'ui-button--bare',
    ]))
    expect(buttonSource).toContain('.ui-button--link')
    expect(buttonSource).toContain('--ui-button-link-color')
    expect(buttonSource).toContain('--ui-button-link-font-size')
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

  it('maps solid button text colors through semantic inverse text tokens', () => {
    const buttonSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiButton.vue'), 'utf8')
    const iconButtonSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiIconButton.vue'), 'utf8')

    expect(buttonSource).toContain('var(--ui-button-primary-color, var(--color-text-inverse))')
    expect(buttonSource).toContain('var(--ui-button-danger-color, var(--color-text-inverse))')
    expect(buttonSource).not.toMatch(/color:\s*var\([^;]+,\s*white\)/)
    expect(iconButtonSource).toContain('color: var(--color-text-inverse)')
    expect(iconButtonSource).not.toMatch(/color:\s*white\b/)

    const iconButton = mount(UiIconButton, {
      props: {
        label: '保存',
        variant: 'primary',
      },
      slots: {
        default: '<span>保存</span>',
      },
    })
    expect(iconButton.classes()).toContain('ui-icon-button--primary')
    expect(iconButton.attributes('aria-label')).toBe('保存')
  })

  it('exposes inverse floating icon-button chrome through public props', () => {
    const closeButton = mount(UiIconButton, {
      props: {
        label: '关闭浮层',
        variant: 'inverse',
        shape: 'circle',
        size: 'sm',
      },
      slots: {
        default: '<span>x</span>',
      },
    })
    const floatingButton = mount(UiIconButton, {
      props: {
        label: '回到顶部',
        variant: 'primary',
        shape: 'circle',
        size: 'xl',
        elevated: true,
      },
      slots: {
        default: '<span>up</span>',
      },
    })

    expect(closeButton.classes()).toEqual(expect.arrayContaining([
      'ui-icon-button--inverse',
      'ui-icon-button--circle',
      'ui-icon-button--sm',
    ]))
    expect(floatingButton.classes()).toEqual(expect.arrayContaining([
      'ui-icon-button--primary',
      'ui-icon-button--circle',
      'ui-icon-button--xl',
      'ui-icon-button--elevated',
    ]))
  })

  it('exposes active icon-button state through a public class', () => {
    const activeButton = mount(UiIconButton, {
      props: {
        label: '打开导航',
        active: true,
      },
      slots: {
        default: '<span>nav</span>',
      },
    })

    expect(activeButton.classes()).toContain('ui-icon-button--active')
    expect(activeButton.attributes('aria-pressed')).toBeUndefined()

    const pressedButton = mount(UiIconButton, {
      props: {
        label: '打开导航',
        active: true,
        pressed: true,
      },
      slots: {
        default: '<span>nav</span>',
      },
    })

    expect(pressedButton.classes()).toContain('ui-icon-button--active')
    expect(pressedButton.attributes('aria-pressed')).toBe('true')

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiIconButton.vue'), 'utf8')
    expect(source).toContain('active?: boolean')
    expect(source).toContain('pressed?: boolean')
    expect(source).toContain("'ui-icon-button--active': active")
    expect(source).toContain(':aria-pressed="pressed"')
  })

  it('allows icon buttons to keep concise labels and richer tooltips', () => {
    const button = mount(UiIconButton, {
      props: {
        label: '上一张图片',
        title: '上一张图片 (A)',
      },
      slots: {
        default: '<span>prev</span>',
      },
    })
    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiIconButton.vue'), 'utf8')

    expect(button.attributes('aria-label')).toBe('上一张图片')
    expect(button.attributes('title')).toBe('上一张图片 (A)')
    expect(source).toContain('title?: string')
    expect(source).toContain(':title="title || label"')
  })

  it('owns disabled visual treatment for styled buttons', () => {
    const buttonSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiButton.vue'), 'utf8')

    expect(buttonSource).toContain('--ui-button-disabled-background')
    expect(buttonSource).toContain('--ui-button-disabled-border')
    expect(buttonSource).toContain('--ui-button-disabled-color')
    expect(buttonSource).toContain('box-shadow: none')
    expect(buttonSource).toContain('opacity: var(--ui-button-disabled-opacity, 1)')
  })

  it('renders shared icons with explicit decorative and labelled contracts', () => {
    const decorative = mount(UiIcon, {
      props: { name: 'search', size: 20 },
    })

    expect(decorative.find('svg').exists()).toBe(true)
    expect(decorative.find('svg').attributes('aria-hidden')).toBe('true')
    expect(decorative.find('svg').attributes('width')).toBe('20')
    expect(decorative.find('svg').classes()).toContain('ui-icon')

    const labelled = mount(UiIcon, {
      props: {
        name: 'settings',
        label: '设置',
        decorative: false,
      },
    })

    expect(labelled.find('svg').attributes('role')).toBe('img')
    expect(labelled.find('svg').attributes('aria-label')).toBe('设置')
    expect(labelled.find('svg').attributes('aria-hidden')).toBeUndefined()
  })

  it('provides shared spinner and progress primitives for task surfaces', () => {
    const spinner = mount(UiSpinner, {
      props: {
        label: '下载中',
        decorative: false,
        size: 16,
      },
    })
    expect(spinner.classes()).toContain('ui-spinner')
    expect(spinner.attributes('role')).toBe('status')
    expect(spinner.attributes('aria-label')).toBe('下载中')
    expect(spinner.attributes('style')).toContain('--ui-spinner-size: 16px;')

    const progress = mount(UiProgressBar, {
      props: {
        value: 2,
        max: 4,
        label: '网页导入下载进度',
      },
      slots: {
        default: '下载进度: 2/4',
      },
    })

    const progressbar = progress.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('4')
    expect(progressbar.attributes('aria-valuenow')).toBe('2')
    expect(progressbar.attributes('aria-label')).toBe('网页导入下载进度')
    expect(progress.get('.ui-progress-bar__fill').attributes('style')).toContain('width: 50%;')
    expect(progress.text()).toContain('下载进度: 2/4')
  })

  it('offers typed task-progress tones, sizes, and motion states', () => {
    const progress = mount(UiProgressBar, {
      props: {
        value: 75,
        label: '保存进度',
        tone: 'success',
        size: 'lg',
        striped: true,
        animated: true,
      },
    })

    expect(progress.classes()).toEqual(expect.arrayContaining([
      'ui-progress-bar',
      'ui-progress-bar--tone-success',
      'ui-progress-bar--size-lg',
      'ui-progress-bar--striped',
      'ui-progress-bar--animated',
    ]))
  })

  it('exposes progress track and stripe variables for owner theming', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/ui/UiProgressBar.vue'),
      'utf8',
    )

    expect(source).toContain('background: var(--ui-progress-bar-track')
    expect(source).toContain('var(--ui-progress-bar-stripe')
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

  it('exposes a studio input variant for character-workspace forms', () => {
    const wrapper = mount(UiInput, {
      props: {
        modelValue: 'studio copy',
        variant: 'studio',
      },
    })

    const input = wrapper.get('input')
    expect(input.classes()).toEqual(expect.arrayContaining([
      'ui-input--studio',
      'ui-input--md',
    ]))

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiInput.vue'), 'utf8')
    expect(source).toContain(':where(.ui-input--studio)')
    expect(source).toContain('var(--ui-input-studio-border')
    expect(source).not.toMatch(/var\(--studio-/)
  })

  it('exposes an editor input variant for edit-toolbar numeric fields', () => {
    const wrapper = mount(UiInput, {
      props: {
        modelValue: 12,
        type: 'number',
        variant: 'editor',
      },
    })

    const input = wrapper.get('input')
    expect(input.classes()).toEqual(expect.arrayContaining([
      'ui-input--editor',
      'ui-input--md',
    ]))

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiInput.vue'), 'utf8')
    expect(source).toContain(':where(.ui-input--editor)')
    expect(source).toContain('var(--ui-input-editor-border')
    expect(source).not.toMatch(/var\(--edit-/)
  })

  it('exposes an embedded input variant for owner-framed fields', () => {
    const wrapper = mount(UiInput, {
      props: {
        modelValue: 'tag',
        variant: 'embedded',
      },
    })

    const input = wrapper.get('input')
    expect(input.classes()).toEqual(expect.arrayContaining([
      'ui-input--embedded',
      'ui-input--md',
    ]))

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiInput.vue'), 'utf8')
    expect(source).toContain("variant?: 'default' | 'editor' | 'studio' | 'embedded'")
    expect(source).toContain(':where(.ui-input--embedded)')
    expect(source).toContain('border: var(--ui-input-embedded-border, none)')
    expect(source).toContain('background: var(--ui-input-embedded-background, transparent)')
  })

  it('provides a compact number field with optional stepper controls', async () => {
    const wrapper = mount(UiNumberField, {
      props: {
        modelValue: 2,
        inputId: 'worker-count',
        min: 1,
        max: 4,
        step: 1,
        controls: true,
        ariaLabel: '并发数',
        variant: 'editor',
      },
    })

    expect(wrapper.classes()).toContain('ui-number-field')
    expect(wrapper.get('input#worker-count').attributes('type')).toBe('number')
    expect(wrapper.get('input#worker-count').attributes('aria-label')).toBe('并发数')
    expect(wrapper.get('input#worker-count').classes()).toContain('ui-input--editor')

    const buttons = wrapper.findAllComponents(UiButton)
    expect(buttons).toHaveLength(2)

    await buttons[1]!.trigger('click')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([3])

    await wrapper.setProps({ modelValue: 4 })
    expect(buttons[1]!.attributes('disabled')).toBeDefined()

    await wrapper.get('input#worker-count').setValue('0')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([1])
  })

  it('preserves nullable number fields when the input is cleared', async () => {
    const wrapper = mount(UiNumberField, {
      props: {
        modelValue: 5,
        inputId: 'optional-page',
        min: 1,
        nullable: true,
        ariaLabel: '关联页码',
      },
    })

    await wrapper.get('input#optional-page').setValue('')

    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([null])
    expect(wrapper.emitted('change')?.at(-1)).toEqual([null])
  })

  it('exposes focus for composite owners without requiring DOM queries', () => {
    const wrapper = mount(UiInput, {
      props: {
        modelValue: 'https://example.com/chapter',
        type: 'url',
      },
    })
    const input = wrapper.get('input')
    const focusSpy = vi.spyOn(input.element, 'focus')

    ;(wrapper.vm as unknown as { focus: () => void }).focus()

    expect(focusSpy).toHaveBeenCalledTimes(1)
  })

  it('provides a shared password field with an accessible reveal control', async () => {
    const wrapper = mount(UiPasswordField, {
      props: {
        modelValue: 'secret-key',
        inputId: 'settingsApiKey',
        placeholder: '请输入 API Key',
        showLabel: '显示翻译 API Key',
        hideLabel: '隐藏翻译 API Key',
      },
    })

    const input = wrapper.get('input#settingsApiKey')
    expect(input.attributes('type')).toBe('password')
    expect(input.attributes('autocomplete')).toBe('off')
    expect(input.attributes('placeholder')).toBe('请输入 API Key')

    const toggle = wrapper.get('button')
    expect(toggle.attributes('aria-label')).toBe('显示翻译 API Key')
    expect(wrapper.findComponent(UiIcon).props('name')).toBe('eye')

    await toggle.trigger('click')

    expect(input.attributes('type')).toBe('text')
    expect(toggle.attributes('aria-label')).toBe('隐藏翻译 API Key')
    expect(wrapper.findComponent(UiIcon).props('name')).toBe('eye-off')

    await input.setValue('new-key')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual(['new-key'])

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiPasswordField.vue'), 'utf8')
    expect(source).not.toContain('input: [event: Event]')
    expect(source).not.toContain('@input="$emit')
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

  it('exposes a panel textarea variant for product workspace text editors', () => {
    const wrapper = mount(UiTextarea, {
      props: {
        modelValue: 'panel copy',
        variant: 'panel',
        size: 'lg',
      },
    })

    const textarea = wrapper.get('textarea')
    expect(textarea.classes()).toEqual(expect.arrayContaining([
      'ui-textarea--panel',
      'ui-textarea--lg',
    ]))
  })

  it('exposes a studio textarea variant for character-workspace forms', () => {
    const wrapper = mount(UiTextarea, {
      props: {
        modelValue: 'studio prompt',
        variant: 'studio',
      },
    })

    const textarea = wrapper.get('textarea')
    expect(textarea.classes()).toEqual(expect.arrayContaining([
      'ui-textarea--studio',
      'ui-textarea--md',
    ]))

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiTextarea.vue'), 'utf8')
    expect(source).toContain(':where(.ui-textarea--studio)')
    expect(source).toContain('var(--ui-textarea-studio-background')
    expect(source).not.toMatch(/var\(--studio-/)
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

  it('does not keep obsolete form compatibility selectors in field styling', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiField.vue'), 'utf8')

    expect(source).not.toContain('ui-checkbox-label')
    expect(source).not.toContain('error-hint')
  })

  it('renders label-adjacent actions through a dedicated field slot', () => {
    const wrapper = mount(UiField, {
      props: {
        label: '附加请求字段',
        controlId: 'openAiExtraBody',
        variant: 'settings',
      },
      slots: {
        'label-actions': '<button type="button">格式化</button>',
        default: '<textarea id="openAiExtraBody" />',
      },
    })

    expect(wrapper.get('.ui-field__header label').text()).toBe('附加请求字段')
    expect(wrapper.get('.ui-field__label-actions button').text()).toBe('格式化')
    expect(wrapper.get('label').attributes('for')).toBe('openAiExtraBody')
  })

  it('supports inline field layout for compact toolbar controls', () => {
    const wrapper = mount(UiField, {
      props: {
        label: '图片大小',
        controlId: 'imageSize',
        layout: 'inline',
      },
      slots: {
        default: '<input id="imageSize" type="range">',
      },
    })

    expect(wrapper.classes()).toContain('ui-field--layout-inline')
    expect(wrapper.get('label').attributes('for')).toBe('imageSize')
  })

  it('exposes an editor field variant for compact edit toolbar labels', () => {
    const wrapper = mount(UiField, {
      props: {
        label: '字号',
        controlId: 'bubbleFontSize',
        variant: 'editor',
      },
      slots: {
        default: '<input id="bubbleFontSize" type="number">',
      },
    })

    expect(wrapper.classes()).toContain('ui-field--editor')
    expect(wrapper.get('label').attributes('for')).toBe('bubbleFontSize')

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiField.vue'), 'utf8')
    expect(source).toContain("variant?: 'default' | 'settings' | 'dialog' | 'editor'")
    expect(source).toContain('.ui-field--editor')
  })

  it('exposes typed field tones for inverse settings panels', () => {
    const wrapper = mount(UiField, {
      props: {
        label: '背景颜色',
        tone: 'inverse',
      },
    })

    expect(wrapper.classes()).toContain('ui-field--tone-inverse')

    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiField.vue'), 'utf8')
    expect(source).toContain("tone?: 'default' | 'inverse'")
    expect(source).toContain('.ui-field--tone-inverse .ui-field__label')
    expect(source).not.toContain('--color-text-default: var(')
  })

  it('exposes typed settings field variants instead of class-triggered form skins', () => {
    const wrapper = mount(UiField, {
      props: {
        variant: 'settings',
        control: 'checkbox',
      },
      slots: { default: '<label>Enabled</label>' },
    })

    expect(wrapper.classes()).toContain('ui-field--settings')
    expect(wrapper.classes()).toContain('ui-field--control-checkbox')
    expect(wrapper.classes()).not.toContain('ui-settings-field')
    expect(wrapper.classes()).not.toContain('ui-settings-field--checkbox')
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
    const shellSource = readFileSync(resolve(process.cwd(), 'src/components/ui/AppShell.vue'), 'utf8')
    expect(shellSource).not.toContain('contentScroll')

    const layout = mount(SidebarLayout, {
      props: {
        height: 'calc(100dvh - 72px)',
        leftWidth: '320px',
        rightWidth: '240px',
        leftInset: '320px',
        rightInset: '240px',
        contentInset: '20px',
        leftOffset: '20px',
        rightOffset: '24px',
        leftTop: '72px',
        rightTop: '20px',
        leftHeight: 'calc(100dvh - 92px)',
        rightHeight: 'calc(100dvh - 40px)',
        mainClass: 'reader-layout__main',
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
    expect(layout.attributes('style')).toContain('--ui-sidebar-height: calc(100dvh - 72px);')
    expect(layout.attributes('style')).toContain('--ui-sidebar-top: 72px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-left-inset: 320px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-right-inset: 240px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-content-inset: 20px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-left-offset: 20px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-right-offset: 24px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-left-top: 72px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-right-top: 20px;')
    expect(layout.attributes('style')).toContain('--ui-sidebar-left-height: calc(100dvh - 92px);')
    expect(layout.attributes('style')).toContain('--ui-sidebar-right-height: calc(100dvh - 40px);')
    expect(layout.get('.ui-sidebar-layout__main').classes()).toContain('reader-layout__main')
  })

  it('provides overlay ownership without forcing business components to use fixed positioning', async () => {
    const overlay = mount(OverlayLayer, {
      props: {
        level: 'popover',
        passthrough: true,
      },
      slots: {
        default: '<button>Close</button>',
      },
    })

    expect(overlay.classes()).toEqual(expect.arrayContaining([
      'ui-overlay-layer',
      'ui-overlay-layer--popover',
      'ui-overlay-layer--passthrough',
    ]))
    await overlay.trigger('click')
    expect(overlay.emitted('backdrop')).toHaveLength(1)
  })

  it('provides select checkbox and file input primitives for current forms', async () => {
    const select = mount(UiSelect, {
      attachTo: document.body,
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
    expect(select.find('select').exists()).toBe(false)
    const selectTrigger = select.get('[role="combobox"]')
    expect(selectTrigger.classes()).toContain('ui-select--sm')
    expect(selectTrigger.classes()).toContain('ui-select--error')
    await selectTrigger.trigger('click')
    expect(document.body.querySelector('.ui-select-dropdown')?.getAttribute('role')).toBe('listbox')
    const firstOption = document.body.querySelector('[role="option"]') as HTMLElement
    expect(firstOption?.textContent).toContain('Option A')
    firstOption.click()
    expect(select.emitted('update:modelValue')?.at(-1)).toEqual(['a'])

    const studioSelect = mount(UiSelect, {
      props: {
        modelValue: 'before_char',
        variant: 'studio',
        options: [{ label: 'Before', value: 'before_char' }],
      },
    })
    expect(studioSelect.get('[role="combobox"]').classes()).toEqual(expect.arrayContaining([
      'ui-select--studio',
      'ui-select--md',
    ]))

    const numericSelect = mount(UiSelect, {
      attachTo: document.body,
      props: {
        modelValue: 2,
        options: [
          { label: 'One', value: 1 },
          { label: 'Two', value: 2 },
        ],
      },
    })
    await numericSelect.get('[role="combobox"]').trigger('click')
    ;(document.body.querySelector('[data-ui-select-value="1"]') as HTMLElement).click()
    expect(numericSelect.emitted('update:modelValue')?.at(-1)).toEqual([1])
    expect(numericSelect.emitted('change')?.at(-1)).toEqual([1])

    const checkbox = mount(UiCheckbox, {
      props: { modelValue: true, label: 'Enabled', description: 'Turn it on' },
    })
    expect(checkbox.get('input').attributes('type')).toBe('checkbox')
    expect(checkbox.get('input').element.checked).toBe(true)
    expect(checkbox.text()).toContain('Enabled')
    expect(checkbox.text()).toContain('Turn it on')
    await checkbox.get('input').setValue(false)
    expect(checkbox.emitted('update:modelValue')?.at(-1)).toEqual([false])

    const externallyLabelledCheckbox = mount(UiCheckbox, {
      props: { inputId: 'external-checkbox', ariaLabel: 'External checkbox' },
    })
    expect(externallyLabelledCheckbox.get('input').attributes('id')).toBe('external-checkbox')
    expect(externallyLabelledCheckbox.get('input').attributes('aria-label')).toBe('External checkbox')

    const selectSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiSelect.vue'), 'utf8')
    expect(selectSource).toContain(':where(.ui-select--studio)')
    expect(selectSource).toContain('var(--ui-selector-control-border')
    expect(selectSource).not.toContain('var(--ui-select-studio-')
    expect(selectSource).not.toMatch(/var\(--studio-/)

    const fileInput = mount(UiFileInput, {
      props: { accept: '.json', multiple: true, hidden: true },
    })
    expect(fileInput.get('input').attributes('type')).toBe('file')
    expect(fileInput.get('input').attributes('accept')).toBe('.json')
    expect(fileInput.get('input').attributes('multiple')).toBeDefined()
    expect(fileInput.get('input').attributes('hidden')).toBeDefined()
  })

  it('routes fixed and searchable selectors through the shared selector visual contract', () => {
    const componentTokenSource = readFileSync(
      resolve(process.cwd(), 'src/styles/tokens/component.css'),
      'utf8'
    )
    const selectSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiSelect.vue'), 'utf8')
    const comboboxSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiCombobox.vue'), 'utf8')
    const fieldSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiField.vue'), 'utf8')

    expect(componentTokenSource).toContain('--ui-selector-control-text: var(--color-text-default);')
    expect(componentTokenSource).toContain('--ui-selector-control-font-size: 14px;')
    expect(componentTokenSource).toContain('--ui-selector-control-background: var(--color-surface-base);')
    expect(componentTokenSource).toContain('--ui-selector-dropdown-background: var(--color-surface-base);')
    expect(componentTokenSource).toContain('--ui-selector-option-selected-text: var(--color-text-brand);')
    expect(selectSource).toContain('color: var(--ui-selector-control-text,')
    expect(comboboxSource).toContain('color: var(--ui-selector-control-text,')
    expect(selectSource).not.toContain('var(--ui-combobox-option-')
    expect(comboboxSource).not.toContain('var(--ui-combobox-trigger-text')
    expect(fieldSource).not.toContain('--ui-select-color:')
    expect(fieldSource).not.toContain('--ui-select-font-size:')
  })

  it('keeps teleported selector dropdown surfaces opaque when a shared token is unavailable', () => {
    const selectSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiSelect.vue'), 'utf8')
    const comboboxSource = readFileSync(resolve(process.cwd(), 'src/components/ui/UiCombobox.vue'), 'utf8')

    const opaqueSurface = 'var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas))'
    expect(selectSource).toContain(`background: ${opaqueSurface};`)
    expect(comboboxSource).toContain(`background: ${opaqueSurface};`)
  })

  it('emits typed file arrays while preserving native change listeners', async () => {
    const changeSpy = vi.fn()
    const wrapper = mount(UiFileInput, {
      attrs: {
        onChange: changeSpy,
      },
    })
    const file = new File(['card'], 'card.json', { type: 'application/json' })
    const input = wrapper.get('input')

    Object.defineProperty(input.element, 'files', {
      configurable: true,
      value: [file],
    })

    await input.trigger('change')

    expect(wrapper.emitted('files-change')).toEqual([[[file]]])
    expect(changeSpy).toHaveBeenCalledTimes(1)

    ;(wrapper.vm as unknown as { clear: () => void }).clear()

    expect((input.element as HTMLInputElement).value).toBe('')
  })

  it('provides an accessible color swatch group for color choices', async () => {
    const wrapper = mount(UiColorSwatchGroup, {
      props: {
        modelValue: '#ffffff',
        ariaLabel: '阅读背景颜色',
        options: [
          { value: '#1a1a2e', label: '深蓝' },
          { value: '#ffffff', label: '白色' },
        ],
      },
    })

    expect(wrapper.attributes('role')).toBe('group')
    expect(wrapper.attributes('aria-label')).toBe('阅读背景颜色')

    const buttons = wrapper.findAll('button')
    expect(buttons).toHaveLength(2)
    expect(buttons[0].attributes('aria-label')).toBe('深蓝')
    expect(buttons[0].attributes('aria-pressed')).toBe('false')
    expect(buttons[0].attributes('style')).toContain('--ui-swatch-background: #1a1a2e;')
    expect(buttons[1].attributes('aria-pressed')).toBe('true')
    expect(buttons[1].classes()).toContain('ui-color-swatch-group__swatch--selected')
    const source = readFileSync(resolve(process.cwd(), 'src/components/ui/UiColorSwatchGroup.vue'), 'utf8')
    expect(source).toContain('button.ui-color-swatch-group__swatch')
    expect(source).toContain('width: var(--ui-button-icon-width)')
    expect(source).toContain('height: var(--ui-button-icon-height)')

    await buttons[0].trigger('click')

    expect(wrapper.emitted('update:modelValue')).toEqual([['#1a1a2e']])
    expect(wrapper.emitted('change')).toEqual([['#1a1a2e']])
  })

  it('provides a shared color input primitive for arbitrary color values', async () => {
    const component = await import('@/components/ui/UiColorInput.vue')
    const wrapper = mount(component.default, {
      props: {
        inputId: 'textColor',
        modelValue: '#112233',
        disabled: true,
        ariaLabel: '文字颜色',
        size: 'sm',
      },
    })

    const input = wrapper.get('input[type="color"]')
    expect(input.attributes('id')).toBe('textColor')
    expect(input.attributes('aria-label')).toBe('文字颜色')
    expect(input.attributes('disabled')).toBeDefined()
    expect(input.classes()).toEqual(expect.arrayContaining([
      'ui-color-input',
      'ui-color-input--sm',
    ]))

    await wrapper.setProps({ disabled: false })
    await wrapper.get('input[type="color"]').setValue('#445566')

    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['#445566'])
    expect(wrapper.emitted('change')?.[0]).toEqual(['#445566'])

    const hiddenWrapper = mount(component.default, {
      props: {
        modelValue: '#000000',
        hidden: true,
      },
    })
    const hiddenInput = hiddenWrapper.get('input[type="color"]')
    expect(hiddenInput.attributes('hidden')).toBeDefined()
    expect(hiddenInput.classes()).toContain('ui-color-input--hidden')

    const clickSpy = vi.spyOn(hiddenInput.element, 'click').mockImplementation(() => undefined)
    ;(hiddenWrapper.vm as unknown as { click: () => void }).click()
    expect(clickSpy).toHaveBeenCalledTimes(1)
  })

  it('provides a shared model picker for settings model fetch controls', async () => {
    const wrapper = mount(UiModelPicker, {
      props: {
        modelValue: 'gpt-4o',
        inputId: 'settingsModelName',
        placeholder: '请输入模型名称',
        fetchTitle: '获取可用模型列表',
        fetching: false,
        fetchDisabled: false,
        options: [
          { label: '-- 选择模型 --', value: '' },
          { label: 'gpt-4o', value: 'gpt-4o' },
          { label: 'gpt-4.1', value: 'gpt-4.1' },
        ],
      },
    })

    const input = wrapper.get('input#settingsModelName')
    expect(input.attributes('placeholder')).toBe('请输入模型名称')
    await input.setValue('gpt-4.1')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual(['gpt-4.1'])

    const fetchButton = wrapper.get('button')
    expect(fetchButton.attributes('title')).toBe('获取可用模型列表')
    expect(fetchButton.text()).toContain('获取模型')
    expect(wrapper.findComponent(UiIcon).props('name')).toBe('search')
    await fetchButton.trigger('click')
    expect(wrapper.emitted('fetch')).toHaveLength(1)

    const combobox = wrapper.getComponent({ name: 'UiCombobox' })
    expect(combobox.props('modelValue')).toBe('gpt-4o')
    expect(combobox.props('options')).toHaveLength(3)
    await combobox.vm.$emit('change', 'gpt-4.1')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual(['gpt-4.1'])
    expect(wrapper.emitted('change')?.at(-1)).toEqual(['gpt-4.1'])
    expect(wrapper.text()).toContain('共 2 个模型')

    await wrapper.setProps({ fetching: true, fetchDisabled: true })
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    expect(wrapper.get('button').text()).toContain('获取中...')

    await wrapper.setProps({ disabled: true })
    expect(wrapper.get('input').attributes('disabled')).toBeDefined()
    expect(wrapper.getComponent({ name: 'UiCombobox' }).props('disabled')).toBe(true)
  })

  it('lets model picker callers choose fetch action emphasis through a typed button variant', () => {
    const wrapper = mount(UiModelPicker, {
      props: {
        modelValue: 'gpt-4o',
        fetchVariant: 'primary',
      },
    })

    expect(wrapper.getComponent(UiButton).props('variant')).toBe('primary')
  })

  it('provides an accessible switch primitive for binary product controls', async () => {
    const wrapper = mount(UiSwitch, {
      props: {
        modelValue: true,
        accessibilityLabel: '启用角色 Saber',
        size: 'sm',
      },
    })

    const control = wrapper.get('[role="switch"]')
    expect(control.attributes('aria-label')).toBe('启用角色 Saber')
    expect(control.attributes('aria-checked')).toBe('true')
    expect(control.attributes('aria-pressed')).toBeUndefined()
    expect(control.classes()).toContain('ui-switch--sm')

    await control.trigger('click')

    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
    expect(wrapper.emitted('change')).toEqual([[false]])
  })
})
