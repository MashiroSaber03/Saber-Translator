import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h, type PropType } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ApplyOptionsSection from '@/components/translate/settings-sidebar/ApplyOptionsSection.vue'
import WorkflowSection from '@/components/translate/settings-sidebar/WorkflowSection.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useSettingsStore } from '@/stores/settings'

const {
  getFontListMock,
  savePreferencesMock,
  uploadFontMock,
} = vi.hoisted(() => ({
  getFontListMock: vi.fn(),
  savePreferencesMock: vi.fn(),
  uploadFontMock: vi.fn(),
}))

vi.mock('@/api/v2/settings', async importOriginal => ({
  ...await importOriginal<typeof import('@/api/v2/settings')>(),
  listV2Fonts: getFontListMock,
  uploadV2Font: uploadFontMock,
  updateV2WorkflowPreferences: savePreferencesMock,
}))

vi.mock('@/components/ui/UiCombobox.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: [String, Number] as PropType<string | number | undefined>,
        default: undefined,
      },
      options: {
        type: Array as PropType<Array<{ label: string; value: string | number }>>,
        default: () => [],
      },
    },
    emits: ['change'],
    setup(props, { attrs, emit }) {
      return () => h(
        'select',
        {
          ...attrs,
          value: props.modelValue,
          onChange: (event: Event) => emit('change', (event.target as HTMLSelectElement).value),
        },
        props.options.map(option => h('option', { value: option.value }, option.label))
      )
    },
  }),
}))

vi.mock('@/components/product/ProductCollapsibleSection.vue', () => ({
  default: defineComponent({
    props: {
      title: {
        type: String,
        default: '',
      },
    },
    setup(props, { slots }) {
      return () => h('section', [h('h3', props.title), slots.default?.()])
    },
  }),
}))

vi.mock('@/components/translate/PageSelectionModal.vue', () => ({
  default: defineComponent({
    setup() {
      return () => null
    },
  }),
}))

import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'

function getRememberWorkflowToggle(wrapper: ReturnType<typeof mount>) {
  const toggle = wrapper.findAllComponents(UiCheckbox)
    .find(checkbox => checkbox.props('label') === '记住操作模式')
  if (!toggle) {
    throw new Error('Remember workflow mode toggle not found')
  }
  return toggle
}

function getWorkflowModeSelect(wrapper: ReturnType<typeof mount>) {
  return wrapper.getComponent(WorkflowSection).getComponent(UiSelect)
}

function selectWorkflowMode(wrapper: ReturnType<typeof mount>, mode: string) {
  getWorkflowModeSelect(wrapper).vm.$emit('change', mode)
}

describe('SettingsSidebar workflow preferences', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    setActivePinia(createPinia())
    getFontListMock.mockReset()
    savePreferencesMock.mockReset()
    uploadFontMock.mockReset()
    getFontListMock.mockResolvedValue([])
    uploadFontMock.mockResolvedValue({
      id: 'font-uploaded',
      kind: 'uploaded',
      displayName: 'UploadedFont',
      builtinKey: null,
      assetUrl: '/api/v2/assets/font',
    })
    savePreferencesMock.mockResolvedValue({
      domain: 'workflow_preferences',
      payload: {},
      revision: 1,
      schemaVersion: 1,
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps the default workflow mode when remembering is disabled', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    expect(getWorkflowModeSelect(wrapper).props('modelValue')).toBe('translate-current')
    expect(wrapper.getComponent(UiSelect).exists()).toBe(true)
    expect(getRememberWorkflowToggle(wrapper).props('modelValue')).toBe(false)
  })

  it('restores a remembered dangerous workflow mode', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.workflowPreferences = {
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'clear-all',
    }

    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    expect(getWorkflowModeSelect(wrapper).props('modelValue')).toBe('clear-all')
    expect(getRememberWorkflowToggle(wrapper).props('modelValue')).toBe(true)
  })

  it('saves the last workflow mode immediately when the dropdown changes', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    selectWorkflowMode(wrapper, 'hq-batch')

    expect(savePreferencesMock).toHaveBeenCalledWith(
      {
        rememberWorkflowModeEnabled: false,
        lastWorkflowMode: 'hq-batch',
      },
      0,
    )
  })

  it('saves the remember switch immediately with the current workflow mode', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    getRememberWorkflowToggle(wrapper).vm.$emit('change', true)
    await flushPromises()

    expect(savePreferencesMock).toHaveBeenCalledWith(
      {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'translate-current',
      },
      0,
    )
  })

  it('keeps the selected workflow mode in the UI even when saving fails', async () => {
    savePreferencesMock.mockRejectedValueOnce(new Error('network down'))
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const wrapper = mount(SettingsSidebar)
      await flushPromises()

      selectWorkflowMode(wrapper, 'hq-batch')
      await flushPromises()

      expect(getWorkflowModeSelect(wrapper).props('modelValue')).toBe('hq-batch')
    } finally {
      warnSpy.mockRestore()
    }
  })

  it('serializes rapid workflow preference saves so the latest value wins', async () => {
    let resolveFirstSave!: () => void
    savePreferencesMock
      .mockImplementationOnce(() => new Promise(resolve => {
        resolveFirstSave = () => resolve({
          domain: 'workflow_preferences',
          payload: {},
          revision: 1,
          schemaVersion: 1,
        })
      }))
      .mockResolvedValueOnce({
        domain: 'workflow_preferences',
        payload: {},
        revision: 2,
        schemaVersion: 1,
      })

    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    selectWorkflowMode(wrapper, 'hq-batch')
    selectWorkflowMode(wrapper, 'clear-all')

    expect(savePreferencesMock).toHaveBeenCalledTimes(1)
    expect(savePreferencesMock).toHaveBeenNthCalledWith(
      1,
      {
        rememberWorkflowModeEnabled: false,
        lastWorkflowMode: 'hq-batch',
      },
      0,
    )

    resolveFirstSave()
    await flushPromises()

    expect(savePreferencesMock).toHaveBeenCalledTimes(2)
    expect(savePreferencesMock).toHaveBeenNthCalledWith(
      2,
      {
        rememberWorkflowModeEnabled: false,
        lastWorkflowMode: 'clear-all',
      },
      1,
    )
  })

  it('does not let late bootstrap hydration overwrite a manual mode change', async () => {
    const wrapper = mount(SettingsSidebar)

    selectWorkflowMode(wrapper, 'proofread-batch')
    useSettingsStore().workflowPreferences = {
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'clear-all',
    }
    await flushPromises()

    expect(getWorkflowModeSelect(wrapper).props('modelValue')).toBe('proofread-batch')
  })

  it('does not let late bootstrap hydration overwrite after the remember switch changes', async () => {
    const wrapper = mount(SettingsSidebar)

    getRememberWorkflowToggle(wrapper).vm.$emit('change', true)
    useSettingsStore().workflowPreferences = {
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'clear-all',
    }
    await flushPromises()

    expect(getWorkflowModeSelect(wrapper).props('modelValue')).toBe('translate-current')
    expect(getRememberWorkflowToggle(wrapper).props('modelValue')).toBe(true)
  })

  it('registers outside-click handling and accepts non-element event targets', async () => {
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')

    const wrapper = mount(SettingsSidebar)

    expect(addEventListenerSpy).toHaveBeenCalledWith('click', expect.any(Function))
    wrapper.getComponent(UiIconButton).vm.$emit('click')
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.apply-options-section__menu').exists()).toBe(true)

    const textNode = document.createTextNode('outside')
    document.body.append(textNode)
    expect(() => textNode.dispatchEvent(new MouseEvent('click', { bubbles: true }))).not.toThrow()
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.apply-options-section__menu').exists()).toBe(false)

    wrapper.unmount()
    expect(removeEventListenerSpy).toHaveBeenCalledWith('click', expect.any(Function))

    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
  })

  it('updates auto font size without routine console noise', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      const wrapper = mount(SettingsSidebar)
      const autoFontToggle = wrapper.findAllComponents(UiCheckbox)
        .find(checkbox => checkbox.props('label') === '自动计算初始字号')

      expect(autoFontToggle).toBeDefined()
      autoFontToggle!.vm.$emit('change', true)
      await flushPromises()

      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })

  it('receives font uploads through the typed file-input boundary', async () => {
    const sidebarSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/SettingsSidebar.vue'),
      'utf8'
    )
    const composableSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/useSettingsSidebar.ts'),
      'utf8'
    )

    expect(sidebarSource).toContain('@files-change="handleFontUpload"')
    expect(sidebarSource).not.toContain('id="fontUpload"')
    expect(composableSource).toContain("import type UiFileInput from '@/components/ui/UiFileInput.vue'")
    expect(composableSource).toContain('ref<InstanceType<typeof UiFileInput> | null>')
    expect(`${sidebarSource}\n${composableSource}`).not.toMatch(
      /target\.files|target\.value\s*=|@change="handleFontUpload"|ref<HTMLInputElement/
    )

    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    const file = new File(['font-bytes'], 'UploadedFont.ttf', { type: 'font/ttf' })
    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await flushPromises()

    expect(uploadFontMock).toHaveBeenCalledWith(file)
    expect(useSettingsStore().fontCatalog).toContainEqual({
      id: 'font-uploaded',
      kind: 'uploaded',
      displayName: 'UploadedFont',
      builtinKey: null,
      assetUrl: '/api/v2/assets/font',
    })
    expect(getFontListMock).not.toHaveBeenCalled()
  })

  it('maps workflow section owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/WorkflowSection.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('maps apply-options section owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/ApplyOptionsSection.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('uses product chips and standard buttons for workflow status and run action', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/WorkflowSection.vue'),
      'utf8'
    )

    expect(source).not.toContain('id="runWorkflowButton"')

    const wrapper = mount(WorkflowSection, {
      props: {
        canRunWorkflow: true,
        isDangerousWorkflow: true,
        rememberWorkflowModeEnabled: false,
        selectedWorkflowMode: 'clear-all',
        workflowContextTag: '全书',
        workflowDescription: '清空所有翻译结果',
        workflowModeOptions: [{ label: '清空全部', value: 'clear-all' }],
        workflowModeTag: '危险操作',
        workflowStartLabel: '清空全部',
      },
    })

    const chips = wrapper.getComponent(ProductChipList)
    expect(wrapper.classes()).toContain('workflow-section')
    expect(chips.props('ariaLabel')).toBe('当前操作模式')
    expect(chips.props('items')).toEqual([
      expect.objectContaining({ label: '全书', tone: 'neutral' }),
      expect.objectContaining({ label: '危险操作', tone: 'danger' }),
    ])

    const runButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '清空全部')
    expect(runButton).toBeDefined()
    expect(runButton!.props('variant')).toBe('danger')
    expect(runButton!.props('block')).toBe(true)
    expect(wrapper.find('.workflow-chip').exists()).toBe(false)
    expect(wrapper.find('.workflow-run-button').exists()).toBe(false)

    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('action-buttons')
    expect(source).not.toContain('workflow-run-button')
    expect(source).not.toContain('--ui-button-')
  })

  it('keeps workflow structure hooks under the workflow owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/WorkflowSection.vue'),
      'utf8'
    )

    for (const currentHook of [
      'workflow-section',
      'workflow-section__mode-field',
      'workflow-section__remember-toggle',
      'workflow-section__meta',
      'workflow-section__run-action',
      'workflow-section__description',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'settings-sidebar__workflow-controls',
      'workflow-controls',
      'workflow-mode-field',
      'remember-workflow-mode-toggle',
      'workflow-meta',
      'workflow-run-action',
      'workflow-description',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
    }
  })

  it('keeps workflow checkbox tests on the public component boundary', () => {
    const sources = [
      'src/components/translate/SettingsSidebar.test.ts',
      'tests/unit/settingsSidebar.workflowPreferences.spec.ts',
    ].map(file => readFileSync(resolve(process.cwd(), file), 'utf8'))

    for (const source of sources) {
      expect(source).not.toMatch(/\.workflow-section__remember-toggle\s+input/)
      expect(source).not.toMatch(/\.remember-workflow-mode-toggle\s+input/)
    }

    const wrapper = mount(WorkflowSection, {
      props: {
        canRunWorkflow: true,
        isDangerousWorkflow: false,
        rememberWorkflowModeEnabled: true,
        selectedWorkflowMode: 'translate-current',
        workflowContextTag: '当前页',
        workflowDescription: '翻译当前页',
        workflowModeOptions: [{ label: '翻译当前页', value: 'translate-current' }],
        workflowModeTag: '翻译',
        workflowStartLabel: '开始翻译',
      },
    })

    const rememberToggle = wrapper.getComponent(UiCheckbox)
    expect(rememberToggle.props('label')).toBe('记住操作模式')
    expect(rememberToggle.props('modelValue')).toBe(true)
  })

  it('keeps SettingsSidebar select stubs typed to option contracts', () => {
    const sources = [
      'tests/unit/settingsSidebar.pageSelection.spec.ts',
      'tests/unit/settingsSidebar.workflowPreferences.spec.ts',
    ].map(file => readFileSync(resolve(process.cwd(), file), 'utf8'))

    for (const source of sources) {
      expect(source).not.toContain('option: ' + 'any')
    }
  })

  it('routes workflow mode labels through typed settings fields', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/WorkflowSection.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('settings-sidebar__field label')

    const wrapper = mount(WorkflowSection, {
      props: {
        canRunWorkflow: true,
        isDangerousWorkflow: false,
        rememberWorkflowModeEnabled: false,
        selectedWorkflowMode: 'translate-current',
        workflowContextTag: '当前页',
        workflowDescription: '翻译当前页',
        workflowModeOptions: [{ label: '翻译当前页', value: 'translate-current' }],
        workflowModeTag: '翻译',
        workflowStartLabel: '开始翻译',
      },
    })

    const workflowModeField = wrapper.findAllComponents(UiField)
      .find(field => field.props('controlId') === 'workflowModeSelect')
    expect(workflowModeField?.props('label')).toBe('操作模式')
  })

  it('uses product action rows and standard buttons for apply options controls', () => {
    const wrapper = mount(ApplyOptionsSection, {
      props: {
        applyOptions: {
          fontSize: true,
          fontFamily: true,
          layoutDirection: true,
          textColor: true,
          fillColor: true,
          strokeEnabled: true,
          strokeColor: true,
          strokeWidth: true,
          lineSpacing: true,
          inlineAlign: true,
          blockAlign: true,
        },
        hasImages: true,
        showApplyOptions: true,
      },
    })

    expect(wrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('批量应用文字设置')

    const actionButtons = wrapper.findAllComponents(UiButton)
      .filter(button => ['应用到全部', ''].includes(button.text()))
    expect(actionButtons.map(button => button.props('variant'))).toEqual(['primary'])
    const optionAction = wrapper.getComponent(UiIconButton)
    expect(optionAction.props('label')).toBe('选择要应用的参数')
    expect(optionAction.props('title')).toBe('选择要应用的参数')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/ApplyOptionsSection.vue'),
      'utf8'
    )
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('settings-sidebar__apply-button-start')
  })

  it('keeps apply-options menu hooks under the apply owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/ApplyOptionsSection.vue'),
      'utf8'
    )

    for (const currentHook of [
      'apply-options-section',
      'apply-options-section__actions',
      'apply-options-section__menu',
      'apply-options-section__option',
      'apply-options-section__divider',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'settings-sidebar__apply-group',
      'settings-sidebar__apply-actions',
      'settings-sidebar__apply-menu',
      'apply-options-dropdown',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
    }

    expect(source).not.toMatch(/\.apply-options-section__menu\s+hr/)
  })

  it('renders apply-options menu from a typed option contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/ApplyOptionsSection.vue'),
      'utf8'
    )

    expect(source).toContain('APPLY_OPTION_ITEMS')
    expect(source).toContain('satisfies ReadonlyArray')
    expect(source).toContain('v-for="option in APPLY_OPTION_ITEMS"')
    expect(source).toContain('applyOptions[option.key]')
    expect(source).toContain("$emit('updateOption', option.key, $event)")
    expect(source).not.toContain('applyOptions.fontSize')
    expect(source).not.toContain("updateOption', 'fontSize'")
    expect(source).not.toContain('applyOptions.strokeWidth')
    expect(source).not.toContain("updateOption', 'strokeWidth'")
  })

  it('links the apply-options trigger to its checklist menu', () => {
    const wrapper = mount(ApplyOptionsSection, {
      props: {
        applyOptions: {
          fontSize: true,
          fontFamily: true,
          layoutDirection: true,
          textColor: true,
          fillColor: true,
          strokeEnabled: true,
          strokeColor: true,
          strokeWidth: true,
          lineSpacing: true,
          inlineAlign: true,
          blockAlign: true,
        },
        hasImages: true,
        showApplyOptions: true,
      },
    })

    const trigger = wrapper.getComponent(UiIconButton)
    expect(trigger.attributes('aria-label')).toBe('选择要应用的参数')
    expect(trigger.attributes('aria-haspopup')).toBe('true')
    expect(trigger.attributes('aria-expanded')).toBe('true')
    expect(trigger.attributes('aria-controls')).toBe('apply-options-section-menu')

    const menu = wrapper.get('#apply-options-section-menu')
    expect(menu.attributes('role')).toBe('group')
    expect(menu.attributes('aria-label')).toBe('可应用的文字设置')
  })
})
