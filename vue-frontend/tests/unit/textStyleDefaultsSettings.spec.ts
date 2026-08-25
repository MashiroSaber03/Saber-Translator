import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { getTextStyleDefaults } from '@/defaults/textStyleDefaults'

const initialDefaults = {
  fontSize: 26,
  autoFontSize: false,
  fontFamily: 'font-default',
  layoutDirection: 'auto',
  textColor: '#000000',
  fillColor: '#FFFFFF',
  inpaintMethod: 'solid',
  useAutoTextColor: false,
  strokeEnabled: true,
  strokeColor: '#FFFFFF',
  strokeWidth: 3,
  lineSpacing: 1,
  inlineAlign: 'start',
  blockAlign: 'start',
}

const factoryDefaults = getTextStyleDefaults()

const {
  getV2SettingsMock,
  listV2FontsMock,
  saveV2SettingsTransactionMock,
  uploadV2FontMock,
} = vi.hoisted(() => ({
  getV2SettingsMock: vi.fn(),
  listV2FontsMock: vi.fn(),
  saveV2SettingsTransactionMock: vi.fn(),
  uploadV2FontMock: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: getV2SettingsMock,
  listV2Fonts: listV2FontsMock,
  saveV2SettingsTransaction: saveV2SettingsTransactionMock,
  uploadV2Font: uploadV2FontMock,
}))

import TextStyleDefaultsSettings from '@/components/settings/TextStyleDefaultsSettings.vue'

const uiComboboxStub = {
  props: ['modelValue', 'options'],
  template: '<div class="ui-combobox-stub" :data-value="modelValue">{{ options?.length || 0 }}</div>',
}

describe('TextStyleDefaultsSettings', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    getV2SettingsMock.mockReset()
    listV2FontsMock.mockReset()
    saveV2SettingsTransactionMock.mockReset()
    uploadV2FontMock.mockReset()

    const settings = createDefaultSettings()
    settings.textStyle = { ...initialDefaults }
    getV2SettingsMock.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          payload: settings,
          revision: 1,
          schemaVersion: 8,
        },
        {
          domain: 'text_style_defaults',
          payload: initialDefaults,
          revision: 1,
          schemaVersion: 2,
        },
        {
          domain: 'workflow_preferences',
          payload: {
            rememberWorkflowModeEnabled: false,
            lastWorkflowMode: 'translate-current',
          },
          revision: 1,
          schemaVersion: 1,
        },
        {
          domain: 'export_preferences',
          payload: { preserveOriginalFilenames: false },
          revision: 1,
          schemaVersion: 1,
        },
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
    uploadV2FontMock.mockResolvedValue({
      id: 'font-uploaded',
      displayName: 'UploadedFont',
      kind: 'uploaded',
      builtinKey: null,
      assetUrl: '/api/v2/assets/font-uploaded',
    })
    listV2FontsMock.mockResolvedValue([{
      id: 'font-default',
      displayName: '思源黑体',
      kind: 'builtin',
      builtinKey: 'source-han-sans',
      assetUrl: null,
    }])
    useSettingsStore().textStyleDefaults = { ...initialDefaults }
  })

  it('renders the defaults already loaded by the parent settings modal', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()

    expect(getV2SettingsMock).not.toHaveBeenCalled()
    expect((wrapper.get('#textDefaultsFontSize').element as HTMLInputElement).value).toBe('26')
  })

  it('restores factory defaults into the shared parent draft without a standalone write', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')

    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
    expect((wrapper.get('#textDefaultsFontSize').element as HTMLInputElement).value).toBe(
      String(factoryDefaults.fontSize),
    )
    expect(useSettingsStore().textStyleDefaults).toEqual(factoryDefaults)
  })

  it('publishes modified defaults directly into the parent settings draft', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')

    expect(useSettingsStore().textStyleDefaults).toEqual(factoryDefaults)
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
    expect(wrapper.emitted('save-complete')).toBeUndefined()
  })

  it('uses fixed select primitives for layout, alignment, and fill method fields', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()

    const optionValues = wrapper.findAllComponents(UiSelect).map(select =>
      (select.props('options') || []).map((option: { value: string | number }) => option.value)
    )

    expect(optionValues).toContainEqual(expect.arrayContaining(['auto', 'vertical', 'horizontal']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['start', 'center', 'end']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['solid', 'lama_mpe', 'litelama']))
  })

  it('ignores values outside the backend text-style contracts', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })
    await flushPromises()

    const store = useSettingsStore()
    const before = { ...store.textStyleDefaults }
    const selects = wrapper.findAllComponents(UiSelect)
    selects[0]!.vm.$emit('change', 1)
    selects[1]!.vm.$emit('change', 'middle')
    selects[2]!.vm.$emit('change', 'middle')
    selects[3]!.vm.$emit('change', 'legacy')

    const numberFields = wrapper.findAllComponents(UiNumberField)
    const field = (inputId: string) => numberFields.find(item => item.props('inputId') === inputId)!
    field('textDefaultsFontSize').vm.$emit('change', 1.5)
    field('textDefaultsLineSpacing').vm.$emit('change', 0)
    field('textDefaultsStrokeWidth').vm.$emit('change', 1.5)
    await flushPromises()

    expect(store.textStyleDefaults).toEqual(before)
  })

  it('does not impose frontend-only upper bounds on text-style values', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })
    await flushPromises()

    const numberFields = wrapper.findAllComponents(UiNumberField)
    const field = (inputId: string) => numberFields.find(
      item => item.props('inputId') === inputId,
    )!
    field('textDefaultsFontSize').vm.$emit('change', 1024)
    field('textDefaultsLineSpacing').vm.$emit('change', 12.5)
    field('textDefaultsStrokeWidth').vm.$emit('change', 80)
    await flushPromises()

    expect(useSettingsStore().textStyleDefaults).toMatchObject({
      fontSize: 1024,
      lineSpacing: 12.5,
      strokeWidth: 80,
    })
    expect(field('textDefaultsFontSize').props('max')).toBeUndefined()
    expect(field('textDefaultsLineSpacing').props('max')).toBeUndefined()
    expect(field('textDefaultsStrokeWidth').props('max')).toBeUndefined()
  })

  it('routes text-style default labels and feedback through typed settings primitives', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('ui-form-hint')
    expect(source).not.toContain('class="action-row"')
    expect(source).toContain('ProductStatusBanner')
    expect(source).toContain('ProductActionRow')

    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })
    await flushPromises()

    const fields = wrapper.findAllComponents(UiField)
    const fieldByControlId = (controlId: string) =>
      fields.find(field => field.props('controlId') === controlId)

    expect(fieldByControlId('textDefaultsFontSize')?.props('label')).toBe('字号')
    expect(fieldByControlId('textDefaultsAutoFontSize')?.props('label')).toBe('自动计算初始字号')
    expect(fieldByControlId('textDefaultsFontFamily')?.props('label')).toBe('文本字体')
    expect(fieldByControlId('textDefaultsLayoutDirection')?.props('label')).toBe('排版方向')
    expect(fieldByControlId('textDefaultsInlineAlign')?.props('label')).toBe('行内对齐')
    expect(fieldByControlId('textDefaultsBlockAlign')?.props('label')).toBe('文本块对齐')
    expect(fieldByControlId('textDefaultsLineSpacing')?.props('hint')).toBe('行间距倍数，必须大于 0。')
    expect(fieldByControlId('textDefaultsUseAutoTextColor')?.props('label')).toBe('自动识别文字颜色')
    expect(fieldByControlId('textDefaultsTextColor')?.props('label')).toBe('文字颜色')
    expect(fieldByControlId('textDefaultsInpaintMethod')?.props('label')).toBe('气泡填充方式')
    expect(fieldByControlId('textDefaultsFillColor')?.props('label')).toBe('填充颜色')
    expect(fieldByControlId('textDefaultsStrokeEnabled')?.props('label')).toBe('启用描边')
    expect(fieldByControlId('textDefaultsStrokeColor')?.props('label')).toBe('描边颜色')
    expect(fieldByControlId('textDefaultsStrokeWidth')?.props('hint')).toBe('0 表示无描边。')
  })

  it('styles the config path through an explicit owner hook', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).toContain('后端数据库中的全局默认文字设置')
    expect(source).not.toContain('config/text_style_defaults.json')
    expect(source).not.toContain('BUILTIN_FONTS')
  })

  it('uses the shared color input primitive for default color fields', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).toContain("import UiColorInput from '@/components/ui/UiColorInput.vue'")
    expect(source).not.toContain('type="color"')
    expect(source).not.toMatch(/<UiInput[\s\S]*?textDefaults(Text|Fill|Stroke)Color/)
  })

  it('uses the file-input hidden prop instead of inline display styles', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).not.toContain('style="display: none"')
    expect(source).toMatch(/:accept="FONT_FILE_ACCEPT"[\s\S]*?\shidden/)
  })

  it('receives custom fonts through the typed file-input boundary', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).toContain('@files-change="handleFontUpload"')
    expect(source).toContain('ref<InstanceType<typeof UiFileInput> | null>')
    expect(source).not.toMatch(/target\.files|target\.value\s*=|@change="handleFontUpload"|ref<HTMLInputElement/)

    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })
    await flushPromises()

    const file = new File(['font-bytes'], 'UploadedFont.ttf', { type: 'font/ttf' })
    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await flushPromises()

    expect(uploadV2FontMock).toHaveBeenCalledWith(file)
    expect(listV2FontsMock).toHaveBeenCalledTimes(1)
    expect(useSettingsStore().fontCatalog).toContainEqual({
      id: 'font-uploaded',
      displayName: 'UploadedFont',
      kind: 'uploaded',
      builtinKey: null,
      assetUrl: '/api/v2/assets/font-uploaded',
    })
    expect(wrapper.get('.ui-combobox-stub').attributes('data-value')).toBe('font-uploaded')
    expect(useSettingsStore().textStyleDefaults.fontFamily).toBe('font-uploaded')
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })

  it('uses normal save when the user edits fields after resetting to factory defaults', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')
    wrapper.getComponent({ name: 'UiCheckbox' }).vm.$emit('change', false)
    await wrapper.vm.$nextTick()
    await wrapper.get('#textDefaultsFontSize').setValue('35')

    expect(useSettingsStore().textStyleDefaults).toEqual({
      ...factoryDefaults,
      autoFontSize: false,
      fontSize: 35,
    })
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })

  it('shows a font-catalog error and leaves the parent draft unchanged', async () => {
    listV2FontsMock.mockRejectedValue(new Error('font list failed'))
    const before = { ...useSettingsStore().textStyleDefaults }

    const wrapper = mount(TextStyleDefaultsSettings, {
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()

    expect(wrapper.text()).toContain('font list failed')
    expect(useSettingsStore().textStyleDefaults).toEqual(before)
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })

  it('does not expose the removed child save handshake', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).not.toContain('saveRequestId')
    expect(source).not.toContain('save-complete')
    expect(source).not.toContain('defineExpose')
    expect(source).toContain('settingsStore.textStyleDefaults = normalized')
  })
})
