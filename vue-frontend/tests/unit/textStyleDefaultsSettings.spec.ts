import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

const initialDefaults = {
  fontSize: 26,
  autoFontSize: false,
  fontFamily: 'fonts/思源黑体SourceHanSansK-Bold.TTF',
  layoutDirection: 'auto',
  textColor: '#000000',
  fillColor: '#FFFFFF',
  inpaintMethod: 'solid',
  useAutoTextColor: false,
  strokeEnabled: true,
  strokeColor: '#FFFFFF',
  strokeWidth: 3,
  lineSpacing: 1,
  textAlign: 'start',
}

const factoryDefaults = {
  ...initialDefaults,
  fontSize: 31,
  textColor: '#223344',
}

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

vi.mock('@/defaults/textStyleFactoryDefaults', () => ({
  getFactoryTextStyleDefaults: () => ({ ...factoryDefaults }),
}))

import TextStyleDefaultsSettings from '@/components/settings/TextStyleDefaultsSettings.vue'

const uiComboboxStub = {
  props: ['modelValue', 'options'],
  template: '<div class="ui-combobox-stub" :data-value="modelValue">{{ options?.length || 0 }}</div>',
}

async function requestDefaultsSave(wrapper: ReturnType<typeof mount>, requestId = 1) {
  await wrapper.setProps({ saveRequestId: requestId, 'save-request-id': requestId })
  expect(wrapper.props('saveRequestId')).toBe(requestId)
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await flushPromises()
    const emissions = wrapper.emitted('save-complete') ?? wrapper.emitted('saveComplete')
    const latest = emissions?.at(-1)?.[0]
    if (latest) return latest
  }
  return undefined
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
      settings: [{
        domain: 'translation',
        payload: settings,
        revision: 1,
        schemaVersion: 3,
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
    uploadV2FontMock.mockResolvedValue({
      id: 'font-uploaded',
      assetUrl: '/api/v2/assets/font-uploaded',
    })
    listV2FontsMock.mockResolvedValue([{
      id: 'fonts/思源黑体SourceHanSansK-Bold.TTF',
      displayName: '思源黑体',
      kind: 'builtin',
      builtinKey: 'source-han-sans',
      assetUrl: null,
    }])
  })

  it('loads current defaults when the settings modal opens', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()

    expect(getV2SettingsMock).toHaveBeenCalledTimes(1)
    expect((wrapper.get('#textDefaultsFontSize').element as HTMLInputElement).value).toBe('26')
  })

  it('restores factory defaults into draft only until save is called', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')

    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
    expect((wrapper.get('#textDefaultsFontSize').element as HTMLInputElement).value).toBe('31')
  })

  it('saves modified draft defaults through the typed save request contract', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true, saveRequestId: 0 },
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')

    const result = await requestDefaultsSave(wrapper)
    expect(result).toEqual({ success: true, changed: true })
    expect(useSettingsStore().settings.textStyle).toEqual(factoryDefaults)
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })

  it('uses fixed select primitives for layout, alignment, and fill method fields', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true, saveRequestId: 0 },
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

  it('routes text-style default labels and feedback through typed settings primitives', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('ui-form-hint')
    expect(source).not.toContain('class="action-row"')
    expect(source).toContain('ProductStatusBanner')
    expect(source).toContain('ProductActionRow')

    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true, saveRequestId: 0 },
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
    expect(fieldByControlId('textDefaultsTextAlign')?.props('label')).toBe('对齐方式')
    expect(fieldByControlId('textDefaultsLineSpacing')?.props('hint')).toBe('行间距倍数（0.5 - 3.0）')
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
    expect(source).toMatch(/accept="\.ttf,\.otf,\.woff,\.woff2"\s+hidden/)
  })

  it('receives custom fonts through the typed file-input boundary', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'), 'utf8')

    expect(source).toContain('@files-change="handleFontUpload"')
    expect(source).toContain('ref<InstanceType<typeof UiFileInput> | null>')
    expect(source).not.toMatch(/target\.files|target\.value\s*=|@change="handleFontUpload"|ref<HTMLInputElement/)

    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true, saveRequestId: 0 },
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
    expect(wrapper.get('.ui-combobox-stub').attributes('data-value')).toBe('font-uploaded')
    expect(await requestDefaultsSave(wrapper)).toEqual({
      success: true,
      changed: true,
    })
    expect(useSettingsStore().settings.textStyle.fontFamily).toBe('font-uploaded')
  })

  it('uses normal save when the user edits fields after resetting to factory defaults', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')
    await wrapper.get('#textDefaultsFontSize').setValue('35')

    expect(await requestDefaultsSave(wrapper)).toEqual({ success: true, changed: true })
    expect(useSettingsStore().settings.textStyle).toEqual({
      ...factoryDefaults,
      fontSize: 35,
    })
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })

  it('becomes a no-op when current defaults failed to load but the user did not touch text defaults', async () => {
    getV2SettingsMock.mockRejectedValue(new Error('load failed'))

    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()

    expect(await requestDefaultsSave(wrapper)).toEqual({
      success: true,
      changed: false,
    })
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })

  it('still reports an error if the user edits text defaults after load failure', async () => {
    getV2SettingsMock.mockRejectedValue(new Error('load failed'))

    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          UiCombobox: uiComboboxStub,
        },
      },
    })

    await flushPromises()
    await wrapper.get('#textDefaultsFontSize').setValue('40')

    expect(await requestDefaultsSave(wrapper)).toEqual({
      success: false,
      changed: false,
      error: '请先成功加载当前默认值，或先点击“恢复出厂默认”再保存'
    })
    expect(saveV2SettingsTransactionMock).not.toHaveBeenCalled()
  })
})
