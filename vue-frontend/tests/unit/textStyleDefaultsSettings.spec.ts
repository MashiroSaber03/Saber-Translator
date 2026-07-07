import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

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
  getDefaultsMock,
  saveDefaultsMock,
  resetDefaultsMock,
  getFontListMock,
  uploadFontMock,
} = vi.hoisted(() => ({
  getDefaultsMock: vi.fn(),
  saveDefaultsMock: vi.fn(),
  resetDefaultsMock: vi.fn(),
  getFontListMock: vi.fn(),
  uploadFontMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  configApi: {
    getTextStyleDefaults: getDefaultsMock,
    saveTextStyleDefaults: saveDefaultsMock,
    resetTextStyleDefaults: resetDefaultsMock,
    getFontList: getFontListMock,
    uploadFont: uploadFontMock,
  },
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
    getDefaultsMock.mockReset()
    saveDefaultsMock.mockReset()
    resetDefaultsMock.mockReset()
    getFontListMock.mockReset()
    uploadFontMock.mockReset()

    getDefaultsMock.mockResolvedValue({ success: true, defaults: { ...initialDefaults } })
    saveDefaultsMock.mockResolvedValue({ success: true, defaults: { ...factoryDefaults } })
    resetDefaultsMock.mockResolvedValue({ success: true, defaults: { ...factoryDefaults } })
    uploadFontMock.mockResolvedValue({ success: true, fontPath: 'fonts/UploadedFont.ttf' })
    getFontListMock.mockResolvedValue({
      fonts: [{
        file_name: '思源黑体SourceHanSansK-Bold.TTF',
        display_name: '思源黑体',
        path: 'fonts/思源黑体SourceHanSansK-Bold.TTF',
        is_default: false,
      }],
    })
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

    expect(getDefaultsMock).toHaveBeenCalledTimes(1)
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

    expect(saveDefaultsMock).not.toHaveBeenCalled()
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
    expect(resetDefaultsMock).toHaveBeenCalledTimes(1)
    expect(result).toEqual({ success: true, changed: true })
    expect(saveDefaultsMock).not.toHaveBeenCalled()
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

    expect(source).toContain('class="text-style-defaults-settings__config-path"')
    expect(source).toContain('.text-style-defaults-settings__config-path')
    expect(source).not.toContain('.text-style-defaults-settings code')
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
    expect(source).toContain('accept=".ttf,.ttc,.otf"\n          hidden')
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

    expect(uploadFontMock).toHaveBeenCalledWith(file)
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
    expect(saveDefaultsMock).toHaveBeenCalledWith({
      ...factoryDefaults,
      fontSize: 35,
    })
    expect(resetDefaultsMock).not.toHaveBeenCalled()
  })

  it('becomes a no-op when current defaults failed to load but the user did not touch text defaults', async () => {
    getDefaultsMock.mockResolvedValue({ success: false, error: 'load failed' })

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
    expect(saveDefaultsMock).not.toHaveBeenCalled()
    expect(resetDefaultsMock).not.toHaveBeenCalled()
  })

  it('still reports an error if the user edits text defaults after load failure', async () => {
    getDefaultsMock.mockResolvedValue({ success: false, error: 'load failed' })

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
    expect(saveDefaultsMock).not.toHaveBeenCalled()
    expect(resetDefaultsMock).not.toHaveBeenCalled()
  })
})
