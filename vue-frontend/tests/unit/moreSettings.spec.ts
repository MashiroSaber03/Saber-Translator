import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import UiField from '@/components/ui/UiField.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

const {
  getFontListMock,
  uploadFontMock,
  cleanDebugFilesMock,
  cleanTempFilesMock,
  toastSuccessMock,
  toastErrorMock,
  settingsStoreMock,
} = vi.hoisted(() => ({
  getFontListMock: vi.fn(),
  uploadFontMock: vi.fn(),
  cleanDebugFilesMock: vi.fn(),
  cleanTempFilesMock: vi.fn(),
  toastSuccessMock: vi.fn(),
  toastErrorMock: vi.fn(),
  settingsStoreMock: {
    settings: {
      pdfProcessingMethod: 'frontend',
      autoSaveInBookshelfMode: false,
      removeTextWithOcr: false,
      enableVerboseLogs: false,
      lamaDisableResize: false,
    },
    setPdfProcessingMethod: vi.fn(),
    setAutoSaveInBookshelfMode: vi.fn(),
    setRemoveTextWithOcr: vi.fn(),
    setEnableVerboseLogs: vi.fn(),
    setLamaDisableResize: vi.fn(),
  },
}))

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

vi.mock('@/api/config', () => ({
  configApi: {
    getFontList: getFontListMock,
    uploadFont: uploadFontMock,
  },
}))

vi.mock('@/api/system', () => ({
  cleanDebugFiles: cleanDebugFilesMock,
  cleanTempFiles: cleanTempFilesMock,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}))

import MoreSettings from '@/components/settings/MoreSettings.vue'

const componentSourcePath = resolve(process.cwd(), 'src/components/settings/MoreSettings.vue')

describe('MoreSettings font upload UI', () => {
  beforeEach(() => {
    getFontListMock.mockReset()
    uploadFontMock.mockReset()
    cleanDebugFilesMock.mockReset()
    cleanTempFilesMock.mockReset()
    toastSuccessMock.mockReset()
    toastErrorMock.mockReset()
    settingsStoreMock.setPdfProcessingMethod.mockReset()
    settingsStoreMock.setAutoSaveInBookshelfMode.mockReset()
    settingsStoreMock.setRemoveTextWithOcr.mockReset()
    settingsStoreMock.setEnableVerboseLogs.mockReset()
    settingsStoreMock.setLamaDisableResize.mockReset()

    getFontListMock.mockResolvedValue({
      fonts: [{
        file_name: 'TestFont.ttf',
        display_name: 'TestFont',
        path: 'fonts/TestFont.ttf',
        is_default: false,
      }],
    })
    uploadFontMock.mockResolvedValue({ success: true, fontPath: 'fonts/TestFont.ttf' })
  })

  it('renders a styled upload trigger with a hidden file input', () => {
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          ParallelSettings: {
            name: 'ParallelSettings',
            template: '<div class="parallel-settings-stub" />',
          },
        },
      },
    })

    const trigger = wrapper.get('[data-testid="font-upload-trigger"]')
    const input = wrapper.get('[data-testid="font-upload-input"]')
    const fileName = wrapper.get('[data-testid="font-upload-filename"]')

    expect(trigger.text()).toContain('选择字体文件')
    expect(input.attributes('accept')).toBe('.ttf,.ttc,.otf')
    expect(input.classes()).toContain('more-settings__hidden-file-input')
    expect(fileName.text()).toBe('未选择文件')
  })

  it('receives font uploads through the typed file-input boundary', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain('@files-change="handleFontUpload"')
    expect(source).toContain('ref<InstanceType<typeof UiFileInput> | null>')
    expect(source).not.toMatch(/target\.files|target\.value\s*=|@change="handleFontUpload"/)
  })

  it('keeps settings action copy free of decorative emoji', () => {
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          ParallelSettings: {
            name: 'ParallelSettings',
            template: '<div class="parallel-settings-stub" />',
          },
        },
      },
    })

    const renderedText = wrapper.text()
    for (const decorativeEmoji of ['🔄', '🗑️', '📖', '🐙', '⚠️']) {
      expect(renderedText).not.toContain(decorativeEmoji)
    }
  })

  it('uses a fixed select primitive for PDF processing mode', () => {
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          ParallelSettings: {
            name: 'ParallelSettings',
            template: '<div class="parallel-settings-stub" />',
          },
        },
      },
    })

    const pdfSelect = wrapper.getComponent(UiSelect)
    expect(pdfSelect.props('modelValue')).toBe('frontend')
    expect(pdfSelect.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'frontend' }),
      expect.objectContaining({ value: 'backend' }),
    ]))
  })

  it('keeps miscellaneous settings labels and hints on the typed field API', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          ParallelSettings: {
            name: 'ParallelSettings',
            template: '<div class="parallel-settings-stub" />',
          },
        },
      },
    })

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')
    expect(source).not.toContain('hint-note')

    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map((field) => field.props('label')).filter(Boolean)).toEqual(expect.arrayContaining([
      'PDF处理方式',
      '系统字体列表',
      '上传自定义字体',
      '清理调试文件',
      '清理临时文件',
    ]))
    expect(fields.map((field) => field.props('hint')).filter(Boolean)).toEqual(expect.arrayContaining([
      '前端处理速度更快，后端处理适配性更好',
      '支持 .ttf, .ttc, .otf 格式',
      '清理调试过程中生成的临时文件',
      '清理下载和处理过程中的临时文件',
    ]))
  })

  it('keeps miscellaneous settings local visuals under MoreSettings owner hooks', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    for (const requiredClass of [
      'more-settings__font-count',
      'more-settings__font-upload-row',
      'more-settings__hidden-file-input',
      'more-settings__font-upload-filename',
      'more-settings__about',
      'more-settings__about-title',
      'more-settings__about-description',
      'more-settings__about-links',
      'more-settings__about-link',
      'more-settings__about-disclaimer',
    ]) {
      expect(classTokens).toContain(requiredClass)
    }

    for (const forbiddenSelector of [
      '.about-info p',
      '.about-info .links',
      '.about-info .links a',
      '.about-info .disclaimer',
    ]) {
      expect(source).not.toContain(forbiddenSelector)
    }

    expect(source).not.toContain('class="about-info"')
    expect(source).not.toContain('class="links"')
    expect(source).not.toContain('class="disclaimer"')
    expect(source).not.toContain('class="font-upload-row"')
    expect(source).not.toContain('class="font-count"')
  })

  it('shows the selected file name after choosing a custom font', async () => {
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          ParallelSettings: {
            name: 'ParallelSettings',
            template: '<div class="parallel-settings-stub" />',
          },
        },
      },
    })

    const file = new File(['font-bytes'], 'MyCustomFont.ttf', { type: 'font/ttf' })

    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await flushPromises()

    expect(uploadFontMock).toHaveBeenCalledWith(file)
    expect(wrapper.get('[data-testid="font-upload-filename"]').text()).toBe('MyCustomFont.ttf')
  })
})
