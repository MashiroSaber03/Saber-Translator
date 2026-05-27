import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

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

vi.mock('@/stores/settingsStore', () => ({
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

    getFontListMock.mockResolvedValue({ fonts: ['fonts/TestFont.ttf'] })
    uploadFontMock.mockResolvedValue({ success: true, fontPath: 'fonts/TestFont.ttf' })
  })

  it('renders a styled upload trigger with a hidden file input', () => {
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            template: '<div class="custom-select-stub" />',
          },
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
    expect(input.classes()).toContain('visually-hidden-file-input')
    expect(fileName.text()).toBe('未选择文件')
  })

  it('shows the selected file name after choosing a custom font', async () => {
    const wrapper = mount(MoreSettings, {
      global: {
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            template: '<div class="custom-select-stub" />',
          },
          ParallelSettings: {
            name: 'ParallelSettings',
            template: '<div class="parallel-settings-stub" />',
          },
        },
      },
    })

    const fileInput = wrapper.get('[data-testid="font-upload-input"]')
    const file = new File(['font-bytes'], 'MyCustomFont.ttf', { type: 'font/ttf' })

    Object.defineProperty(fileInput.element, 'files', {
      configurable: true,
      value: [file],
    })

    await fileInput.trigger('change')
    await flushPromises()

    expect(uploadFontMock).toHaveBeenCalledWith(file)
    expect(wrapper.get('[data-testid="font-upload-filename"]').text()).toBe('MyCustomFont.ttf')
  })
})
