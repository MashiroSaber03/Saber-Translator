import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import MoreSettings from '@/components/settings/MoreSettings.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

const mocks = vi.hoisted(() => ({
  cleanDebug: vi.fn(),
  cleanTemp: vi.fn(),
  listFonts: vi.fn(),
  uploadFont: vi.fn(),
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
  settings: {
    settings: {
      removeTextWithOcr: false,
      enableVerboseLogs: false,
      lamaDisableResize: false,
    },
    setRemoveTextWithOcr: vi.fn(),
    setEnableVerboseLogs: vi.fn(),
    setLamaDisableResize: vi.fn(),
  },
}))

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => mocks.settings,
}))

vi.mock('@/api/v2/settings', () => ({
  cleanV2DebugFiles: mocks.cleanDebug,
  cleanV2TempFiles: mocks.cleanTemp,
  listV2Fonts: mocks.listFonts,
  uploadV2Font: mocks.uploadFont,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => mocks.toast,
}))

function mountSettings() {
  return mount(MoreSettings, {
    global: {
      stubs: {
        ParallelSettings: true,
      },
    },
  })
}

describe('MoreSettings backend controls', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listFonts.mockResolvedValue([{
      builtinKey: null,
      displayName: 'Test Font',
      id: 'font-1',
      kind: 'uploaded',
    }])
    mocks.uploadFont.mockResolvedValue({ id: 'font-1' })
    mocks.cleanDebug.mockResolvedValue({ removed: 3 })
    mocks.cleanTemp.mockResolvedValue({ recovered: 2 })
  })

  it('uploads fonts through the v2 backend asset endpoint', async () => {
    const wrapper = mountSettings()
    const file = new File(['font'], 'custom.ttf', { type: 'font/ttf' })

    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await flushPromises()

    expect(mocks.uploadFont).toHaveBeenCalledWith(file)
    expect(mocks.listFonts).toHaveBeenCalled()
    expect(mocks.toast.success).toHaveBeenCalledWith('字体 "custom.ttf" 上传成功')
  })

  it('rejects unsupported font extensions before upload', async () => {
    const wrapper = mountSettings()

    wrapper.getComponent(UiFileInput).vm.$emit(
      'files-change',
      [new File(['bad'], 'font.exe')],
    )
    await flushPromises()

    expect(mocks.uploadFont).not.toHaveBeenCalled()
    expect(mocks.toast.error).toHaveBeenCalled()
  })

  it('runs maintenance through v2 backend endpoints', async () => {
    const wrapper = mountSettings()
    const buttons = wrapper.findAll('button')
    await buttons.find(button => button.text() === '清理调试文件')?.trigger('click')
    await flushPromises()
    await buttons.find(button => button.text() === '清理临时文件')?.trigger('click')
    await flushPromises()

    expect(mocks.cleanDebug).toHaveBeenCalled()
    expect(mocks.cleanTemp).toHaveBeenCalled()
    expect(mocks.toast.success).toHaveBeenCalledWith('已清理 3 个调试文件')
    expect(mocks.toast.success).toHaveBeenCalledWith('已恢复或清理 2 个临时记录')
  })

  it('does not expose browser PDF processing or optional auto-save controls', () => {
    const wrapper = mountSettings()

    expect(wrapper.text()).not.toContain('PDF 处理方式')
    expect(wrapper.text()).not.toContain('自动保存')
  })
})
