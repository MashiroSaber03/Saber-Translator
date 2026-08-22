import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import MoreSettings from '@/components/settings/MoreSettings.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(promiseResolve => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

const mocks = vi.hoisted(() => ({
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
    fontCatalog: [] as Array<Record<string, unknown>>,
    promptCatalog: [] as Array<Record<string, unknown>>,
    hydrateResourceCatalogs: vi.fn(),
    upsertFont: vi.fn(),
    setRemoveTextWithOcr: vi.fn(),
    setEnableVerboseLogs: vi.fn(),
    setLamaDisableResize: vi.fn(),
  },
}))

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => mocks.settings,
}))

vi.mock('@/api/v2/settings', () => ({
  cleanV2TempFiles: mocks.cleanTemp,
  listV2Fonts: mocks.listFonts,
  uploadV2Font: mocks.uploadFont,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => mocks.toast,
}))

vi.mock('@/composables/usePublicUserAccess', () => ({
  usePublicUserAccess: () => ({
    lamaDisableResizeEditable: () => true,
    lamaDisableResizeValue: () => false,
  }),
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
    mocks.settings.fontCatalog = []
    mocks.settings.promptCatalog = []
    mocks.settings.hydrateResourceCatalogs.mockImplementation((fonts, prompts) => {
      mocks.settings.fontCatalog = fonts
      mocks.settings.promptCatalog = prompts
    })
    mocks.settings.upsertFont.mockImplementation((font) => {
      mocks.settings.fontCatalog = [
        ...mocks.settings.fontCatalog.filter(item => item.id !== font.id),
        font,
      ]
    })
    mocks.listFonts.mockResolvedValue([{
      builtinKey: null,
      displayName: 'Test Font',
      id: 'font-1',
      kind: 'uploaded',
    }])
    mocks.uploadFont.mockResolvedValue({
      id: 'font-1',
      kind: 'uploaded',
      displayName: 'custom',
      builtinKey: null,
      assetUrl: '/api/v2/assets/font-1',
    })
    mocks.cleanTemp.mockResolvedValue({ recovered: 2 })
  })

  it('uploads fonts through the v2 backend asset endpoint', async () => {
    const wrapper = mountSettings()
    const file = new File(['font'], 'custom.ttf', { type: 'font/ttf' })

    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await flushPromises()

    expect(mocks.uploadFont).toHaveBeenCalledWith(file)
    expect(mocks.settings.fontCatalog).toContainEqual({
      id: 'font-1',
      kind: 'uploaded',
      displayName: 'custom',
      builtinKey: null,
      assetUrl: '/api/v2/assets/font-1',
    })
    expect(mocks.listFonts).not.toHaveBeenCalled()
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

  it('does not start an upload while the shared font catalog is refreshing', async () => {
    const fonts = deferred<Array<Record<string, unknown>>>()
    mocks.listFonts.mockReturnValueOnce(fonts.promise)
    const wrapper = mountSettings()

    const refreshButton = wrapper.findAll('button').find(
      candidate => candidate.text() === '刷新字体列表',
    )
    await refreshButton?.trigger('click')

    const fileInput = wrapper.getComponent(UiFileInput)
    const uploadButton = wrapper.get('[data-testid="font-upload-trigger"]')
    expect(fileInput.props('disabled')).toBe(true)
    expect(uploadButton.attributes('disabled')).toBeDefined()

    fileInput.vm.$emit('files-change', [
      new File(['font'], 'overlap.ttf', { type: 'font/ttf' }),
    ])
    await flushPromises()
    expect(mocks.uploadFont).not.toHaveBeenCalled()

    fonts.resolve([])
    await flushPromises()
  })

  it('runs maintenance through v2 backend endpoints', async () => {
    const wrapper = mountSettings()
    const button = wrapper.findAll('button').find(candidate => candidate.text() === '检查并修复')
    await button?.trigger('click')
    await flushPromises()

    expect(mocks.cleanTemp).toHaveBeenCalled()
    expect(mocks.toast.success).toHaveBeenCalledWith('已处理 2 个临时文件记录')
  })

  it('does not expose browser PDF processing or optional auto-save controls', () => {
    const wrapper = mountSettings()

    expect(wrapper.text()).not.toContain('PDF 处理方式')
    expect(wrapper.text()).not.toContain('自动保存')
  })
})
