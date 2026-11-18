import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LocalModelsSettings from './LocalModelsSettings.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { createPinia, setActivePinia } from 'pinia'
import { useRuntimeStore } from '@/stores/runtimeStore'
import type { LocalModelCatalog } from '@/api/v2/models'

const api = vi.hoisted(() => ({ getLocalModels: vi.fn(), installLocalModels: vi.fn(), cancelLocalModelInstall: vi.fn() }))
vi.mock('@/api/v2/models', () => api)

function catalog(): LocalModelCatalog {
  return {
    directory: 'C:/app/models',
    sources: [{ id: 'auto', label: '自动', configured: true }, { id: 'modelscope', label: '魔搭社区', configured: false }, { id: 'github', label: 'GitHub', configured: true }],
    models: [
      { id: 'detector_default', label: 'Default', description: '文字检测', sizeBytes: 100, installed: true, directory: 'models/default', partialBytes: 0 },
      { id: 'manga_ocr', label: 'MangaOCR', description: '文字识别', sizeBytes: 1000, installed: false, directory: 'models/manga_ocr', partialBytes: 500 },
    ],
    task: { status: 'idle', modelIds: [], results: {} },
  }
}

describe('model management', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const runtime = useRuntimeStore()
    runtime.assumeLocalForTests()
    vi.spyOn(runtime, 'load').mockResolvedValue(runtime.capabilities!)
    vi.useFakeTimers()
    vi.clearAllMocks()
    api.getLocalModels.mockImplementation(async () => catalog())
    api.installLocalModels.mockResolvedValue({ status: 'completed', modelIds: [], results: {} })
    api.cancelLocalModelInstall.mockResolvedValue({ status: 'cancelled', modelIds: [], results: {} })
  })
  afterEach(() => vi.useRealTimers())

  it('never downloads on mount and disables unconfigured sources without idle polling', async () => {
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    expect(api.installLocalModels).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('C:/app/models')
    expect(wrapper.findComponent(UiSelect).props('options')).toContainEqual({ value: 'modelscope', label: '魔搭社区（尚未配置）', disabled: true })
    await vi.advanceTimersByTimeAsync(5000)
    expect(api.getLocalModels).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('refreshes an idle visible list after another window changes installations', async () => {
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    const installed = catalog()
    installed.models[1]!.installed = true
    api.getLocalModels.mockResolvedValueOnce(installed)
    useRuntimeStore().capabilities!.installedModels = ['detector_default', 'manga_ocr']
    await flushPromises()
    expect(api.getLocalModels).toHaveBeenCalledTimes(2)
    expect(wrapper.get('[data-model-id="manga_ocr"]').text()).toContain('已安装')
    wrapper.unmount()
  })

  it('uses one installer for missing models and verification', async () => {
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text().includes('一键安装全部'))!.trigger('click')
    await flushPromises()
    expect(api.installLocalModels).toHaveBeenLastCalledWith({ modelIds: ['manga_ocr'], source: 'auto', verify: false })
    await wrapper.get('[aria-label="校验 Default"]').trigger('click')
    await flushPromises()
    expect(api.installLocalModels).toHaveBeenLastCalledWith({ modelIds: ['detector_default'], source: 'auto', verify: true })
    wrapper.unmount()
  })

  it('keeps failures readable and enables retry', async () => {
    api.installLocalModels.mockRejectedValueOnce(new Error('磁盘空间不足'))
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    await wrapper.get('[aria-label="下载或修复 MangaOCR"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('磁盘空间不足')
    expect(wrapper.get('[aria-label="下载或修复 MangaOCR"]').attributes('disabled')).toBeUndefined()
    wrapper.unmount()
  })

  it('stops polling on hidden tabs and unmount without cancelling backend work', async () => {
    api.getLocalModels.mockImplementation(async () => ({ ...catalog(), task: { status: 'running', modelIds: ['manga_ocr'], results: {}, phase: 'downloading', currentModelId: 'manga_ocr', totalBytes: 1000, downloadedBytes: 500 } }))
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    expect(wrapper.get('progress').attributes('value')).toBe('50')
    await vi.advanceTimersByTimeAsync(1000)
    expect(api.getLocalModels).toHaveBeenCalledTimes(2)
    await wrapper.setProps({ active: false })
    await vi.advanceTimersByTimeAsync(4000)
    expect(api.getLocalModels).toHaveBeenCalledTimes(2)
    await wrapper.setProps({ active: true })
    await flushPromises()
    expect(api.getLocalModels).toHaveBeenCalledTimes(3)
    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(4000)
    expect(api.getLocalModels).toHaveBeenCalledTimes(3)
    expect(api.cancelLocalModelInstall).not.toHaveBeenCalled()
  })

  it('cancels only after an explicit click', async () => {
    api.getLocalModels.mockImplementationOnce(async () => ({ ...catalog(), task: { status: 'running', modelIds: ['manga_ocr'], results: {} } }))
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text() === '取消下载')!.trigger('click')
    await flushPromises()
    expect(api.cancelLocalModelInstall).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('does not resume polling when an install response arrives after the dialog closes', async () => {
    let acceptInstall!: (value: unknown) => void
    api.getLocalModels.mockResolvedValueOnce(catalog()).mockResolvedValue({ ...catalog(), task: { status: 'running', modelIds: ['manga_ocr'], results: {} } })
    api.installLocalModels.mockImplementationOnce(() => new Promise(resolve => { acceptInstall = resolve }))
    const wrapper = mount(LocalModelsSettings, { props: { active: true } })
    await flushPromises()
    await wrapper.get('[aria-label="下载或修复 MangaOCR"]').trigger('click')
    wrapper.unmount()
    acceptInstall({ status: 'running', modelIds: ['manga_ocr'], results: {} })
    await flushPromises()
    await vi.advanceTimersByTimeAsync(4000)
    expect(api.getLocalModels).toHaveBeenCalledTimes(1)
  })
})
