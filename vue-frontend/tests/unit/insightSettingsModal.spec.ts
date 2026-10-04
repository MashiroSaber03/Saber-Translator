import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { createPinia, setActivePinia, type Pinia } from 'pinia'
import { defineComponent } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import InsightSettingsModal from '@/components/insight/InsightSettingsModal.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { useInsightStore } from '@/stores/insightStore'
import { deepClone } from '@/utils/deepClone'

const apiMocks = vi.hoisted(() => ({
  getGlobalConfig: vi.fn(), saveGlobalConfig: vi.fn(), getDefaultPrompts: vi.fn(), getPromptsLibrary: vi.fn(),
}))
vi.mock('@/api/insight', async importOriginal => ({
  ...await importOriginal<typeof import('@/api/insight')>(), ...apiMocks, __v_isRef: false,
}))

const baseModalStub = defineComponent({
  props: ['showCloseButton', 'closeOnEsc', 'closeOnOverlay'],
  emits: ['close'],
  template: '<div><slot/><slot name="footer"/></div>',
})
const tabNames = ['VlmSettingsTab', 'LlmSettingsTab', 'BatchSettingsTab', 'EmbeddingSettingsTab',
  'RerankerSettingsTab', 'ImageGenSettingsTab', 'PromptsSettingsTab'] as const
const stubs = Object.fromEntries(tabNames.map(name => [name, defineComponent({
  name, emits: ['update:config', 'update:prompts', 'showMessage'], template: '<div/>',
})]))

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => { resolve = res })
  return { promise, resolve }
}

let pinia: Pinia
let store: ReturnType<typeof useInsightStore>
let baseline: ReturnType<typeof store.getConfigForApi>
const wrappers: VueWrapper[] = []
function openSettings() {
  const wrapper = mount(InsightSettingsModal, {
    global: { plugins: [pinia], stubs: { BaseModal: baseModalStub, ...stubs } },
  })
  wrappers.push(wrapper)
  return wrapper
}
function done(wrapper: VueWrapper) {
  return wrapper.findAll('button').find(button => button.text() === '完成')!
}

beforeEach(() => {
  vi.useFakeTimers()
  pinia = createPinia()
  setActivePinia(pinia)
  store = useInsightStore()
  store.updateVlmConfig({ apiKey: 'stored-vlm-key', model: 'stored-vlm-model' })
  store.updateLlmConfig({ apiKey: 'stored-llm-key', model: 'stored-llm-model' })
  store.updateEmbeddingConfig({ apiKey: 'stored-embedding-key' })
  store.updateRerankerConfig({ apiKey: 'stored-reranker-key' })
  store.updateImageGenConfig({ apiKey: 'stored-image-key' })
  store.updatePrompts({ batch_analysis: 'batch', segment_summary: 'segment', chapter_summary: 'chapter', qa_response: 'QA' })
  baseline = store.getConfigForApi()
  apiMocks.getGlobalConfig.mockReset().mockResolvedValue(deepClone(baseline))
  apiMocks.saveGlobalConfig.mockReset().mockImplementation(async snapshot => deepClone(snapshot))
  apiMocks.getDefaultPrompts.mockReset().mockResolvedValue(deepClone(baseline.config.prompts))
  apiMocks.getPromptsLibrary.mockReset().mockResolvedValue([])
})
afterEach(async () => {
  wrappers.splice(0).forEach(wrapper => wrapper.unmount())
  await flushPromises()
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('InsightSettingsModal automatic settings persistence', () => {
  it('shows the seven settings tabs and the same automatic-save footer as translation', async () => {
    const wrapper = openSettings()
    await flushPromises()
    expect(wrapper.getComponent(ProductSegmentedTabs).props('tabs')).toHaveLength(7)
    expect(wrapper.getComponent(ProductActionRow).props('variant')).toBe('dialog')
    expect(wrapper.text()).toContain('修改后自动保存')
    expect(wrapper.findAll('button').map(button => button.text())).not.toContain('保存')
    expect(wrapper.findAll('button').map(button => button.text())).not.toContain('取消')
    expect(done(wrapper).exists()).toBe(true)
  })

  it('does not write settings just because the menu loads or a tab is visited', async () => {
    const wrapper = openSettings()
    await flushPromises()
    for (const tab of wrapper.findAll('[role="tab"]')) await tab.trigger('click')
    await vi.advanceTimersByTimeAsync(1000)
    expect(apiMocks.saveGlobalConfig).not.toHaveBeenCalled()
  })

  it('does not change or save loaded values when every real settings tab mounts', async () => {
    baseline.config.llm.useSameAsVlm = true
    apiMocks.getGlobalConfig.mockResolvedValue(deepClone(baseline))
    const wrapper = mount(InsightSettingsModal, { global: { plugins: [pinia], stubs: { BaseModal: baseModalStub } } })
    wrappers.push(wrapper)
    await flushPromises()
    for (const tab of wrapper.findAll('[role="tab"]')) {
      await tab.trigger('click')
      await flushPromises()
    }
    await vi.advanceTimersByTimeAsync(1000)
    expect(store.getConfigForApi()).toEqual(baseline)
    expect(apiMocks.saveGlobalConfig).not.toHaveBeenCalled()
  })

  it('coalesces typing and leaves the menu open after saving', async () => {
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ model: 'first' })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(300)
    store.updateVlmConfig({ model: 'latest' })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(449)
    expect(apiMocks.saveGlobalConfig).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1)
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledOnce()
    expect(apiMocks.saveGlobalConfig.mock.calls[0][0].config.vlm.model).toBe('latest')
    expect(apiMocks.saveGlobalConfig.mock.calls[0][1]).toEqual(baseline)
    expect(wrapper.emitted('close')).toBeUndefined()
  })

  it.each([
    ['VLM 多模态', 'VlmSettingsTab', 'vlm'],
    ['LLM 对话', 'LlmSettingsTab', 'llm'],
    ['向量模型', 'EmbeddingSettingsTab', 'embedding'],
    ['重排序', 'RerankerSettingsTab', 'reranker'],
    ['生图模型', 'ImageGenSettingsTab', 'imageGen'],
  ] as const)('automatically saves cleared fields from %s', async (label, component, kind) => {
    const wrapper = openSettings()
    await flushPromises()
    await wrapper.findAll('[role="tab"]').find(tab => tab.text().includes(label))!.trigger('click')
    wrapper.getComponent(stubs[component]).vm.$emit('update:config', {
      ...store.config[kind], apiKey: '', model: '', baseUrl: '',
    })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(450)
    const submitted = apiMocks.saveGlobalConfig.mock.calls[0][0].config[kind]
    expect(submitted).toMatchObject({ apiKey: '', model: '', baseUrl: '' })
    expect(store.config[kind]).toMatchObject({ apiKey: '', model: '', baseUrl: '' })
  })

  it('automatically saves batch zero values and cleared prompts', async () => {
    const wrapper = openSettings()
    await flushPromises()
    await wrapper.findAll('[role="tab"]').find(tab => tab.text().includes('批量分析'))!.trigger('click')
    wrapper.getComponent(stubs.BatchSettingsTab).vm.$emit('update:config', {
      ...store.config.batch, pagesPerBatch: 8, contextBatchCount: 0,
    })
    await wrapper.findAll('[role="tab"]').find(tab => tab.text().includes('提示词'))!.trigger('click')
    wrapper.getComponent(stubs.PromptsSettingsTab).vm.$emit('update:prompts', {
      batch_analysis: '', segment_summary: '', chapter_summary: '', qa_response: '',
    })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(450)
    expect(apiMocks.saveGlobalConfig.mock.calls[0][0].config.batch).toMatchObject({ pagesPerBatch: 8, contextBatchCount: 0 })
    expect(apiMocks.saveGlobalConfig.mock.calls[0][0].config.prompts).toEqual({
      batch_analysis: '', segment_summary: '', chapter_summary: '', qa_response: '',
    })
  })

  it('advances only the saved baseline for the next automatic transaction', async () => {
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ model: 'first' })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(450)
    const first = deepClone(apiMocks.saveGlobalConfig.mock.calls[0][0])
    store.updateVlmConfig({ model: 'second' })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(450)
    expect(apiMocks.saveGlobalConfig.mock.calls[1][1]).toEqual(first)
    expect(store.config.vlm.model).toBe('second')
  })

  it('flushes immediately on Done and retains the changed values', async () => {
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ apiKey: '', model: 'changed' })
    await done(wrapper).trigger('click')
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledOnce()
    expect(wrapper.emitted('close')).toHaveLength(1)
    expect(store.config.vlm).toMatchObject({ apiKey: '', model: 'changed' })
    await vi.advanceTimersByTimeAsync(1000)
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledOnce()
  })

  it('keeps edits on failure and retries them when closing again', async () => {
    apiMocks.saveGlobalConfig.mockRejectedValueOnce(new Error('write conflict'))
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ model: 'unsaved-model' })
    await done(wrapper).trigger('click')
    await flushPromises()
    expect(wrapper.emitted('close')).toBeUndefined()
    expect(wrapper.text()).toContain('保存失败')
    expect(store.config.vlm.model).toBe('unsaved-model')
    await done(wrapper).trigger('click')
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledTimes(2)
    expect(apiMocks.saveGlobalConfig.mock.calls[1][1]).toEqual(baseline)
    expect(wrapper.emitted('close')).toHaveLength(1)
  })

  it('allows explicitly closing after a failed save without rolling back the store', async () => {
    apiMocks.saveGlobalConfig.mockRejectedValueOnce(new Error('offline'))
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ model: 'unsaved-model' })
    await done(wrapper).trigger('click')
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text() === '仍然关闭')!.trigger('click')
    expect(wrapper.emitted('close')).toHaveLength(1)
    expect(store.config.vlm.model).toBe('unsaved-model')
  })

  it('keeps fields editable and serializes their latest values while saving', async () => {
    const pending = deferred<typeof baseline>()
    apiMocks.saveGlobalConfig.mockReturnValueOnce(pending.promise)
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ model: 'pending-model' })
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(450)
    expect((wrapper.get('fieldset').element as HTMLFieldSetElement).disabled).toBe(false)
    expect(done(wrapper).attributes('disabled')).toBeDefined()
    expect(wrapper.getComponent(baseModalStub).props()).toMatchObject({ showCloseButton: false, closeOnEsc: false, closeOnOverlay: false })
    wrapper.getComponent(baseModalStub).vm.$emit('close')
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledOnce()
    expect(wrapper.emitted('close')).toBeUndefined()
    store.updateVlmConfig({ model: 'newer-model' })
    pending.resolve(deepClone(apiMocks.saveGlobalConfig.mock.calls[0][0]))
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledTimes(2)
    expect(apiMocks.saveGlobalConfig.mock.calls[1][0].config.vlm.model).toBe('newer-model')
    expect(store.config.vlm.model).toBe('newer-model')
    expect(wrapper.emitted('close')).toBeUndefined()
    expect((wrapper.get('fieldset').element as HTMLFieldSetElement).disabled).toBe(false)
    await done(wrapper).trigger('click')
    await flushPromises()
    expect(wrapper.emitted('close')).toHaveLength(1)
  })

  it('does not mount editable fields or write anything if loading fails', async () => {
    apiMocks.getGlobalConfig.mockRejectedValueOnce(new Error('load failed'))
    const wrapper = openSettings()
    await flushPromises()
    expect(wrapper.find('fieldset').exists()).toBe(false)
    expect(wrapper.text()).toContain('设置加载失败')
    await done(wrapper).trigger('click')
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).not.toHaveBeenCalled()
    expect(wrapper.emitted('close')).toHaveLength(1)
  })

  it('ignores an initial configuration response after unmount', async () => {
    const pending = deferred<typeof baseline>()
    apiMocks.getGlobalConfig.mockReturnValueOnce(pending.promise)
    const wrapper = openSettings()
    wrapper.unmount()
    pending.resolve({ ...deepClone(baseline), config: { ...deepClone(baseline.config), vlm: { ...baseline.config.vlm, model: 'late-model' } } })
    await flushPromises()
    expect(store.config.vlm.model).toBe('stored-vlm-model')
    expect(apiMocks.saveGlobalConfig).not.toHaveBeenCalled()
  })

  it('flushes pending changes on unmount without reverting them', async () => {
    const wrapper = openSettings()
    await flushPromises()
    store.updateVlmConfig({ model: 'last-model' })
    wrapper.unmount()
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledOnce()
    expect(store.config.vlm.model).toBe('last-model')
  })
})
