import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, onMounted, watch } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import InsightSettingsModal from '@/components/insight/InsightSettingsModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { useInsightStore } from '@/stores/insightStore'

const apiMocks = vi.hoisted(() => ({
  getGlobalConfig: vi.fn(),
  saveGlobalConfig: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getGlobalConfig: apiMocks.getGlobalConfig,
  saveGlobalConfig: apiMocks.saveGlobalConfig,
}))

const baseModalStub = defineComponent({
  emits: ['close'],
  template: '<div><slot /><slot name="footer" /></div>',
})

function createSettingsTabStub(payload: Record<string, unknown> = {}) {
  return defineComponent({
    props: {
      syncRequestId: {
        type: Number,
        default: 0,
      },
    },
    emits: ['update:config'],
    setup(props, { emit }) {
      const emitDraft = () => emit('update:config', payload)
      onMounted(emitDraft)
      watch(() => props.syncRequestId, emitDraft)
      return {}
    },
    template: '<div />',
  })
}

function createPromptsTabStub(payload: Record<string, string> = {}) {
  return defineComponent({
    props: {
      syncRequestId: {
        type: Number,
        default: 0,
      },
    },
    emits: ['update:prompts'],
    setup(props, { emit }) {
      const emitDraft = () => emit('update:prompts', payload)
      onMounted(emitDraft)
      watch(() => props.syncRequestId, emitDraft)
      return {}
    },
    template: '<div />',
  })
}

const messageSettingsTabStub = defineComponent({
  props: {
    syncRequestId: {
      type: Number,
      default: 0,
    },
  },
  emits: ['showMessage', 'update:config'],
  setup(_, { emit }) {
    onMounted(() => emit('update:config', {}))
    return {}
  },
  template:
    '<button type="button" class="emit-message" @click="$emit(\'showMessage\', \'连接成功\', \'success\')">emit</button>',
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => {
    resolve = res
  })
  return { promise, resolve }
}

describe('InsightSettingsModal', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setActivePinia(createPinia())
    apiMocks.getGlobalConfig.mockReset().mockResolvedValue(useInsightStore().getConfigForApi())
    apiMocks.saveGlobalConfig.mockReset().mockImplementation(async config => config)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('clears delayed message and save-success timers when the modal unmounts', async () => {
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()

    const saveButton = wrapper.findAll('button').find(button => button.text() === '保存')
    expect(saveButton).toBeTruthy()

    await saveButton!.trigger('click')
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalled()

    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
    wrapper.unmount()

    expect(clearTimeoutSpy.mock.calls.length).toBeGreaterThanOrEqual(2)
  })

  it('uses product segmented tabs for analysis settings sections', async () => {
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()

    expect(wrapper.find('.settings-tabs').exists()).toBe(false)
    expect(wrapper.find('.product-segmented-tabs').exists()).toBe(true)

    const llmTab = wrapper.findAll('[role="tab"]').find(tab => tab.text().includes('LLM 对话'))
    expect(llmTab).toBeTruthy()

    await llmTab!.trigger('click')
    expect(wrapper.find('[role="tab"][aria-selected="true"]').text()).toContain('LLM 对话')
  })

  it('uses product feedback for settings messages', async () => {
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: messageSettingsTabStub,
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()
    await wrapper.get('.emit-message').trigger('click')

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('success')
    expect(banner.text()).toContain('连接成功')
    expect(wrapper.find('.test-message').exists()).toBe(false)
  })

  it('uses the product dialog action row for modal footer actions', async () => {
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('漫画分析设置操作')
  })

  it('uses typed draft events instead of child exposed methods for saving settings', async () => {
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub({
            provider: 'custom-vlm',
            apiKey: 'vlm-key',
            model: 'vlm-model',
          }),
          LlmSettingsTab: createSettingsTabStub({
            provider: 'custom-llm',
            apiKey: 'llm-key',
            model: 'llm-model',
          }),
          BatchSettingsTab: createSettingsTabStub({ pagesPerBatch: 8, contextBatchCount: 4 }),
          EmbeddingSettingsTab: createSettingsTabStub({
            provider: 'custom-embedding',
            apiKey: 'embedding-key',
            model: 'embedding-model',
          }),
          RerankerSettingsTab: createSettingsTabStub({
            provider: 'custom-reranker',
            apiKey: 'reranker-key',
            model: 'reranker-model',
          }),
          PromptsSettingsTab: createPromptsTabStub({ qa_response: '回答提示词 draft' }),
          ImageGenSettingsTab: createSettingsTabStub({
            provider: 'newapi',
            apiKey: 'image-key',
            model: 'image-model',
          }),
        },
      },
    })

    await flushPromises()

    for (const label of ['LLM 对话', '批量分析', '向量模型', '重排序', '生图模型', '提示词']) {
      const tab = wrapper.findAll('[role="tab"]').find(item => item.text().includes(label))
      expect(tab).toBeTruthy()
      await tab!.trigger('click')
      await flushPromises()
    }

    const saveButton = wrapper.findAll('button').find(button => button.text() === '保存')
    expect(saveButton).toBeTruthy()

    await saveButton!.trigger('click')
    await flushPromises()

    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          vlm: expect.objectContaining({
            provider: 'custom-vlm',
            apiKey: 'vlm-key',
            model: 'vlm-model',
          }),
          llm: expect.objectContaining({
            provider: 'custom-llm',
            apiKey: 'llm-key',
            model: 'llm-model',
          }),
          embedding: expect.objectContaining({
            provider: 'custom-embedding',
            apiKey: 'embedding-key',
            model: 'embedding-model',
          }),
          reranker: expect.objectContaining({
            provider: 'custom-reranker',
            apiKey: 'reranker-key',
            model: 'reranker-model',
          }),
          imageGen: expect.objectContaining({
            provider: 'newapi',
            apiKey: 'image-key',
            model: 'image-model',
          }),
          batch: expect.objectContaining({ pagesPerBatch: 8, contextBatchCount: 4 }),
          prompts: expect.objectContaining({ qa_response: '回答提示词 draft' }),
        }),
      })
    )
  })

  it('restores the authoritative config and provider cache when saving fails', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    apiMocks.saveGlobalConfig.mockRejectedValueOnce(new Error('write conflict'))

    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [pinia],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub({
            provider: 'unsaved-vlm',
            apiKey: 'unsaved-key',
            model: 'unsaved-model',
          }),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()
    const authoritativeState = insightStore.snapshotConfigState()

    const saveButton = wrapper.findAll('button').find(button => button.text() === '保存')
    expect(saveButton).toBeTruthy()
    await saveButton!.trigger('click')
    await flushPromises()

    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          vlm: expect.objectContaining({ provider: 'unsaved-vlm' }),
        }),
      })
    )
    expect(insightStore.snapshotConfigState()).toEqual(authoritativeState)
    expect(wrapper.text()).toContain('保存失败')
  })

  it('rolls back config and provider-cache mutations when the user cancels', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()

    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [pinia],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()
    const authoritativeState = insightStore.snapshotConfigState()
    insightStore.updateVlmConfig({
      provider: 'cancelled-provider',
      apiKey: 'cancelled-key',
      model: 'cancelled-model',
    })
    expect(insightStore.snapshotConfigState()).not.toEqual(authoritativeState)

    const cancelButton = wrapper.findAll('button').find(button => button.text() === '取消')
    expect(cancelButton).toBeTruthy()
    await cancelButton!.trigger('click')

    expect(insightStore.snapshotConfigState()).toEqual(authoritativeState)
    expect(wrapper.emitted('close')).toHaveLength(1)
  })

  it('rolls back config and provider-cache mutations when its owner unmounts', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()

    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [pinia],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()
    const authoritativeState = insightStore.snapshotConfigState()
    insightStore.updateVlmConfig({
      provider: 'unmounted-provider',
      apiKey: 'unmounted-key',
      model: 'unmounted-model',
    })

    wrapper.unmount()

    expect(insightStore.snapshotConfigState()).toEqual(authoritativeState)
  })

  it('ignores an initial config response after the modal owner unmounts', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    const authoritativeState = insightStore.snapshotConfigState()
    const pendingConfig = deferred<Record<string, unknown>>()
    apiMocks.getGlobalConfig.mockReturnValueOnce(pendingConfig.promise)
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [pinia],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    wrapper.unmount()
    pendingConfig.resolve({
      vlm: { provider: 'late-provider', model: 'late-model' },
    })
    await flushPromises()

    expect(insightStore.snapshotConfigState()).toEqual(authoritativeState)
  })

  it('blocks close and duplicate save while the backend save is pending', async () => {
    const pendingSave = deferred<void>()
    apiMocks.saveGlobalConfig.mockReturnValueOnce(pendingSave.promise)
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createPromptsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })
    await flushPromises()
    const saveButton = wrapper.findAll('button').find(button => button.text() === '保存')
    if (!saveButton) throw new Error('Missing settings save button')

    await saveButton.trigger('click')
    wrapper.getComponent(baseModalStub).vm.$emit('close')
    await saveButton.trigger('click')

    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('close')).toBeUndefined()

    wrapper.unmount()
    pendingSave.resolve()
    await flushPromises()
    expect(apiMocks.getGlobalConfig).toHaveBeenCalledTimes(1)
  })

  it('does not wire settings tabs through exposed child instance methods', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/InsightSettingsModal.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/\b(vlm|llm|batch|embedding|reranker|prompts|imageGen)TabRef\b/)
    expect(source).not.toMatch(/\.(getConfig|getCustomPrompts|syncFromStore)\(/)
    expect(source).toContain('@update:config')
    expect(source).toContain('@update:prompts')
  })
})
