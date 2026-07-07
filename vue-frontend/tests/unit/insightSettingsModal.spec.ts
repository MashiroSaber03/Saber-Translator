import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, onMounted, watch } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import InsightSettingsModal from '@/components/insight/InsightSettingsModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'

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
  template: '<button type="button" class="emit-message" @click="$emit(\'showMessage\', \'连接成功\', \'success\')">emit</button>',
})

describe('InsightSettingsModal', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setActivePinia(createPinia())
    apiMocks.getGlobalConfig.mockReset().mockResolvedValue({ success: true, config: {} })
    apiMocks.saveGlobalConfig.mockReset().mockResolvedValue({ success: true })
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
          VlmSettingsTab: createSettingsTabStub({ provider: 'custom-vlm', apiKey: 'vlm-key', model: 'vlm-model' }),
          LlmSettingsTab: createSettingsTabStub({ provider: 'custom-llm', apiKey: 'llm-key', model: 'llm-model' }),
          BatchSettingsTab: createSettingsTabStub({ pagesPerBatch: 8, contextBatchCount: 4 }),
          EmbeddingSettingsTab: createSettingsTabStub({ provider: 'custom-embedding', apiKey: 'embedding-key', model: 'embedding-model' }),
          RerankerSettingsTab: createSettingsTabStub({ provider: 'custom-reranker', apiKey: 'reranker-key', model: 'reranker-model' }),
          PromptsSettingsTab: createPromptsTabStub({ qa_response: '回答提示词 draft' }),
          ImageGenSettingsTab: createSettingsTabStub({ provider: 'newapi', apiKey: 'image-key', model: 'image-model' }),
        },
      },
    })

    await flushPromises()

    const saveButton = wrapper.findAll('button').find(button => button.text() === '保存')
    expect(saveButton).toBeTruthy()

    await saveButton!.trigger('click')
    await flushPromises()

    expect(apiMocks.saveGlobalConfig).toHaveBeenCalledWith(expect.objectContaining({
      vlm: expect.objectContaining({ provider: 'custom-vlm', api_key: 'vlm-key', model: 'vlm-model' }),
      chat_llm: expect.objectContaining({ provider: 'custom-llm', api_key: 'llm-key', model: 'llm-model' }),
      embedding: expect.objectContaining({ provider: 'custom-embedding', api_key: 'embedding-key', model: 'embedding-model' }),
      reranker: expect.objectContaining({ provider: 'custom-reranker', api_key: 'reranker-key', model: 'reranker-model' }),
      image_gen: expect.objectContaining({ provider: 'newapi', api_key: 'image-key', model: 'image-model' }),
      analysis: expect.objectContaining({
        batch: expect.objectContaining({ pages_per_batch: 8, context_batch_count: 4 }),
      }),
      prompts: expect.objectContaining({ qa_response: '回答提示词 draft' }),
    }))
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
