import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { useSettingsStore } from '@/stores/settings'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductLogPanel from '@/components/product/ProductLogPanel.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const {
  getPluginAgentSettingsMock,
  getPluginAgentSessionMock,
  createPluginAgentSessionMock,
  deletePluginAgentSessionMock,
  sendPluginAgentMessageMock,
  lockPluginAgentTargetMock,
  startPluginAgentExecutionMock,
  subscribePluginAgentEventsMock,
  fetchModelsMock,
  testAiTranslateConnectionMock,
} = vi.hoisted(() => ({
  getPluginAgentSettingsMock: vi.fn(),
  getPluginAgentSessionMock: vi.fn(),
  createPluginAgentSessionMock: vi.fn(),
  deletePluginAgentSessionMock: vi.fn(),
  sendPluginAgentMessageMock: vi.fn(),
  lockPluginAgentTargetMock: vi.fn(),
  startPluginAgentExecutionMock: vi.fn(),
  subscribePluginAgentEventsMock: vi.fn(),
  fetchModelsMock: vi.fn(),
  testAiTranslateConnectionMock: vi.fn(),
}))

vi.mock('@/api/pluginAgent', () => ({
  getPluginAgentSettings: getPluginAgentSettingsMock,
  getPluginAgentSession: getPluginAgentSessionMock,
  createPluginAgentSession: createPluginAgentSessionMock,
  deletePluginAgentSession: deletePluginAgentSessionMock,
  sendPluginAgentMessage: sendPluginAgentMessageMock,
  lockPluginAgentTarget: lockPluginAgentTargetMock,
  startPluginAgentExecution: startPluginAgentExecutionMock,
  subscribePluginAgentEvents: subscribePluginAgentEventsMock,
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: Boolean,
        default: false,
      },
    },
    emits: ['update:modelValue', 'close', 'open'],
    setup(_props, { slots }) {
      return () => h('div', [
        h('div', { class: 'modal-body-stub' }, slots.default ? slots.default() : []),
        h('div', { class: 'modal-footer-stub' }, slots.footer ? slots.footer() : []),
      ])
    },
  }),
}))

vi.mock('@/components/ui/UiCombobox.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: [String, Number],
        default: '',
      },
      options: {
        type: Array,
        default: () => [],
      },
    },
    emits: ['change'],
    setup(props, { emit }) {
      return () => h(
        'select',
        {
          class: 'ui-combobox-stub',
          value: props.modelValue,
          onChange: (event: Event) => emit('change', (event.target as HTMLSelectElement).value),
        },
        (props.options || []).map((option: { label: string; value: string }) =>
          h('option', { value: option.value }, option.label),
        ),
      )
    },
  }),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  }),
}))

vi.mock('@/api/v2/diagnostics', () => ({
  fetchModels: fetchModelsMock,
  testAiTranslateConnection: testAiTranslateConnectionMock,
}))

import PluginAgentModal from '@/components/settings/PluginAgentModal.vue'

function getButtonByText(wrapper: ReturnType<typeof mount>, text: string) {
  const button = wrapper.findAll('button').find(candidate => candidate.text().includes(text))
  expect(button, `Expected button containing text "${text}"`).toBeTruthy()
  return button!
}

function createDeferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

describe('PluginAgentModal', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    setActivePinia(createPinia())
    getPluginAgentSettingsMock.mockReset()
    getPluginAgentSessionMock.mockReset()
    createPluginAgentSessionMock.mockReset()
    deletePluginAgentSessionMock.mockReset()
    sendPluginAgentMessageMock.mockReset()
    lockPluginAgentTargetMock.mockReset()
    startPluginAgentExecutionMock.mockReset()
    subscribePluginAgentEventsMock.mockReset()
    fetchModelsMock.mockReset()
    testAiTranslateConnectionMock.mockReset()

    getPluginAgentSettingsMock.mockResolvedValue({
      success: true,
      overview: ['插件只能操作单个目录'],
      overview_sections: [
        {
          title: '基础规则',
          items: ['插件只能操作单个目录'],
        },
        {
          title: '翻译与渲染类 Hook',
          items: ['`before_translate` / `after_translate`：普通翻译前 / 普通翻译后。'],
        },
      ],
      prompt_examples: ['做一个 OCR 插件'],
      providers: [
        { value: 'siliconflow', label: 'SiliconFlow' },
        { value: 'deepseek', label: 'DeepSeek' },
      ],
      plugins: [
        {
          id: 'existing_plugin',
          display_name: 'Existing Plugin',
          description: 'demo',
          version: '1.0.0',
          enabled: false,
          default_enabled: false,
          has_config: false,
          supported_steps: ['ocr'],
          supported_modes: ['standard'],
        },
      ],
    })

    createPluginAgentSessionMock.mockResolvedValue({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'drafting',
        messages: [],
        events: [],
        touched_files: [],
        file_previews: {},
      },
    })
    getPluginAgentSessionMock.mockResolvedValue({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'drafting',
        messages: [
          { id: 'user-1', role: 'user', content: '做一个 OCR 插件', timestamp: '2026-01-01T00:00:00Z' },
        ],
        events: [],
        touched_files: [],
        file_previews: {},
      },
    })

    sendPluginAgentMessageMock.mockResolvedValue({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'awaiting_target_lock',
        pending_target: {
          plugin_id: 'auto_plugin',
          display_name: 'Auto Plugin',
          supported_steps: ['ocr'],
          supported_modes: ['standard'],
        },
        messages: [
          { id: 'user-1', role: 'user', content: '做一个 OCR 插件', timestamp: '2026-01-01T00:00:00Z' },
          { id: 'assistant-1', role: 'assistant', content: '建议创建新插件。', timestamp: '2026-01-01T00:00:01Z' },
        ],
        events: [
          {
            id: 1,
            type: 'state',
            payload: {
              run_state: 'awaiting_target_lock',
              label: '等待锁定',
              message: 'Agent 已提出插件方案，等待你锁定目标插件。',
            },
            timestamp: '2026-01-01T00:00:01Z',
          },
        ],
        touched_files: [],
        file_previews: {},
      },
    })

    lockPluginAgentTargetMock.mockResolvedValue({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'ready',
        pending_target: null,
        locked_target: {
          plugin_id: 'auto_plugin',
          display_name: 'Auto Plugin',
          plugin_dir: 'C:/plugins/auto_plugin',
          supported_steps: ['ocr'],
          supported_modes: ['standard'],
        },
        messages: [
          { id: 'user-1', role: 'user', content: '做一个 OCR 插件', timestamp: '2026-01-01T00:00:00Z' },
          { id: 'assistant-1', role: 'assistant', content: '建议创建新插件。', timestamp: '2026-01-01T00:00:01Z' },
        ],
        events: [],
        touched_files: [],
        file_previews: {},
      },
    })

    startPluginAgentExecutionMock.mockResolvedValue({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'running',
        messages: [],
        events: [
          {
            id: 2,
            type: 'state',
            payload: {
              run_state: 'running',
              label: '开始执行',
              message: 'Agent 已开始在锁定插件目录中执行。',
            },
            timestamp: '2026-01-01T00:00:02Z',
          },
        ],
        touched_files: [],
        file_previews: {},
      },
    })
    subscribePluginAgentEventsMock.mockImplementation(async (_sessionId, options) => {
      await options.onEvent({
        id: 3,
        type: 'assistant_delta',
        payload: {
          stream_id: 'exec-1',
          phase: 'execution',
          delta: '正在编写插件骨架',
          content: '正在编写插件骨架',
        },
        timestamp: '2026-01-01T00:00:03Z',
      })
      await options.onEvent({
        id: 4,
        type: 'assistant',
        payload: {
          stream_id: 'exec-1',
          phase: 'execution',
          message: '正在编写插件骨架',
        },
        timestamp: '2026-01-01T00:00:04Z',
      })
      await options.onEvent({
        id: 5,
        type: 'tool_call',
        payload: {
          group_id: 'tool-1',
          tool: 'write_file',
          summary: '写入插件入口文件 __init__.py',
          args_preview: {
            path: '__init__.py',
          },
        },
        timestamp: '2026-01-01T00:00:05Z',
      })
      await options.onEvent({
        id: 6,
        type: 'tool_result',
        payload: {
          group_id: 'tool-1',
          tool: 'write_file',
          summary: '已写入 __init__.py',
          success: true,
          changed_files: ['__init__.py'],
          file_previews: {
            '__init__.py': 'from .plugin import AutoPlugin',
          },
          debug_result: {
            success: true,
            path: '__init__.py',
          },
        },
        timestamp: '2026-01-01T00:00:06Z',
      })
      await options.onEvent({
        id: 7,
        type: 'validation',
        payload: {
          summary: '插件校验通过',
          success: true,
          details: {
            success: true,
          },
        },
        timestamp: '2026-01-01T00:00:07Z',
      })
      await options.onEvent({
        id: 8,
        type: 'done',
        payload: {
          message: '插件开发已完成',
          validation: {
            success: true,
          },
          refresh_result: {
            success: true,
          },
          run_state: 'completed',
        },
        timestamp: '2026-01-01T00:00:08Z',
      })
    })

    fetchModelsMock.mockResolvedValue({
      success: true,
      models: [
        { id: 'glm-4.5', name: 'GLM-4.5' },
        { id: 'glm-5.1', name: 'GLM-5.1' },
      ],
    })
    testAiTranslateConnectionMock.mockResolvedValue({ success: true, message: '连接成功' })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('requires selecting an existing plugin before starting a modify session', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await getButtonByText(wrapper, '修改现有插件').trigger('click')
    await flushPromises()

    const beginButton = wrapper.find('.plugin-agent-submit-message-action')
    expect(beginButton.attributes('disabled')).toBeDefined()

    await wrapper.find('.ui-combobox-stub').setValue('existing_plugin')
    await flushPromises()

    expect(beginButton.attributes('disabled')).toBeDefined()

    await wrapper.find('.plugin-agent-input').setValue('修改这个插件')
    await flushPromises()

    expect(beginButton.attributes('disabled')).toBeUndefined()
  })

  it('renders task mode selection through shared product segmented tabs', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const tabs = wrapper.getComponent(ProductSegmentedTabs)

    expect(tabs.props('ariaLabel')).toBe('插件 Agent 任务模式')
    expect(tabs.props('activeTab')).toBe('create')
    expect(tabs.props('tabs')).toEqual([
      { id: 'create', label: '新建插件' },
      { id: 'modify', label: '修改现有插件' },
    ])

    tabs.vm.$emit('select', 'modify')
    await flushPromises()

    expect(wrapper.text()).toContain('目标插件')
  })

  it('keeps scoped modal colors on semantic tokens instead of raw owner palettes', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--color-action-brand')
    expect(styleBlock).toContain('--color-status-success')
  })

  it('lets UiButton own disabled button styling instead of local modal skins', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')

    expect(source).not.toContain('--plugin-agent-disabled-border')
    expect(source).not.toContain('--plugin-agent-disabled-background')
    expect(source).not.toContain('--plugin-agent-disabled-text')
    expect(source).not.toMatch(/\.plugin-agent-(start|begin)-btn:disabled/)
  })

  it('does not keep stale button-era hooks for execution cancellation', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')

    expect(source).toContain('plugin-agent-cancel-action')
    expect(source).not.toContain('plugin-agent-cancel-btn')
  })

  it('uses owner action hooks for plugin agent commands', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')

    expect(source).toContain('plugin-agent-save-settings-action')
    expect(source).toContain('plugin-agent-clear-session-action')
    expect(source).toContain('plugin-agent-lock-target-action')
    expect(source).toContain('plugin-agent-start-execution-action')
    expect(source).toContain('plugin-agent-submit-message-action')
    expect(source).not.toMatch(/plugin-agent-(save-settings|clear|lock|start|begin)-btn/)
  })

  it('uses owner modifiers for plugin agent timeline card state', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')

    expect(source).toContain('plugin-agent-step-card--${item.kind}')
    expect(source).toContain('plugin-agent-step-card--status-${item.status}')
    expect(source).toContain('plugin-agent-step-card--streaming')
    expect(source).toContain('.plugin-agent-step-card--status-streaming::before')
    expect(source).toContain('.plugin-agent-step-card--status-success::before')
    expect(source).toContain('.plugin-agent-step-card--status-error::before')
    expect(source).toContain('.plugin-agent-step-card--status-waiting::before')
    expect(source).toContain('.plugin-agent-step-card--assistant.plugin-agent-step-card--streaming')

    expect(source).not.toContain('plugin-agent-step-card-${item.kind}')
    expect(source).not.toContain('`status-${item.status}`')
    expect(source).not.toMatch(/\{\s*streaming:\s*item\.status === 'streaming'\s*\}/)
    expect(source).not.toContain('.plugin-agent-step-card.status-streaming')
    expect(source).not.toContain('.plugin-agent-step-card.status-success')
    expect(source).not.toContain('.plugin-agent-step-card.status-error')
    expect(source).not.toContain('.plugin-agent-step-card.status-waiting')
    expect(source).not.toContain('.plugin-agent-step-card-assistant.streaming')
  })

  it('keeps plugin agent modal presentation hooks explicit outside markdown content', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    for (const requiredClass of [
      'plugin-agent-modal__block-title',
      'plugin-agent-overview-section__title',
      'plugin-agent-overview-section__list',
      'plugin-agent-overview-section__list-item',
      'plugin-agent-modal__examples-title',
      'plugin-agent-meta-row__label',
      'plugin-agent-meta-row__value',
      'plugin-agent-modal__pending-target-title',
      'plugin-agent-modal__validation-title',
      'plugin-agent-modal__validation-payload',
      'plugin-agent-step-details__summary',
    ]) {
      expect(classTokens).toContain(requiredClass)
    }

    for (const forbiddenSelector of [
      '.plugin-agent-overview-section h4',
      '.plugin-agent-overview-section .plugin-agent-list',
      '.plugin-agent-overview-section .plugin-agent-list li',
      '.plugin-agent-step-details summary',
      '.plugin-agent-validation pre',
    ]) {
      expect(source).not.toContain(forbiddenSelector)
    }

    expect(source).toContain('.plugin-agent-overview-item p')
    expect(source).toContain('.plugin-agent-step-content p')
    expect(source).not.toMatch(/<h3>(?:任务模式|Agent 设置|插件开发提示|对话与过程|输入|本轮任务工件|触达文件)<\/h3>/)
    expect(source).not.toMatch(/<h4>(?:示例描述|待锁定目标|最后校验)<\/h4>/)
  })

  it('uses the fixed select primitive for the agent provider', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const providerSelect = wrapper.getComponent(UiSelect)
    expect(providerSelect.props('modelValue')).toBe('siliconflow')
    expect(providerSelect.props('options')).toEqual([
      { value: 'siliconflow', label: 'SiliconFlow' },
      { value: 'deepseek', label: 'DeepSeek' },
    ])

    providerSelect.vm.$emit('change', 'deepseek')
    await flushPromises()

    const store = useSettingsStore()
    expect(store.settings.pluginAgent.provider).toBe('deepseek')
  })

  it('uses the shared credential primitive for the agent API key', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const passwordField = wrapper.getComponent(UiPasswordField)
    expect(passwordField.props('placeholder')).toBe('请输入 API Key')
    expect(passwordField.props('showLabel')).toBe('显示插件 Agent API Key')
    expect(passwordField.props('hideLabel')).toBe('隐藏插件 Agent API Key')
  })

  it('uses shared field and action primitives for agent settings controls', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const fieldLabels = wrapper.findAllComponents(UiField).map(field => field.props('label'))
    expect(fieldLabels).toEqual(expect.arrayContaining([
      '服务商',
      'API Key',
      'Base URL',
      '模型名称',
      'RPM',
      '业务重试',
      '传输重试',
      '输出选项',
    ]))

    const settingsActions = wrapper.findAllComponents(ProductActionRow)
      .find(row => row.props('ariaLabel') === 'Agent 设置操作')
    expect(settingsActions?.props('justify')).toBe('start')
  })

  it('binds agent settings field labels to stable primitive ids', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const fields = wrapper.findAllComponents(UiField)
    const controlIds = fields
      .map(field => field.props('controlId'))
      .filter(Boolean)

    expect(controlIds).toEqual(expect.arrayContaining([
      'pluginAgentProvider',
      'pluginAgentApiKey',
      'pluginAgentBaseUrl',
      'pluginAgentModelName',
      'pluginAgentRpmLimit',
      'pluginAgentBusinessRetries',
      'pluginAgentTransportRetries',
    ]))

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')
    expect(source).toContain('id="pluginAgentProvider"')
    expect(source).toContain('input-id="pluginAgentApiKey"')
    expect(source).toContain('id="pluginAgentBaseUrl"')
    expect(source).toContain('input-id="pluginAgentModelName"')
    expect(source).toContain('input-id="pluginAgentRpmLimit"')
    expect(source).toContain('input-id="pluginAgentBusinessRetries"')
    expect(source).toContain('input-id="pluginAgentTransportRetries"')
  })

  it('uses the shared number primitive for agent numeric settings', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const numberFields = wrapper.findAllComponents(UiNumberField)
    expect(numberFields).toHaveLength(3)
    expect(numberFields.map(field => field.props('min'))).toEqual([0, 0, 0])
    expect(numberFields.map(field => field.props('max'))).toEqual([undefined, 10, 10])
    expect(numberFields.map(field => field.props('step'))).toEqual([1, 1, 1])

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')
    expect(source).not.toMatch(/UiInput[^>]+type="number"|type="number"[^>]+UiInput/)
  })

  it('renders conversation history through shared scroll and message primitives', async () => {
    let resolveSend: ((value: unknown) => void) | null = null
    sendPluginAgentMessageMock.mockImplementation(
      () => new Promise((resolve) => { resolveSend = resolve }),
    )

    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-submit-message-action').trigger('click')
    await flushPromises()

    const historyStack = wrapper.getComponent(ProductScrollStack)
    expect(historyStack.props('role')).toBe('log')
    expect(historyStack.props('ariaLabel')).toBe('插件 Agent 对话和过程')

    const bubbles = wrapper.findAllComponents(ProductMessageBubble)
    expect(bubbles.map(bubble => bubble.props('role'))).toEqual(['user', 'assistant'])
    expect(bubbles.map(bubble => bubble.props('avatarLabel'))).toEqual(['你', 'Agent'])
    expect(wrapper.find('.plugin-agent-message').exists()).toBe(false)
    expect(wrapper.find('.plugin-agent-message-loading').exists()).toBe(true)

    resolveSend?.({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'awaiting_target_lock',
        pending_target: {
          plugin_id: 'auto_plugin',
          display_name: 'Auto Plugin',
          supported_steps: ['ocr'],
          supported_modes: ['standard'],
        },
        messages: [
          { id: 'user-1', role: 'user', content: '做一个 OCR 插件', timestamp: '2026-01-01T00:00:00Z' },
          { id: 'assistant-1', role: 'assistant', content: '建议创建新插件。', timestamp: '2026-01-01T00:00:01Z' },
        ],
        events: [],
        touched_files: [],
        file_previews: {},
      },
    })
    await flushPromises()

    expect(wrapper.findAllComponents(ProductMessageBubble)).toHaveLength(2)
    expect(wrapper.text()).toContain('建议创建新插件。')
  })

  it('uses product status feedback for empty conversation and touched files', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')
    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('plugin-agent-empty')

    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const statusBanners = wrapper.findAllComponents(ProductStatusBanner)
    expect(statusBanners.map(banner => banner.props('title'))).toEqual(expect.arrayContaining([
      '插件 Agent',
      '暂无文件变更',
    ]))
    expect(statusBanners.map(banner => banner.props('tone'))).toEqual(expect.arrayContaining([
      'neutral',
    ]))
    expect(statusBanners.map(banner => banner.props('role'))).toEqual(expect.arrayContaining([
      'note',
    ]))
    expect(wrapper.text()).toContain('描述你想创建或修改的插件需求')
    expect(wrapper.text()).toContain('执行后会在这里显示本轮写入或修改的文件')
    expect(wrapper.find('.plugin-agent-empty').exists()).toBe(false)
  })

  it('keeps execution disabled until a create target has been locked', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-submit-message-action').trigger('click')
    await flushPromises()

    const startButton = wrapper.find('.plugin-agent-start-execution-action')
    expect(startButton.attributes('disabled')).toBeDefined()

    const lockButton = wrapper.find('.plugin-agent-lock-target-action')
    expect(lockButton.exists()).toBe(true)

    await lockButton.trigger('click')
    await flushPromises()

    expect(wrapper.find('.plugin-agent-start-execution-action').attributes('disabled')).toBeUndefined()
  })

  it('shows the user message immediately and renders a waiting animation while the agent is replying', async () => {
    let resolveSend: ((value: unknown) => void) | null = null
    sendPluginAgentMessageMock.mockImplementation(
      () => new Promise((resolve) => { resolveSend = resolve }),
    )

    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-submit-message-action').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('做一个 OCR 插件')
    expect(wrapper.text()).toContain('Agent 正在分析需求')
    expect(wrapper.find('.plugin-agent-message-loading').exists()).toBe(true)
    expect(wrapper.find('.plugin-agent-submit-message-action').text()).toContain('等待回复...')

    resolveSend?.({
      success: true,
      session: {
        session_id: 'session-1',
        mode: 'create',
        run_state: 'awaiting_target_lock',
        pending_target: {
          plugin_id: 'auto_plugin',
          display_name: 'Auto Plugin',
          supported_steps: ['ocr'],
          supported_modes: ['standard'],
        },
        messages: [
          { id: 'user-1', role: 'user', content: '做一个 OCR 插件', timestamp: '2026-01-01T00:00:00Z' },
          { id: 'assistant-1', role: 'assistant', content: '建议创建新插件。', timestamp: '2026-01-01T00:00:01Z' },
        ],
        events: [],
        touched_files: [],
        file_previews: {},
      },
    })
    await flushPromises()

    expect(wrapper.find('.plugin-agent-message-loading').exists()).toBe(false)
    expect(wrapper.text()).toContain('建议创建新插件。')
  })

  it('renders friendly timeline cards instead of raw json events after conversation starts', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-submit-message-action').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('等待锁定')
    expect(wrapper.text()).toContain('Agent 已提出插件方案，等待你锁定目标插件。')
    expect(wrapper.text()).not.toContain('awaiting_target_lock')
    expect(wrapper.text()).not.toContain('"run_state"')
    expect(wrapper.find('.plugin-agent-step-card').exists()).toBe(true)
  })

  it('splits history and composer into separate panels and saves only agent settings', async () => {
    const store = useSettingsStore()
    const saveSpy = vi.spyOn(store, 'savePluginAgentSettings').mockResolvedValue(true)

    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    expect(wrapper.find('.plugin-agent-history-panel').exists()).toBe(true)
    expect(wrapper.find('.plugin-agent-composer-panel').exists()).toBe(true)
    expect(wrapper.find('.plugin-agent-scroll-column').exists()).toBe(true)
    expect(wrapper.get('textarea.plugin-agent-input').getComponent(UiTextarea).props()).toMatchObject({
      variant: 'panel',
      rows: 4,
    })

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')
    expect(source).toContain('variant="panel"')
    expect(source).not.toMatch(/--ui-(?:input|textarea)-/)

    const saveButton = wrapper.find('.plugin-agent-save-settings-action')
    expect(saveButton.exists()).toBe(true)

    await saveButton.trigger('click')
    await flushPromises()

    expect(saveSpy).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('基础规则')
    expect(wrapper.text()).toContain('翻译与渲染类 Hook')
  })

  it('collapses the agent workbench from the modal content container', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginAgentModal.vue'), 'utf8')

    expect(source).toContain('class="plugin-agent-layout-shell"')
    expect(source).toContain('container: plugin-agent-modal / inline-size')
    expect(source).toContain('@container plugin-agent-modal')
    expect(source).not.toContain('@media (--breakpoint-modal-wide-down)')
  })

  it('does not assert shared textarea primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/pluginAgentModal.spec.ts'), 'utf8')
    const textareaClassPrefix = 'ui-' + 'textarea--'

    expect(source).not.toContain(textareaClassPrefix)
  })

  it('renders prompt examples through the shared chip primitive', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const exampleChips = wrapper.findAllComponents(ProductChipList)
      .find(chips => chips.props('ariaLabel') === '插件 Agent 示例描述')
    expect(exampleChips).toBeTruthy()
    expect(exampleChips?.props('items')).toEqual([
      {
        id: '做一个 OCR 插件',
        label: '做一个 OCR 插件',
        interactive: true,
        tone: 'neutral',
      },
    ])

    exampleChips?.vm.$emit('select', '做一个 OCR 插件')
    await flushPromises()

    const input = wrapper.find('.plugin-agent-input')
    expect((input.element as HTMLTextAreaElement).value).toBe('做一个 OCR 插件')
    expect(wrapper.find('.plugin-agent-example').exists()).toBe(false)
  })

  it('sanitizes plugin agent markdown before rendering it as html', async () => {
    getPluginAgentSettingsMock.mockResolvedValueOnce({
      success: true,
      overview: [],
      overview_sections: [
        {
          title: '安全提示',
          items: [
            '<img src=x onerror="alert(1)">[危险链接](javascript:alert(2))',
          ],
        },
      ],
      prompt_examples: [],
      providers: [],
      plugins: [],
    })

    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const html = wrapper.html()
    expect(html).not.toContain('onerror')
    expect(html).not.toContain('javascript:')
  })

  it('shows a fetched model selector and lets the user choose a fetched model', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    const modelPicker = wrapper.getComponent(UiModelPicker)
    expect(modelPicker.props('modelValue')).toBe('')
    expect(modelPicker.props('modelCount')).toBe(0)

    wrapper.getComponent(UiPasswordField).vm.$emit('update:modelValue', 'model-key')
    await flushPromises()
    modelPicker.vm.$emit('fetch')
    await flushPromises()

    expect(fetchModelsMock).toHaveBeenCalledWith('siliconflow', 'model-key', '', 'plugin_agent')
    const updatedModelPicker = wrapper.getComponent(UiModelPicker)
    expect(updatedModelPicker.props('modelCount')).toBe(2)
    expect(updatedModelPicker.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ label: 'GLM-4.5', value: 'glm-4.5' }),
      expect.objectContaining({ label: 'GLM-5.1', value: 'glm-5.1' }),
    ]))
    expect(wrapper.text()).toContain('共 2 个模型')

    updatedModelPicker.vm.$emit('change', 'glm-5.1')
    await flushPromises()

    const modelInput = wrapper.find('input[placeholder="请输入模型名称"]')
    expect((modelInput.element as HTMLInputElement).value).toBe('glm-5.1')
  })

  it('ignores stale fetched model responses after the agent provider changes', async () => {
    const pendingModels = createDeferred<{ success: boolean; models: Array<{ id: string; name: string }> }>()
    fetchModelsMock.mockReset()
    fetchModelsMock.mockReturnValueOnce(pendingModels.promise)

    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    wrapper.getComponent(UiPasswordField).vm.$emit('update:modelValue', 'model-key')
    await flushPromises()
    wrapper.getComponent(UiModelPicker).vm.$emit('fetch')
    expect(fetchModelsMock).toHaveBeenCalledWith('siliconflow', 'model-key', '', 'plugin_agent')

    const providerSelect = wrapper.getComponent(UiSelect)
    providerSelect.vm.$emit('change', 'deepseek')

    pendingModels.resolve({
      success: true,
      models: [{ id: 'stale-plugin-agent-model', name: 'Stale Plugin Agent Model' }],
    })
    await flushPromises()

    expect(wrapper.text()).not.toContain('Stale Plugin Agent Model')
  })

  it('streams assistant output into a single timeline card and keeps raw debug json collapsed', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-submit-message-action').trigger('click')
    await flushPromises()
    await wrapper.find('.plugin-agent-lock-target-action').trigger('click')
    await flushPromises()
    await wrapper.find('.plugin-agent-start-execution-action').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('.plugin-agent-step-card--assistant').length).toBe(1)
    expect(wrapper.text()).toContain('正在编写插件骨架')
    expect(wrapper.text()).toContain('写入插件入口文件 __init__.py')
    expect(wrapper.text()).toContain('插件校验通过')
    expect(wrapper.text()).toContain('__init__.py')
    expect(wrapper.text()).not.toContain('tool_result')
    expect(wrapper.text()).not.toContain('debug_result')

    const debugLog = wrapper.getComponent(ProductLogPanel)
    expect(debugLog.props('title')).toBe('调试事件')
    expect(debugLog.props('ariaLabel')).toBe('插件 Agent 调试事件')
    expect(debugLog.props('items')).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          message: 'tool_result',
          detail: expect.stringContaining('"group_id": "tool-1"'),
        }),
        expect.objectContaining({
          message: 'tool_result',
          detail: expect.stringContaining('"debug_result"'),
        }),
      ]),
    )
    expect(wrapper.text()).not.toContain('"group_id": "tool-1"')

    await debugLog.get('.product-log-panel__header').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('"group_id": "tool-1"')
    expect(wrapper.find('.plugin-agent-debug-toggle').exists()).toBe(false)
  })

  it('renders touched files through the shared record-card primitive', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-submit-message-action').trigger('click')
    await flushPromises()
    await wrapper.find('.plugin-agent-lock-target-action').trigger('click')
    await flushPromises()
    await wrapper.find('.plugin-agent-start-execution-action').trigger('click')
    await flushPromises()

    const fileCards = wrapper.findAllComponents(ProductRecordCard)
      .filter(card => card.props('ariaLabel')?.startsWith('触达文件：'))
    expect(fileCards).toHaveLength(1)
    expect(fileCards[0].props('ariaLabel')).toBe('触达文件：__init__.py')
    expect(fileCards[0].text()).toContain('from .plugin import AutoPlugin')
    expect(wrapper.find('.plugin-agent-file-card').exists()).toBe(false)
  })
})
