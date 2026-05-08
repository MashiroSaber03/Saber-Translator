/* eslint-disable vue/one-component-per-file */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { useSettingsStore } from '@/stores/settingsStore'

const {
  getPluginAgentSettingsMock,
  createPluginAgentSessionMock,
  deletePluginAgentSessionMock,
  sendPluginAgentMessageMock,
  lockPluginAgentTargetMock,
  startPluginAgentExecutionMock,
  subscribePluginAgentEventsMock,
} = vi.hoisted(() => ({
  getPluginAgentSettingsMock: vi.fn(),
  createPluginAgentSessionMock: vi.fn(),
  deletePluginAgentSessionMock: vi.fn(),
  sendPluginAgentMessageMock: vi.fn(),
  lockPluginAgentTargetMock: vi.fn(),
  startPluginAgentExecutionMock: vi.fn(),
  subscribePluginAgentEventsMock: vi.fn(),
}))

vi.mock('@/api/pluginAgent', () => ({
  getPluginAgentSettings: getPluginAgentSettingsMock,
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

vi.mock('@/components/common/CustomSelect.vue', () => ({
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
          class: 'custom-select-stub',
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

import PluginAgentModal from '@/components/settings/PluginAgentModal.vue'

describe('PluginAgentModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    getPluginAgentSettingsMock.mockReset()
    createPluginAgentSessionMock.mockReset()
    deletePluginAgentSessionMock.mockReset()
    sendPluginAgentMessageMock.mockReset()
    lockPluginAgentTargetMock.mockReset()
    startPluginAgentExecutionMock.mockReset()
    subscribePluginAgentEventsMock.mockReset()

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
  })

  it('requires selecting an existing plugin before starting a modify session', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-mode-modify').trigger('click')
    await flushPromises()

    const beginButton = wrapper.find('.plugin-agent-begin-btn')
    expect(beginButton.attributes('disabled')).toBeDefined()

    await wrapper.find('.custom-select-stub').setValue('existing_plugin')
    await flushPromises()

    expect(beginButton.attributes('disabled')).toBeDefined()

    await wrapper.find('.plugin-agent-input').setValue('修改这个插件')
    await flushPromises()

    expect(beginButton.attributes('disabled')).toBeUndefined()
  })

  it('keeps execution disabled until a create target has been locked', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-begin-btn').trigger('click')
    await flushPromises()

    const startButton = wrapper.find('.plugin-agent-start-btn')
    expect(startButton.attributes('disabled')).toBeDefined()

    const lockButton = wrapper.find('.plugin-agent-lock-btn')
    expect(lockButton.exists()).toBe(true)

    await lockButton.trigger('click')
    await flushPromises()

    expect(wrapper.find('.plugin-agent-start-btn').attributes('disabled')).toBeUndefined()
  })

  it('renders friendly timeline cards instead of raw json events after conversation starts', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-begin-btn').trigger('click')
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

    const saveButton = wrapper.find('.plugin-agent-save-settings-btn')
    expect(saveButton.exists()).toBe(true)

    await saveButton.trigger('click')
    await flushPromises()

    expect(saveSpy).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('基础规则')
    expect(wrapper.text()).toContain('翻译与渲染类 Hook')
  })

  it('streams assistant output into a single timeline card and keeps raw debug json collapsed', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-begin-btn').trigger('click')
    await flushPromises()
    await wrapper.find('.plugin-agent-lock-btn').trigger('click')
    await flushPromises()
    await wrapper.find('.plugin-agent-start-btn').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('.plugin-agent-step-card-assistant').length).toBe(1)
    expect(wrapper.text()).toContain('正在编写插件骨架')
    expect(wrapper.text()).toContain('写入插件入口文件 __init__.py')
    expect(wrapper.text()).toContain('插件校验通过')
    expect(wrapper.text()).toContain('__init__.py')
    expect(wrapper.text()).not.toContain('tool_result')
    expect(wrapper.text()).not.toContain('debug_result')

    const debugToggle = wrapper.find('.plugin-agent-debug-toggle')
    expect(debugToggle.exists()).toBe(true)
    expect(wrapper.text()).not.toContain('"group_id": "tool-1"')

    await debugToggle.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('"group_id": "tool-1"')
  })
})
