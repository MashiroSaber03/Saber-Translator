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
} = vi.hoisted(() => ({
  getPluginAgentSettingsMock: vi.fn(),
  createPluginAgentSessionMock: vi.fn(),
  deletePluginAgentSessionMock: vi.fn(),
  sendPluginAgentMessageMock: vi.fn(),
  lockPluginAgentTargetMock: vi.fn(),
  startPluginAgentExecutionMock: vi.fn(),
}))

vi.mock('@/api/pluginAgent', () => ({
  getPluginAgentSettings: getPluginAgentSettingsMock,
  createPluginAgentSession: createPluginAgentSessionMock,
  deletePluginAgentSession: deletePluginAgentSessionMock,
  sendPluginAgentMessage: sendPluginAgentMessageMock,
  lockPluginAgentTarget: lockPluginAgentTargetMock,
  startPluginAgentExecution: startPluginAgentExecutionMock,
  subscribePluginAgentEvents: vi.fn(),
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: ['modelValue'],
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
    props: ['modelValue', 'options'],
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

    getPluginAgentSettingsMock.mockResolvedValue({
      success: true,
      overview: ['插件只能操作单个目录'],
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
        events: [],
        touched_files: [],
        file_previews: {},
      },
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

  it('renders returned session events immediately after conversation starts', async () => {
    const wrapper = mount(PluginAgentModal, {
      props: {
        modelValue: true,
      },
    })
    await flushPromises()

    await wrapper.find('.plugin-agent-input').setValue('做一个 OCR 插件')
    await wrapper.find('.plugin-agent-begin-btn').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('state')
    expect(wrapper.text()).toContain('awaiting_target_lock')
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
  })
})
