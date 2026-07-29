import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import ParallelSettings from '@/components/settings/ParallelSettings.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import { useSettingsStore } from '@/stores/settings'

describe('ParallelSettings', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('uses the shared switch primitive for enabling parallel mode', () => {
    const store = useSettingsStore()
    store.settings.parallel.enabled = true

    const wrapper = mount(ParallelSettings)
    const switchControl = wrapper.getComponent(UiSwitch)

    expect(switchControl.props('accessibilityLabel')).toBe('启用并行模式')
    expect(switchControl.props('modelValue')).toBe(true)

    switchControl.vm.$emit('change', false)

    expect(store.settings.parallel.enabled).toBe(false)
  })

  it('uses the shared number field for deep learning concurrency', () => {
    const store = useSettingsStore()
    store.settings.parallel.enabled = true
    store.settings.parallel.deepLearningLockSize = 2

    const wrapper = mount(ParallelSettings)
    const numberField = wrapper.getComponent(UiNumberField)

    expect(numberField.props('modelValue')).toBe(2)
    expect(numberField.props('min')).toBe(1)
    expect(numberField.props('max')).toBe(4)
    expect(numberField.props('controls')).toBe(true)

    numberField.vm.$emit('update:modelValue', 5)
    expect(store.settings.parallel.deepLearningLockSize).toBe(4)

    expect(wrapper.find('.number-input').exists()).toBe(false)
    expect(wrapper.find('.number-control').exists()).toBe(false)
  })

  it('maps warning note colors to semantic status tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/ParallelSettings.vue'), 'utf8')

    expect(source).not.toContain('rgba(255, 193, 7')
    expect(source).not.toContain('#ffc107')
    expect(source).toContain('--color-status-warning')
  })

  it('keeps the warning note under the ParallelSettings owner hooks', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/ParallelSettings.vue'), 'utf8')
    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))

    expect(classTokens).toContain('parallel-settings__note')
    expect(classTokens).toContain('parallel-settings__note-title')
    expect(classTokens).toContain('parallel-settings__note-list')
    expect(classTokens).toContain('parallel-settings__note-item')
    expect(source).not.toContain('class="settings-note"')
    expect(source).not.toContain('class="note-title"')
    expect(source).not.toContain('.settings-note ul')
    expect(source).not.toContain('.settings-note li')
  })

  it('routes settings row labels and hints through typed UiField props', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/ParallelSettings.vue'), 'utf8')

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')
    expect(source).toContain('label="启用并行模式"')
    expect(source).toContain('label="深度学习并发数"')

    const wrapper = mount(ParallelSettings)
    const fields = wrapper.findAllComponents(UiField)
    const fieldByLabel = (label: string) => fields.find(field => field.props('label') === label)

    expect(fieldByLabel('启用并行模式')?.props('hint')).toContain('流水线并行处理')
    expect(fieldByLabel('深度学习并发数')?.props('controlId')).toBe('parallelDeepLearningLockSize')
    expect(fieldByLabel('深度学习并发数')?.props('hint')).toContain('建议1-2')
  })
})
