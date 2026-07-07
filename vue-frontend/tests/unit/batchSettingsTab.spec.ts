import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia, type Pinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import BatchSettingsTab from '@/components/insight/settings/BatchSettingsTab.vue'
import InsightSettingsPanel from '@/components/insight/settings/InsightSettingsPanel.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useInsightStore } from '@/stores/insightStore'
import type { CustomLayer } from '@/components/insight/settings/types'

function latestConfig<T>(wrapper: ReturnType<typeof mount>): T {
  const latestEvent = wrapper.emitted('update:config')?.at(-1)
  if (!latestEvent) throw new Error('Missing update:config event')
  return latestEvent[0] as T
}

describe('BatchSettingsTab', () => {
  let pinia: Pinia

  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
  })

  it('reads only the current frontend custom layer schema', () => {
    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'
    store.config.batch.customLayers = [
      {
        name: 'API-shaped layer',
        units_per_group: 12,
        align_to_chapter: true,
      } as unknown as CustomLayer,
    ]

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(latestConfig<{ customLayers: CustomLayer[] }>(wrapper).customLayers).toEqual([
      {
        name: 'API-shaped layer',
        units: 5,
        align: false,
      },
    ])
  })

  it('uses the current checkbox primitive for custom layer chapter alignment', () => {
    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(UiCheckbox).length).toBeGreaterThan(0)
  })

  it('uses the fixed select primitive for architecture presets', () => {
    const store = useInsightStore()
    store.config.batch.architecturePreset = 'standard'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const architectureSelect = wrapper.getComponent(UiSelect)
    expect(architectureSelect.props('modelValue')).toBe('standard')
    expect(architectureSelect.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'custom', label: expect.stringContaining('自定义模式') }),
    ]))
  })

  it('uses the shared settings field primitives for batch controls', () => {
    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(UiField).length).toBeGreaterThanOrEqual(3)
    expect(wrapper.findComponent(UiFormGrid).exists()).toBe(true)
    expect(wrapper.find('.insight-settings-field').exists()).toBe(false)
    expect(wrapper.find('.form-hint').exists()).toBe(false)
  })

  it('uses the shared Insight settings shell and number fields for batch controls', () => {
    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.getComponent(InsightSettingsPanel).props('description')).toContain('批量分析')

    const numberFields = wrapper.findAllComponents(UiNumberField)
    expect(numberFields.map(field => ({
      inputId: field.props('inputId'),
      min: field.props('min'),
      max: field.props('max'),
    }))).toEqual([
      { inputId: 'insight-batch-pages-per-batch', min: 1, max: 10 },
      { inputId: 'insight-batch-context-batch-count', min: 0, max: 5 },
      { inputId: 'insight-batch-layer-units-0', min: 0, max: 20 },
      { inputId: 'insight-batch-layer-units-1', min: 0, max: 20 },
      { inputId: 'insight-batch-layer-units-2', min: 0, max: 20 },
    ])

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/BatchSettingsTab.vue'),
      'utf8'
    )
    expect(source).not.toContain('type="number"')
  })

  it('uses standard button variants for custom layer actions', () => {
    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const layerButtons = wrapper.findAllComponents(UiButton)
    expect(layerButtons.some(button => button.text().includes('添加层级') && button.props('variant') === 'secondary')).toBe(true)
    expect(layerButtons.some(button => button.text().includes('删除') && button.props('variant') === 'danger')).toBe(true)
    expect(wrapper.find('.layer-delete-btn').exists()).toBe(false)
  })

  it('routes batch summaries and custom-layer actions through product primitives', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/BatchSettingsTab.vue'),
      'utf8'
    )
    expect(source).not.toContain('batch-info-box')
    expect(source).not.toContain('batch-estimate-box')
    expect(source).not.toContain('>+ 添加层级<')
    expect(source).toContain('<ProductActionRow')
    expect(source).toContain('<ProductStatusBanner')
    expect(source).toContain('name="plus"')

    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(ProductStatusBanner).map(banner => banner.props('title'))).toEqual([
      '当前架构预览',
      '当前配置',
    ])
    expect(wrapper.findAllComponents(ProductActionRow).length).toBeGreaterThan(0)
  })

  it('updates custom layer names through typed input model events', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/BatchSettingsTab.vue'),
      'utf8'
    )
    expect(source).not.toContain("($event.target as HTMLInputElement).value")
    expect(source).toContain("@update:model-value=\"updateCustomLayer(idx, 'name', $event)\"")

    const store = useInsightStore()
    store.config.batch.architecturePreset = 'custom'

    const wrapper = mount(BatchSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const layerNameInput = wrapper
      .findAllComponents(UiInput)
      .find(input => input.props('modelValue') === '段落总结')
    expect(layerNameInput).toBeDefined()

    layerNameInput?.vm.$emit('update:modelValue', '章节总结')
    await flushPromises()

    expect(latestConfig<{ customLayers: CustomLayer[] }>(wrapper).customLayers[1]).toMatchObject({
      name: '章节总结',
      units: 5,
      align: false,
    })
  })

  it('maps batch owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/BatchSettingsTab.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('keeps batch settings custom-layer hooks under the tab owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/BatchSettingsTab.vue'),
      'utf8'
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const oldHooks = [
      'custom-layers-section',
      'custom-layers-list',
      'custom-layer-row',
      'layer-index',
      'layer-name-input',
      'layer-align-label',
      'layer-align-checkbox',
      'layers-preview-list',
      'align-badge',
      'batch-summary-banner',
      'batch-config-banner',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('batch-settings-tab__custom-layers')
    expect(source).toContain('batch-settings-tab__layer-row')
    expect(source).toContain('batch-settings-tab__layers-preview')
    expect(styleBlock).not.toMatch(/\.batch-settings-tab\s+\./)
  })
})
