import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia, type Pinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import BatchSettingsTab from '@/components/insight/settings/BatchSettingsTab.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import { useInsightStore } from '@/stores/insightStore'
import type { CustomLayer } from '@/components/insight/settings/types'

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

    expect(wrapper.vm.getConfig().customLayers).toEqual([
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
})
