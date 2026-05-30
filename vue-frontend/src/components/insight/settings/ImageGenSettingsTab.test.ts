import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ImageGenSettingsTab from './ImageGenSettingsTab.vue'
import { IMAGE_GEN_PROVIDER_OPTIONS } from './types'
import { useInsightStore } from '@/stores/insightStore'

describe('ImageGenSettingsTab', () => {
  it('renders the image generation provider as a selector', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            props: ['modelValue', 'options'],
            template: '<div class="custom-select-stub" />',
          },
        },
      },
    })

    const providerSelect = wrapper.findComponent({ name: 'CustomSelect' })

    expect(providerSelect.exists()).toBe(true)
    expect(providerSelect.props('options')).toEqual(IMAGE_GEN_PROVIDER_OPTIONS)
    expect(wrapper.vm.getConfig().provider).toBe('gpt2api')
  })

  it('syncs provider changes from the store', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.updateImageGenConfig({
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: 'https://gateway.example.com/v1',
      transportRetries: 5,
      businessRetries: 6,
      timeoutSeconds: 7,
    })

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            props: ['modelValue', 'options'],
            template: '<div class="custom-select-stub" />',
          },
        },
      },
    })

    wrapper.vm.syncFromStore()

    expect(wrapper.vm.getConfig()).toEqual({
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: 'https://gateway.example.com/v1',
      transportRetries: 5,
      businessRetries: 6,
      timeoutSeconds: 7,
    })
  })
})

