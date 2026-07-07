import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import ProductThemeToggle from '@/components/product/ProductThemeToggle.vue'
import { useSettingsStore } from '@/stores/settings'

describe('ProductThemeToggle', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    document.documentElement.removeAttribute('data-theme')
    document.body.removeAttribute('data-theme')
  })

  it('toggles the current product theme through the settings store', async () => {
    const settingsStore = useSettingsStore()
    const wrapper = mount(ProductThemeToggle)

    expect(wrapper.attributes('aria-label')).toBe('切换深色模式')
    expect(wrapper.find('svg.ui-icon').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('☀️')
    expect(wrapper.text()).not.toContain('🌙')

    await wrapper.trigger('click')

    expect(settingsStore.theme).toBe('dark')
    expect(wrapper.attributes('aria-label')).toBe('切换跟随系统')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')

    await wrapper.trigger('click')

    expect(settingsStore.theme).toBe('system')
    expect(wrapper.attributes('aria-label')).toBe('切换浅色模式')
  })
})
