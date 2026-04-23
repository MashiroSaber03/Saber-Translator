import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const initialDefaults = {
  fontSize: 26,
  autoFontSize: false,
  fontFamily: 'fonts/思源黑体SourceHanSansK-Bold.TTF',
  layoutDirection: 'auto',
  textColor: '#000000',
  fillColor: '#FFFFFF',
  inpaintMethod: 'solid',
  useAutoTextColor: false,
  strokeEnabled: true,
  strokeColor: '#FFFFFF',
  strokeWidth: 3,
  lineSpacing: 1,
  textAlign: 'start',
}

const factoryDefaults = {
  ...initialDefaults,
  fontSize: 31,
  textColor: '#223344',
}

const { getDefaultsMock, saveDefaultsMock, resetDefaultsMock, getFontListMock } = vi.hoisted(() => ({
  getDefaultsMock: vi.fn(),
  saveDefaultsMock: vi.fn(),
  resetDefaultsMock: vi.fn(),
  getFontListMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  configApi: {
    getTextStyleDefaults: getDefaultsMock,
    saveTextStyleDefaults: saveDefaultsMock,
    resetTextStyleDefaults: resetDefaultsMock,
    getFontList: getFontListMock,
    uploadFont: vi.fn(),
  },
}))

vi.mock('@/defaults/textStyleFactoryDefaults', () => ({
  getFactoryTextStyleDefaults: () => ({ ...factoryDefaults }),
}))

import TextStyleDefaultsSettings from '@/components/settings/TextStyleDefaultsSettings.vue'

describe('TextStyleDefaultsSettings', () => {
  beforeEach(() => {
    getDefaultsMock.mockReset()
    saveDefaultsMock.mockReset()
    resetDefaultsMock.mockReset()
    getFontListMock.mockReset()

    getDefaultsMock.mockResolvedValue({ success: true, defaults: { ...initialDefaults } })
    saveDefaultsMock.mockResolvedValue({ success: true, defaults: { ...factoryDefaults } })
    resetDefaultsMock.mockResolvedValue({ success: true, defaults: { ...factoryDefaults } })
    getFontListMock.mockResolvedValue({ fonts: ['fonts/思源黑体SourceHanSansK-Bold.TTF'] })
  })

  it('loads current defaults when the settings modal opens', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            template: '<select :value="modelValue"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
          },
        },
      },
    })

    await flushPromises()

    expect(getDefaultsMock).toHaveBeenCalledTimes(1)
    expect((wrapper.get('#textDefaultsFontSize').element as HTMLInputElement).value).toBe('26')
  })

  it('restores factory defaults into draft only until save is called', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            template: '<select :value="modelValue"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
          },
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')

    expect(saveDefaultsMock).not.toHaveBeenCalled()
    expect((wrapper.get('#textDefaultsFontSize').element as HTMLInputElement).value).toBe('31')
  })

  it('saves modified draft defaults through the exposed save method', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            template: '<select :value="modelValue"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
          },
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')

    const exposed = wrapper.vm as unknown as {
      saveDefaults: () => Promise<{ success: boolean; changed: boolean }>
    }

    await expect(exposed.saveDefaults()).resolves.toEqual({ success: true, changed: true })
    expect(resetDefaultsMock).toHaveBeenCalledTimes(1)
    expect(saveDefaultsMock).not.toHaveBeenCalled()
  })

  it('uses normal save when the user edits fields after resetting to factory defaults', async () => {
    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            template: '<select :value="modelValue"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
          },
        },
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="reset-text-style-defaults"]').trigger('click')
    await wrapper.get('#textDefaultsFontSize').setValue('35')

    const exposed = wrapper.vm as unknown as {
      saveDefaults: () => Promise<{ success: boolean; changed: boolean }>
    }

    await expect(exposed.saveDefaults()).resolves.toEqual({ success: true, changed: true })
    expect(saveDefaultsMock).toHaveBeenCalledWith({
      ...factoryDefaults,
      fontSize: 35,
    })
    expect(resetDefaultsMock).not.toHaveBeenCalled()
  })

  it('becomes a no-op when current defaults failed to load but the user did not touch text defaults', async () => {
    getDefaultsMock.mockResolvedValue({ success: false, error: 'load failed' })

    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            template: '<select :value="modelValue"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
          },
        },
      },
    })

    await flushPromises()

    const exposed = wrapper.vm as unknown as {
      saveDefaults: () => Promise<{ success: boolean; changed: boolean; error?: string }>
    }

    await expect(exposed.saveDefaults()).resolves.toEqual({
      success: true,
      changed: false,
    })
    expect(saveDefaultsMock).not.toHaveBeenCalled()
    expect(resetDefaultsMock).not.toHaveBeenCalled()
  })

  it('still reports an error if the user edits text defaults after load failure', async () => {
    getDefaultsMock.mockResolvedValue({ success: false, error: 'load failed' })

    const wrapper = mount(TextStyleDefaultsSettings, {
      props: { isOpen: true },
      global: {
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            template: '<select :value="modelValue"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>',
          },
        },
      },
    })

    await flushPromises()
    await wrapper.get('#textDefaultsFontSize').setValue('40')

    const exposed = wrapper.vm as unknown as {
      saveDefaults: () => Promise<{ success: boolean; changed: boolean; error?: string }>
    }

    await expect(exposed.saveDefaults()).resolves.toEqual({
      success: false,
      changed: false,
      error: '请先成功加载当前默认值，或先点击“恢复出厂默认”再保存'
    })
    expect(saveDefaultsMock).not.toHaveBeenCalled()
    expect(resetDefaultsMock).not.toHaveBeenCalled()
  })
})
