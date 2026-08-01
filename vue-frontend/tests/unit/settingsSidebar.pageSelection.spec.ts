import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { enableAutoUnmount, mount } from '@vue/test-utils'
import { defineComponent, h, type PropType } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

vi.mock('@/api/v2/settings', async importOriginal => ({
  ...await importOriginal<typeof import('@/api/v2/settings')>(),
  listV2Fonts: async () => [],
  uploadV2Font: async () => ({ id: 'font-uploaded', assetUrl: '/api/v2/assets/font' }),
}))

vi.mock('@/components/ui/UiCombobox.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: [String, Number] as PropType<string | number | undefined>,
        default: undefined,
      },
      options: {
        type: Array as PropType<Array<{ label: string; value: string | number }>>,
        default: () => [],
      },
    },
    emits: ['change'],
    setup(props, { emit }) {
      return () => h(
        'select',
        {
          value: props.modelValue,
          onChange: (event: Event) => emit('change', (event.target as HTMLSelectElement).value),
        },
        props.options.map(option => h('option', { value: option.value }, option.label))
      )
    },
  }),
}))

vi.mock('@/components/product/ProductCollapsibleSection.vue', () => ({
  default: defineComponent({
    props: {
      title: {
        type: String,
        default: '',
      },
    },
    setup(props, { slots }) {
      return () => h('section', [h('h3', props.title), slots.default?.()])
    },
  }),
}))

vi.mock('@/components/translate/PageSelectionModal.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: Boolean,
        default: false,
      },
      selectedPages: {
        type: Array as PropType<number[]>,
        default: () => [],
      },
    },
    emits: ['update:modelValue', 'confirm'],
    setup(props, { emit }) {
      return () => props.modelValue
        ? h('div', { class: 'page-selection-modal-stub' }, [
            h('button', {
              class: 'confirm-selection',
              onClick: () => emit('confirm', [1, 3, 8, 10]),
            }, 'confirm'),
            h('button', {
              class: 'close-selection',
              onClick: () => emit('update:modelValue', false),
            }, 'close'),
          ])
        : null
    },
  }),
}))

import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'
import PageSelectionSection from '@/components/translate/settings-sidebar/PageSelectionSection.vue'
import TextStyleSection from '@/components/translate/settings-sidebar/TextStyleSection.vue'
import WorkflowSection from '@/components/translate/settings-sidebar/WorkflowSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import { useImageStore } from '@/stores/imageStore'

describe('SettingsSidebar page selection workflow', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    setActivePinia(createPinia())
    const imageStore = useImageStore()
    imageStore.clearImages()
    imageStore.addImage('001.png', 'data:image/png;base64,aaa')
    imageStore.addImage('002.png', 'data:image/png;base64,bbb')
    imageStore.addImage('003.png', 'data:image/png;base64,ccc')
    imageStore.addImage('004.png', 'data:image/png;base64,ddd')
    imageStore.addImage('005.png', 'data:image/png;base64,eee')
    imageStore.addImage('006.png', 'data:image/png;base64,fff')
    imageStore.addImage('007.png', 'data:image/png;base64,ggg')
    imageStore.addImage('008.png', 'data:image/png;base64,hhh')
    imageStore.addImage('009.png', 'data:image/png;base64,iii')
    imageStore.addImage('010.png', 'data:image/png;base64,jjj')
    imageStore.updateCurrentImage({ bubbleStates: [] })
  })

  it('opens page selection modal and emits selected pages for batch workflow', async () => {
    const wrapper = mount(SettingsSidebar)

    const workflowModeSelect = wrapper.findAllComponents(UiSelect)
      .find(select => select.attributes('id') === 'workflowModeSelect')
    expect(workflowModeSelect).toBeTruthy()
    workflowModeSelect!.vm.$emit('change', 'translate-batch')

    const enableSwitch = wrapper.getComponent(UiSwitch)
    expect(enableSwitch.props('accessibilityLabel')).toBe('启用指定翻译页码')
    expect(enableSwitch.props('modelValue')).toBe(false)
    enableSwitch.vm.$emit('change', true)
    await wrapper.vm.$nextTick()

    const openPageSelectionButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '选择页码')
    expect(openPageSelectionButton).toBeDefined()
    await openPageSelectionButton!.trigger('click')
    expect(wrapper.find('.page-selection-modal-stub').exists()).toBe(true)

    await wrapper.find('.confirm-selection').trigger('click')

    const runButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '启动批量翻译')
    expect(runButton).toBeDefined()
    await runButton!.trigger('click')

    expect(wrapper.emitted('runWorkflow')?.[0]).toEqual([
      {
        mode: 'translate-batch',
        pageSelection: {
          pages: [1, 3, 8, 10],
        },
      },
    ])
  })

  it('keeps page settings and workflows disabled until the page document is ready', async () => {
    const imageStore = useImageStore()
    imageStore.updateCurrentImage({ bubbleStates: null })
    const wrapper = mount(SettingsSidebar)

    expect(wrapper.getComponent(TextStyleSection).props('disabled')).toBe(true)
    expect(wrapper.getComponent(WorkflowSection).props('canRunWorkflow')).toBe(false)

    imageStore.updateCurrentImage({ bubbleStates: [] })
    await wrapper.vm.$nextTick()

    expect(wrapper.getComponent(TextStyleSection).props('disabled')).toBe(false)
    expect(wrapper.getComponent(WorkflowSection).props('canRunWorkflow')).toBe(true)
  })

  it('uses product status/action primitives and semantic tokens for page selection', () => {
    const wrapper = mount(PageSelectionSection, {
      props: {
        enabled: true,
        hasValidPageSelection: false,
        isActive: true,
        normalizedSelectedPages: [],
        supportsPageSelection: false,
        totalImages: 10,
        summaryFor: () => '未选择页码',
      },
    })

    const banners = wrapper.findAllComponents(ProductStatusBanner)
    expect(banners.map(banner => banner.props('tone'))).toEqual(['neutral', 'warning', 'danger'])

    const openButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '选择页码')
    expect(openButton).toBeDefined()
    expect(openButton.props('variant')).toBe('secondary')
    expect(openButton.props('size')).toBe('sm')
    expect(openButton.props('block')).toBe(true)
    expect(wrapper.find('.secondary-button').exists()).toBe(false)
    expect(wrapper.find('.page-selection-open-btn').exists()).toBe(false)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/PageSelectionSection.vue'),
      'utf8'
    )
    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('page-selection-open-btn')
    expect(source).not.toMatch(/\.page-selection-open-btn\s*\{[\s\S]*?(display|justify-content|padding)/)
  })

  it('keeps page-selection section hooks under the page-selection owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/PageSelectionSection.vue'),
      'utf8'
    )

    for (const currentHook of [
      'page-selection-section',
      'page-selection-section__form',
      'page-selection-section__header',
      'page-selection-section__enable-control',
      'page-selection-section__total-count',
      'page-selection-section__summary',
      'page-selection-section__summary-value',
      'page-selection-section__note',
      'page-selection-section__error',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'settings-panel',
      'settings-form',
      'page-selection-form',
      'range-header-row',
      'page-selection-enable-control',
      'total-count',
      'page-selection-summary-block',
      'page-selection-note',
      'page-selection-error',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
    }
  })
})
