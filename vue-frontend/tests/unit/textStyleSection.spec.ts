import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import TextStyleSection from '@/components/translate/settings-sidebar/TextStyleSection.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { ApplySettingsOptions } from '@/components/translate/useSettingsSidebar'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import {
  blockAlignOptions,
  inlineAlignOptions,
  inpaintMethodOptions,
  layoutDirectionOptions,
} from '@/utils/textStyleForm'

const applyOptions: ApplySettingsOptions = {
  fontSize: true,
  fontFamily: true,
  layoutDirection: true,
  textColor: true,
  fillColor: true,
  strokeEnabled: true,
  strokeColor: true,
  strokeWidth: true,
  lineSpacing: true,
  inlineAlign: true,
  blockAlign: true,
}

describe('TextStyleSection', () => {
  it('uses fixed select primitives for layout, alignment, and fill method', () => {
    const wrapper = mount(TextStyleSection, {
      props: {
        applyOptions,
        fontSelectOptions: [{ label: '思源黑体', value: TEXT_STYLE_DEFAULTS.fontFamily }],
        hasImages: true,
        inpaintMethodOptions,
        layoutDirectionOptions,
        showApplyOptions: false,
        inlineAlignOptions,
        blockAlignOptions,
        textStyle: {
          ...TEXT_STYLE_DEFAULTS,
          layoutDirection: 'vertical',
          inlineAlign: 'center',
          blockAlign: 'end',
          inpaintMethod: 'solid',
        },
      },
    })

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects).toHaveLength(4)
    expect(selects.map(select => select.props('modelValue'))).toEqual([
      'vertical',
      'center',
      'end',
      'solid',
    ])

    selects[0]!.vm.$emit('change', 'horizontal')
    selects[1]!.vm.$emit('change', 'end')
    selects[2]!.vm.$emit('change', 'center')
    selects[3]!.vm.$emit('change', 'litelama')

    expect(wrapper.emitted('layoutDirectionChange')?.[0]).toEqual(['horizontal'])
    expect(wrapper.emitted('inlineAlignChange')?.[0]).toEqual(['end'])
    expect(wrapper.emitted('blockAlignChange')?.[0]).toEqual(['center'])
    expect(wrapper.emitted('inpaintMethodChange')?.[0]).toEqual(['litelama'])
  })

  it('uses shared number fields for numeric text-style controls', () => {
    const wrapper = mount(TextStyleSection, {
      props: {
        applyOptions,
        fontSelectOptions: [{ label: '思源黑体', value: TEXT_STYLE_DEFAULTS.fontFamily }],
        hasImages: true,
        inpaintMethodOptions,
        layoutDirectionOptions,
        showApplyOptions: false,
        inlineAlignOptions,
        blockAlignOptions,
        textStyle: {
          ...TEXT_STYLE_DEFAULTS,
          strokeEnabled: true,
        },
      },
    })

    const numberFields = wrapper.findAllComponents(UiNumberField)
    expect(numberFields.map(field => field.props('inputId'))).toEqual([
      'fontSize',
      'lineSpacing',
      'strokeWidth',
    ])

    numberFields[0]!.vm.$emit('update:modelValue', 24)
    numberFields[1]!.vm.$emit('update:modelValue', 1.4)
    numberFields[2]!.vm.$emit('update:modelValue', 3)

    expect(wrapper.emitted('updateFontSize')?.[0]).toEqual([24])
    expect(wrapper.emitted('updateLineSpacing')?.[0]).toEqual([1.4])
    expect(wrapper.emitted('updateStrokeWidth')?.[0]).toEqual([3])
    expect(wrapper.find('.compact-number-input').exists()).toBe(false)
    expect(numberFields.map(field => field.props('max'))).toEqual([
      undefined,
      undefined,
      undefined,
    ])
  })

  it('routes text-style labels and hints through typed settings fields', () => {
    const wrapper = mount(TextStyleSection, {
      props: {
        applyOptions,
        fontSelectOptions: [{ label: '思源黑体', value: TEXT_STYLE_DEFAULTS.fontFamily }],
        hasImages: true,
        inpaintMethodOptions,
        layoutDirectionOptions,
        showApplyOptions: false,
        inlineAlignOptions,
        blockAlignOptions,
        textStyle: {
          ...TEXT_STYLE_DEFAULTS,
          useAutoTextColor: true,
          inpaintMethod: 'solid',
          strokeEnabled: true,
        },
      },
    })

    const fields = wrapper.findAllComponents(UiField)
    const fieldContracts = new Map(
      fields.map(field => [
        field.props('label'),
        {
          controlId: field.props('controlId'),
          hint: field.props('hint'),
          variant: field.props('variant'),
        },
      ])
    )

    expect(fieldContracts.get('字号')).toEqual({
      controlId: 'fontSize',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('文本字体')).toEqual({
      controlId: 'fontFamily',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('排版方向')).toEqual({
      controlId: 'layoutDirection',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('行间距')).toEqual({
      controlId: 'lineSpacing',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('行内对齐')).toEqual({
      controlId: 'inlineAlign',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('文本块对齐')).toEqual({
      controlId: 'blockAlign',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('文字颜色')).toEqual({
      controlId: 'textColor',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('气泡填充方式')).toEqual({
      controlId: 'useInpainting',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('填充颜色')).toEqual({
      controlId: 'fillColor',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('描边颜色')).toEqual({
      controlId: 'strokeColor',
      hint: '',
      variant: 'settings',
    })
    expect(fieldContracts.get('描边宽度 (px)')).toEqual({
      controlId: 'strokeWidth',
      hint: '0 表示无描边。',
      variant: 'settings',
    })
  })

})
