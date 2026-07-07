import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import TextStyleSection from '@/components/translate/settings-sidebar/TextStyleSection.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { ApplySettingsOptions } from '@/components/translate/useSettingsSidebar'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { inpaintMethodOptions, layoutDirectionOptions, textAlignOptions } from '@/utils/textStyleForm'

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
  textAlign: true,
}

describe('TextStyleSection', () => {
  it('uses fixed select primitives for layout, alignment, and fill method', () => {
    const wrapper = mount(TextStyleSection, {
      props: {
        applyOptions,
        fontSelectOptions: [{ label: '默认字体', value: TEXT_STYLE_DEFAULTS.fontFamily }],
        hasImages: true,
        inpaintMethodOptions,
        layoutDirectionOptions,
        showApplyOptions: false,
        textAlignOptions,
        textStyle: {
          ...TEXT_STYLE_DEFAULTS,
          layoutDirection: 'vertical',
          textAlign: 'center',
          inpaintMethod: 'solid',
        },
      },
    })

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects).toHaveLength(3)
    expect(selects.map(select => select.props('modelValue'))).toEqual([
      'vertical',
      'center',
      'solid',
    ])

    selects[0]!.vm.$emit('change', 'horizontal')
    selects[1]!.vm.$emit('change', 'end')
    selects[2]!.vm.$emit('change', 'litelama')

    expect(wrapper.emitted('layoutDirectionChange')?.[0]).toEqual(['horizontal'])
    expect(wrapper.emitted('textAlignChange')?.[0]).toEqual(['end'])
    expect(wrapper.emitted('inpaintMethodChange')?.[0]).toEqual(['litelama'])
  })

  it('uses shared number fields for numeric text-style controls', () => {
    const wrapper = mount(TextStyleSection, {
      props: {
        applyOptions,
        fontSelectOptions: [{ label: '默认字体', value: TEXT_STYLE_DEFAULTS.fontFamily }],
        hasImages: true,
        inpaintMethodOptions,
        layoutDirectionOptions,
        showApplyOptions: false,
        textAlignOptions,
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
  })

  it('routes text-style labels and hints through typed settings fields', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/TextStyleSection.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('ui-form-hint')
    expect(source).not.toContain('.settings-sidebar__field > label')
    expect(source).not.toContain('id="solidColorOptions"')
    expect(source).not.toContain('id="strokeOptions"')

    const wrapper = mount(TextStyleSection, {
      props: {
        applyOptions,
        fontSelectOptions: [{ label: '默认字体', value: TEXT_STYLE_DEFAULTS.fontFamily }],
        hasImages: true,
        inpaintMethodOptions,
        layoutDirectionOptions,
        showApplyOptions: false,
        textAlignOptions,
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
    expect(fieldContracts.get('对齐方式')).toEqual({
      controlId: 'textAlign',
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

  it('maps section owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/TextStyleSection.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('uses the shared color input primitive instead of local color input skins', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/TextStyleSection.vue'),
      'utf8'
    )

    expect(source).toContain("import UiColorInput from '@/components/ui/UiColorInput.vue'")
    expect(source).not.toContain('type="color"')
    expect(source).not.toContain('class="color-input')
    expect(source).not.toMatch(/\.color-input\b/)
  })

  it('keeps text-style section layout hooks under the section owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/TextStyleSection.vue'),
      'utf8'
    )

    for (const oldClass of [
      'settings-panel',
      'text-settings-panel',
      'settings-form',
      'text-settings-form',
      'setting-group',
      'setting-group-typography',
      'setting-group-color',
      'setting-group-stroke',
      'group-title-row',
      'group-title',
      'group-note',
      'color-field-row',
      'settings-toggle',
      'auto-fontsize-toggle',
      'auto-color-toggle',
      'stroke-toggle',
      'inline-color-group',
      'inline-hint',
      'stroke-options',
      'stroke-grid',
      'settings-sidebar__field',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }

    for (const ownerClass of [
      'text-style-section',
      'text-style-section__form',
      'text-style-section__group',
      'text-style-section__group--typography',
      'text-style-section__group--color',
      'text-style-section__group--stroke',
      'text-style-section__group-title-row',
      'text-style-section__field',
      'text-style-section__color-field-row',
      'text-style-section__toggle',
      'text-style-section__inline-hint',
      'text-style-section__stroke-grid',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
