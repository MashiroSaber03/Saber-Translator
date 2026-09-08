<template>
  <div class="text-style-form">
    <ProductFormSection>
      <template #title>字体排版</template>
      <UiFormGrid>
        <UiField variant="settings" label="字号" :control-id="id('FontSize')">
          <UiNumberField
            :input-id="id('FontSize')"
            :model-value="draftDefaults.fontSize"
            :min="1"
            :disabled="draftDefaults.autoFontSize"
            size="sm"
            @change="updateFontSize"
          />
        </UiField>
        <UiField
          variant="settings"
          control="checkbox"
          label="自动计算初始字号"
          :control-id="id('AutoFontSize')"
        >
          <UiCheckbox
            :input-id="id('AutoFontSize')"
            :model-value="draftDefaults.autoFontSize"
            @change="updateAutoFontSize"
          />
        </UiField>
      </UiFormGrid>

      <UiField variant="settings" label="文本字体" :control-id="id('FontFamily')">
        <UiCombobox
          :input-id="id('FontFamily')"
          aria-label="文本字体"
          :model-value="draftDefaults.fontFamily"
          :options="fontSelectOptions"
          :disabled="fontDisabled"
          @change="handleFontSelectChange"
        />
        <slot name="font-extra" />
      </UiField>

      <UiFormGrid>
        <UiField variant="settings" label="排版方向" :control-id="id('LayoutDirection')">
          <UiSelect
            :id="id('LayoutDirection')"
            :model-value="draftDefaults.layoutDirection"
            :options="layoutDirectionOptions"
            @change="handleLayoutDirectionChange"
          />
        </UiField>
        <UiField variant="settings" label="行内对齐" :control-id="id('InlineAlign')">
          <UiSelect
            :id="id('InlineAlign')"
            :model-value="draftDefaults.inlineAlign"
            :options="inlineAlignOptions"
            @change="handleInlineAlignChange"
          />
        </UiField>
        <UiField variant="settings" label="文本块对齐" :control-id="id('BlockAlign')">
          <UiSelect
            :id="id('BlockAlign')"
            :model-value="draftDefaults.blockAlign"
            :options="blockAlignOptions"
            @change="handleBlockAlignChange"
          />
        </UiField>
      </UiFormGrid>

      <UiField
        variant="settings"
        label="行间距"
        :control-id="id('LineSpacing')"
        hint="行间距倍数，必须大于 0。"
      >
        <UiNumberField
          :input-id="id('LineSpacing')"
          :model-value="draftDefaults.lineSpacing"
          :min="0.1"
          :step="0.1"
          size="sm"
          @change="updateLineSpacing"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>颜色与填充</template>
      <UiField
        variant="settings"
        control="checkbox"
        label="自动识别文字颜色"
        :control-id="id('UseAutoTextColor')"
      >
        <UiCheckbox
          :input-id="id('UseAutoTextColor')"
          :model-value="draftDefaults.useAutoTextColor"
          @change="updateUseAutoTextColor"
        />
      </UiField>
      <UiFormGrid>
        <UiField variant="settings" label="文字颜色" :control-id="id('TextColor')">
          <UiColorInput
            :input-id="id('TextColor')"
            :model-value="draftDefaults.textColor"
            :disabled="draftDefaults.useAutoTextColor"
            aria-label="文字颜色"
            size="sm"
            @update:model-value="updateTextColor"
          />
        </UiField>
        <UiField variant="settings" label="气泡填充方式" :control-id="id('InpaintMethod')">
          <UiSelect
            :id="id('InpaintMethod')"
            :model-value="draftDefaults.inpaintMethod"
            :options="inpaintMethodOptions"
            @change="handleInpaintMethodChange"
          />
        </UiField>
      </UiFormGrid>
      <UiField
        v-if="draftDefaults.inpaintMethod === 'solid'"
        variant="settings"
        label="填充颜色"
        :control-id="id('FillColor')"
      >
        <UiColorInput
          :input-id="id('FillColor')"
          :model-value="draftDefaults.fillColor"
          aria-label="填充颜色"
          @update:model-value="updateFillColor"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>描边</template>
      <UiField
        variant="settings"
        control="checkbox"
        label="启用描边"
        :control-id="id('StrokeEnabled')"
      >
        <UiCheckbox
          :input-id="id('StrokeEnabled')"
          :model-value="draftDefaults.strokeEnabled"
          @change="updateStrokeEnabled"
        />
      </UiField>
      <UiFormGrid v-if="draftDefaults.strokeEnabled">
        <UiField variant="settings" label="描边颜色" :control-id="id('StrokeColor')">
          <UiColorInput
            :input-id="id('StrokeColor')"
            :model-value="draftDefaults.strokeColor"
            aria-label="描边颜色"
            @update:model-value="updateStrokeColor"
          />
        </UiField>
        <UiField
          variant="settings"
          label="描边宽度 (px)"
          :control-id="id('StrokeWidth')"
          hint="0 表示无描边。"
        >
          <UiNumberField
            :input-id="id('StrokeWidth')"
            :model-value="draftDefaults.strokeWidth"
            :spin-step="1"
            :min="0"
            :step="0.1"
            size="sm"
            @change="updateStrokeWidth"
          />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TextStyleSettings } from '@/types/textStyleSettings'
import type { UiSelectOption } from '@/components/ui/selectTypes'
import {
  layoutDirectionOptions,
  inlineAlignOptions,
  blockAlignOptions,
  inpaintMethodOptions as defaultInpaintOptions,
} from '@/utils/textStyleForm'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'

const props = withDefaults(
  defineProps<{
    modelValue: TextStyleSettings
    fontSelectOptions: UiSelectOption[]
    fontDisabled?: boolean
    inpaintMethodOptions?: UiSelectOption[]
    idPrefix?: string
  }>(),
  {
    fontDisabled: false,
    idPrefix: 'textDefaults',
    inpaintMethodOptions: () => defaultInpaintOptions,
  }
)
const emit = defineEmits<{
  change: [value: Partial<TextStyleSettings>]
  'font-change': [value: string | number]
}>()
const draftDefaults = computed(() => props.modelValue)
const id = (field: string) => props.idPrefix + field
function updateDraft(updates: Partial<TextStyleSettings>): void {
  emit('change', updates)
}
function handleFontSelectChange(value: string | number): void {
  emit('font-change', value)
}
function updateFontSize(value: number | null): void {
  if (value !== null && Number.isInteger(value) && value >= 1) {
    updateDraft({ fontSize: value })
  }
}

function updateAutoFontSize(value: boolean): void {
  updateDraft({ autoFontSize: value })
}

function handleLayoutDirectionChange(value: string | number): void {
  if (value !== 'auto' && value !== 'vertical' && value !== 'horizontal') return
  updateDraft({ layoutDirection: value })
}

function handleInlineAlignChange(value: string | number): void {
  if (value !== 'start' && value !== 'center' && value !== 'end') return
  updateDraft({ inlineAlign: value })
}

function handleBlockAlignChange(value: string | number): void {
  if (value !== 'start' && value !== 'center' && value !== 'end') return
  updateDraft({ blockAlign: value })
}

function handleInpaintMethodChange(value: string | number): void {
  if (value !== 'solid' && value !== 'lama_mpe' && value !== 'litelama') return
  updateDraft({ inpaintMethod: value })
}

function updateLineSpacing(value: number | null): void {
  if (value !== null && Number.isFinite(value) && value > 0) {
    updateDraft({ lineSpacing: value })
  }
}

function updateTextColor(value: string): void {
  updateDraft({ textColor: value })
}

function updateUseAutoTextColor(value: boolean): void {
  updateDraft({ useAutoTextColor: value })
}

function updateFillColor(value: string): void {
  updateDraft({ fillColor: value })
}

function updateStrokeEnabled(value: boolean): void {
  updateDraft({ strokeEnabled: value })
}

function updateStrokeColor(value: string): void {
  updateDraft({ strokeColor: value })
}

function updateStrokeWidth(value: number | null): void {
  if (value !== null && Number.isFinite(value) && value >= 0) {
    updateDraft({ strokeWidth: value })
  }
}
</script>

<style scoped>
.text-style-form {
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-number-field-text-align: left;
}
</style>
