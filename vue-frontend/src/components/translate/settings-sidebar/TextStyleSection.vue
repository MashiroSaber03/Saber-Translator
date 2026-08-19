<script setup lang="ts">
import ProductCollapsibleSection from '@/components/product/ProductCollapsibleSection.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { ref } from 'vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'
import type { TextStyleSettings } from '@/types/settings'
import type { ApplySettingsOptions } from '../useSettingsSidebar'
import ApplyOptionsSection from './ApplyOptionsSection.vue'

withDefaults(defineProps<{
  applyOptions: ApplySettingsOptions
  disabled?: boolean
  fontSelectOptions: UiSelectOption[]
  hasImages: boolean
  inpaintMethodOptions: UiSelectOption[]
  layoutDirectionOptions: UiSelectOption[]
  showApplyOptions: boolean
  inlineAlignOptions: UiSelectOption[]
  blockAlignOptions: UiSelectOption[]
  textStyle: TextStyleSettings
}>(), {
  disabled: false,
})

defineEmits<{
  (event: 'apply'): void
  (event: 'fontSelectChange', value: UiSelectValue): void
  (event: 'inpaintMethodChange', value: UiSelectValue): void
  (event: 'layoutDirectionChange', value: UiSelectValue): void
  (event: 'selectAll'): void
  (event: 'inlineAlignChange', value: UiSelectValue): void
  (event: 'blockAlignChange', value: UiSelectValue): void
  (event: 'toggleApplyOptions'): void
  (event: 'updateApplyOption', key: keyof ApplySettingsOptions, value: boolean): void
  (event: 'updateAutoFontSize', value: boolean): void
  (event: 'updateFillColor', value: string): void
  (event: 'updateFontSize', value: number): void
  (event: 'updateLineSpacing', value: number): void
  (event: 'updateStrokeColor', value: string): void
  (event: 'updateStrokeEnabled', value: boolean): void
  (event: 'updateStrokeWidth', value: number): void
  (event: 'updateTextColor', value: string): void
  (event: 'updateUseAutoTextColor', value: boolean): void
}>()

const isTextStyleExpanded = ref(true)
</script>

<template>
  <ProductCollapsibleSection
    v-model:expanded="isTextStyleExpanded"
    title="文字设置"
    class="text-style-section"
  >
    <div class="text-style-section__form">
      <section class="text-style-section__group text-style-section__group--typography">
        <div class="text-style-section__group-title-row">
          <h3 class="text-style-section__group-title">字体排版</h3>
          <span class="text-style-section__group-note">影响新翻译文本</span>
        </div>
        <UiField
          class="text-style-section__field"
          variant="settings"
          label="字号"
          control-id="fontSize"
        >
          <UiNumberField
            input-id="fontSize"
            :model-value="textStyle.fontSize"
            :min="1"
            :disabled="disabled || textStyle.autoFontSize"
            aria-label="字号"
            :title="textStyle.autoFontSize ? '已启用自动字号，首次翻译时将自动计算' : ''"
            @update:model-value="$event !== null && $emit('updateFontSize', $event)"
          />
          <UiCheckbox
            class="text-style-section__toggle text-style-section__toggle--auto-fontsize"
            input-id="autoFontSize"
            :model-value="textStyle.autoFontSize"
            :disabled="disabled"
            label="自动计算初始字号"
            @change="$emit('updateAutoFontSize', $event)"
          />
        </UiField>

        <UiField
          class="text-style-section__field"
          variant="settings"
          label="文本字体"
          control-id="fontFamily"
        >
          <UiCombobox
            input-id="fontFamily"
            aria-label="文本字体"
            :model-value="textStyle.fontFamily"
            :disabled="disabled"
            :options="fontSelectOptions"
            @change="$emit('fontSelectChange', $event)"
          />
        </UiField>

        <UiField
          class="text-style-section__field"
          variant="settings"
          label="排版方向"
          control-id="layoutDirection"
        >
          <UiSelect
            id="layoutDirection"
            :model-value="textStyle.layoutDirection"
            :options="layoutDirectionOptions"
            :disabled="disabled"
            @change="$emit('layoutDirectionChange', $event)"
          />
        </UiField>

        <UiField
          class="text-style-section__field"
          variant="settings"
          label="行间距"
          control-id="lineSpacing"
        >
          <UiNumberField
            input-id="lineSpacing"
            :model-value="textStyle.lineSpacing"
            :min="0.1"
            :step="0.1"
            :disabled="disabled"
            aria-label="行间距"
            @update:model-value="$event !== null && $emit('updateLineSpacing', $event)"
          />
        </UiField>

        <UiField
          class="text-style-section__field"
          variant="settings"
          label="行内对齐"
          control-id="inlineAlign"
        >
          <UiSelect
            id="inlineAlign"
            :model-value="textStyle.inlineAlign"
            :options="inlineAlignOptions"
            :disabled="disabled"
            @change="$emit('inlineAlignChange', $event)"
          />
        </UiField>

        <UiField
          class="text-style-section__field"
          variant="settings"
          label="文本块对齐"
          control-id="blockAlign"
        >
          <UiSelect
            id="blockAlign"
            :model-value="textStyle.blockAlign"
            :options="blockAlignOptions"
            :disabled="disabled"
            @change="$emit('blockAlignChange', $event)"
          />
        </UiField>
      </section>

      <section class="text-style-section__group text-style-section__group--color">
        <div class="text-style-section__group-title-row">
          <h3 class="text-style-section__group-title">颜色与填充</h3>
        </div>
        <UiField
          class="text-style-section__field"
          variant="settings"
          label="文字颜色"
          control-id="textColor"
        >
          <div class="text-style-section__color-field-row">
            <UiColorInput
              input-id="textColor"
              :model-value="textStyle.textColor"
              :disabled="disabled || textStyle.useAutoTextColor"
              aria-label="文字颜色"
              size="sm"
              @update:model-value="$emit('updateTextColor', $event)"
            />
            <UiCheckbox
              class="text-style-section__toggle text-style-section__toggle--auto-color"
              input-id="useAutoTextColor"
              :model-value="textStyle.useAutoTextColor"
              :disabled="disabled"
              label="自动"
              @change="$emit('updateUseAutoTextColor', $event)"
            />
          </div>
          <div v-if="textStyle.useAutoTextColor" class="text-style-section__inline-hint">
            翻译时将自动使用识别到的文字颜色
          </div>
        </UiField>

        <UiField
          class="text-style-section__field"
          variant="settings"
          label="气泡填充方式"
          control-id="useInpainting"
        >
          <UiSelect
            id="useInpainting"
            :model-value="textStyle.inpaintMethod"
            :options="inpaintMethodOptions"
            :disabled="disabled"
            @change="$emit('inpaintMethodChange', $event)"
          />
        </UiField>

        <Transition name="slide-fade">
          <UiField
            v-if="textStyle.inpaintMethod === 'solid'"
            class="text-style-section__field text-style-section__field--inline-color"
            variant="settings"
            label="填充颜色"
            control-id="fillColor"
          >
            <UiColorInput
              input-id="fillColor"
              :model-value="textStyle.fillColor"
              :disabled="disabled"
              aria-label="填充颜色"
              @update:model-value="$emit('updateFillColor', $event)"
            />
          </UiField>
        </Transition>
      </section>

      <section class="text-style-section__group text-style-section__group--stroke">
        <div class="text-style-section__group-title-row">
          <h3 class="text-style-section__group-title">描边</h3>
          <UiCheckbox
            class="text-style-section__toggle text-style-section__toggle--stroke"
            input-id="strokeEnabled"
            :model-value="textStyle.strokeEnabled"
            :disabled="disabled"
            label="启用描边"
            @change="$emit('updateStrokeEnabled', $event)"
          />
        </div>

        <Transition name="stroke-slide">
          <div v-if="textStyle.strokeEnabled" class="text-style-section__stroke-options">
            <div class="text-style-section__stroke-grid">
              <UiField
                class="text-style-section__field"
                variant="settings"
                label="描边颜色"
                control-id="strokeColor"
              >
                <UiColorInput
                  input-id="strokeColor"
                  :model-value="textStyle.strokeColor"
                  :disabled="disabled"
                  aria-label="描边颜色"
                  @update:model-value="$emit('updateStrokeColor', $event)"
                />
              </UiField>
              <UiField
                class="text-style-section__field"
                variant="settings"
                label="描边宽度 (px)"
                control-id="strokeWidth"
                hint="0 表示无描边。"
              >
                <UiNumberField
                  input-id="strokeWidth"
                  :model-value="textStyle.strokeWidth"
                  :min="0"
                  :disabled="disabled"
                  aria-label="描边宽度"
                  @update:model-value="$event !== null && $emit('updateStrokeWidth', $event)"
                />
              </UiField>
            </div>
          </div>
        </Transition>
      </section>
    </div>

    <ApplyOptionsSection
      :apply-options="applyOptions"
      :disabled="disabled"
      :has-images="hasImages"
      :show-apply-options="showApplyOptions"
      @apply="$emit('apply')"
      @toggle-options="$emit('toggleApplyOptions')"
      @toggle-select-all="$emit('selectAll')"
      @update-option="(key, value) => $emit('updateApplyOption', key, value)"
    />
  </ProductCollapsibleSection>
</template>

<style scoped>
.text-style-section.product-collapsible-section {
  --settings-sidebar-text-style-panel-border: var(--color-border-muted);
  --settings-sidebar-text-style-panel-background: var(--color-surface-quiet);
  --settings-sidebar-text-style-divider-typography: var(--color-border-muted);
  --settings-sidebar-text-style-divider-color: var(--color-action-success);
  --settings-sidebar-text-style-divider-stroke: var(--color-status-warning);
  --settings-sidebar-text-style-title-divider: var(--color-border-muted);
  --settings-sidebar-text-style-title: var(--color-text-heading);
  --settings-sidebar-text-style-title-note: var(--color-text-supporting);
  --settings-sidebar-text-style-header-divider: var(--settings-sidebar-text-style-title-divider);
  --settings-sidebar-text-style-toggle-background: var(--color-surface-muted);
  --settings-sidebar-text-style-toggle-border: var(--color-border-muted);
  --settings-sidebar-text-style-toggle-text: var(--color-text-secondary);
  --settings-sidebar-text-style-toggle-checked-border: var(--color-border-brand);
  --settings-sidebar-text-style-toggle-checked-background: var(--color-focus-brand-soft);
  --settings-sidebar-text-style-toggle-checked-text: var(--color-text-brand);
  --settings-sidebar-text-style-group-title: var(--settings-sidebar-text-style-title);
  --settings-sidebar-text-style-group-note: var(--settings-sidebar-text-style-title-note);
  --settings-sidebar-text-style-field-label: var(--color-text-default);
  --settings-sidebar-text-style-field-hint: var(--color-text-supporting);
  --settings-sidebar-text-style-hint-border: var(--color-border-info);
  --settings-sidebar-text-style-hint-background: var(--color-surface-interactive-hover);
  --settings-sidebar-text-style-hint-text: var(--color-text-link);
  --settings-sidebar-text-style-stroke-divider: var(--color-border-muted);
  --product-collapsible-section-header-gap: 0;
  --product-collapsible-section-header-margin: 0 0 10px;
  --product-collapsible-section-header-padding: 0 0 8px;
  --product-collapsible-section-header-border: 0;
  --product-collapsible-section-header-border-bottom: 1px solid var(--settings-sidebar-text-style-header-divider);
  --product-collapsible-section-header-background: transparent;
  --product-collapsible-section-header-hover-background: transparent;
  --product-collapsible-section-title-color: var(--color-text-heading);
  --product-collapsible-section-title-font-size: 17px;
  --product-collapsible-section-title-font-weight: 700;
  --product-collapsible-section-title-order: 1;
  --product-collapsible-section-hint-order: 2;
  --product-collapsible-section-toggle-order: 3;
  --product-collapsible-section-toggle-margin-left: auto;
  --product-collapsible-section-toggle-color: var(--color-text-supporting);
  --product-collapsible-section-body-padding: 2px 0 0;
  --product-collapsible-section-body-background: transparent;
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-input-min-height: 40px;
  --ui-input-padding: 9px 10px;
  --ui-input-border: 1px solid var(--color-border-input);
  --ui-input-radius: 8px;
  --ui-input-background: var(--color-surface-input);
  --ui-input-font-size: 14px;
  --ui-selector-control-background: var(--color-surface-input);
  --ui-selector-control-border: var(--color-border-input);

  margin: 0 0 12px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-text-style-panel-border);
  border-radius: 12px;
  background: var(--settings-sidebar-text-style-panel-background);
}

.text-style-section__form {
  display: flex;
  flex-direction: column;
}

.text-style-section__group {
  margin: 0;
  padding: 10px 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}

.text-style-section__group:last-child {
  margin-bottom: 0;
}

.text-style-section__group + .text-style-section__group {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 3px solid var(--settings-sidebar-group-divider-color);
}

.text-style-section__group--typography {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-text-style-divider-typography);
}

.text-style-section__group--color {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-text-style-divider-color);
}

.text-style-section__group--stroke {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-text-style-divider-stroke);
}

.text-style-section__group--color + .text-style-section__group--stroke {
  margin-top: 8px;
}

.text-style-section__group-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
  padding: 0 0 10px;
  border-bottom: 1px solid var(--settings-sidebar-text-style-title-divider);
}

.text-style-section__group-title {
  margin: 0;
  color: var(--settings-sidebar-text-style-group-title);
  font-weight: 700;
  font-size: 14px;
  line-height: 1.2;
}

.text-style-section__group-note {
  color: var(--settings-sidebar-text-style-group-note);
  font-size: 11px;
  line-height: 1.2;
}

.text-style-section__field {
  --ui-field-label-color: var(--settings-sidebar-text-style-field-label);
  --ui-field-label-font-size: 13px;
  --ui-field-label-font-weight: 600;
  --ui-field-settings-header-margin-bottom: 3px;
  --ui-field-hint-color: var(--settings-sidebar-text-style-field-hint);
  --ui-field-message-font-size: 11px;
  --ui-field-message-line-height: 1.3;

  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 11px;
}

.text-style-section__field:last-child {
  margin-bottom: 0;
}

.text-style-section__color-field-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.text-style-section__toggle {
  --ui-checkbox-gap: 6px;
  --ui-checkbox-color: var(--settings-sidebar-text-style-toggle-text);
  --ui-checkbox-input-width: 14px;
  --ui-checkbox-input-height: 14px;
  --ui-checkbox-input-margin: 0;
  --ui-checkbox-input-accent-color: var(--color-action-primary);
  --ui-checkbox-label-font-weight: 500;
  --ui-checkbox-checked-border-color: var(--settings-sidebar-text-style-toggle-checked-border);
  --ui-checkbox-checked-background: var(--settings-sidebar-text-style-toggle-checked-background);
  --ui-checkbox-checked-color: var(--settings-sidebar-text-style-toggle-checked-text);

  width: fit-content;
  padding: 5px 10px;
  border: 1px solid var(--settings-sidebar-text-style-toggle-border);
  border-radius: 999px;
  background: var(--settings-sidebar-text-style-toggle-background);
  font-size: 12px;
}

.text-style-section__toggle--auto-fontsize {
  margin-top: 2px;
}

.text-style-section__field--inline-color {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

.text-style-section__inline-hint {
  padding: 6px 8px;
  border: 1px solid var(--settings-sidebar-text-style-hint-border);
  border-radius: 8px;
  background: var(--settings-sidebar-text-style-hint-background);
  color: var(--settings-sidebar-text-style-hint-text);
  font-size: 12px;
  line-height: 1.35;
}

.text-style-section__stroke-options {
  margin-top: 8px;
  padding: 8px 0 0;
  border-top: 1px dashed var(--settings-sidebar-text-style-stroke-divider);
  border-radius: 0;
  background: transparent;
}

.text-style-section__stroke-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.slide-fade-enter-active {
  transition: all 0.28s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  max-height: 0;
  overflow: hidden;
  opacity: 0;
}

.slide-fade-enter-to,
.slide-fade-leave-from {
  max-height: 70px;
  opacity: 1;
}

.stroke-slide-enter-active {
  transition: all 0.3s ease-out;
}

.stroke-slide-leave-active {
  transition: all 0.2s ease-in;
}

.stroke-slide-enter-from,
.stroke-slide-leave-to {
  max-height: 0;
  overflow: hidden;
  opacity: 0;
}

.stroke-slide-enter-to,
.stroke-slide-leave-from {
  max-height: 220px;
  opacity: 1;
}
</style>
