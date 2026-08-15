<script setup lang="ts">
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import type { WebImportSettings } from '@/types/webImport'
import type { WebImportSettingsActions } from './web-import/webImportSettingsActions'

type TargetFormat = WebImportSettings['imagePreprocess']['formatConvert']['targetFormat']

const props = defineProps<{
  draftSettings: WebImportSettings
  settingsActions: WebImportSettingsActions
}>()

const targetFormatOptions: Array<{
  label: string
  value: TargetFormat
}> = [
  { label: '保持原格式', value: 'original' },
  { label: 'JPEG', value: 'jpeg' },
  { label: 'PNG', value: 'png' },
  { label: 'WebP', value: 'webp' },
]

const targetFormatValues = targetFormatOptions.map(option => option.value)

function isTargetFormat(value: string | number): value is TargetFormat {
  return typeof value === 'string' && targetFormatValues.some(format => format === value)
}

function updateTargetFormat(value: string | number): void {
  if (!isTargetFormat(value)) return
  props.settingsActions.setImageTargetFormat(value)
}

function applyNumber(action: (value: number) => void, value: number | null): void {
  if (value !== null) action(value)
}
</script>

<template>
  <ProductFormSection class="web-import-preprocess__section">
    <UiField variant="settings" control="checkbox">
      <UiCheckbox
        :model-value="draftSettings.imagePreprocess.enabled"
        label="启用图片预处理"
        @change="settingsActions.setImagePreprocessEnabled"
      />
    </UiField>

    <template v-if="draftSettings.imagePreprocess.enabled">
      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.imagePreprocess.autoRotate"
          label="根据 EXIF 自动旋转"
          @change="settingsActions.setImageAutoRotate"
        />
      </UiField>
    </template>
  </ProductFormSection>

  <template v-if="draftSettings.imagePreprocess.enabled">
    <ProductFormSection class="web-import-preprocess__section">
      <template #title>压缩设置</template>

      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.imagePreprocess.compression.enabled"
          label="启用压缩"
          @change="settingsActions.setImageCompressionEnabled"
        />
      </UiField>

      <template v-if="draftSettings.imagePreprocess.compression.enabled">
        <UiFormGrid>
          <UiField variant="settings" label="质量 (1-100)" control-id="webImportImageQuality">
            <UiNumberField
              input-id="webImportImageQuality"
              :model-value="draftSettings.imagePreprocess.compression.quality"
              :min="1"
              :max="100"
              @update:model-value="
                value => applyNumber(settingsActions.setImageCompressionQuality, value)
              "
            />
          </UiField>
          <UiField variant="settings" label="最大宽度 (0=不限)" control-id="webImportImageMaxWidth">
            <UiNumberField
              input-id="webImportImageMaxWidth"
              :model-value="draftSettings.imagePreprocess.compression.maxWidth"
              :min="0"
              @update:model-value="value => applyNumber(settingsActions.setImageMaxWidth, value)"
            />
          </UiField>
          <UiField
            variant="settings"
            label="最大高度 (0=不限)"
            control-id="webImportImageMaxHeight"
          >
            <UiNumberField
              input-id="webImportImageMaxHeight"
              :model-value="draftSettings.imagePreprocess.compression.maxHeight"
              :min="0"
              @update:model-value="value => applyNumber(settingsActions.setImageMaxHeight, value)"
            />
          </UiField>
        </UiFormGrid>
      </template>
    </ProductFormSection>

    <ProductFormSection class="web-import-preprocess__section">
      <template #title>格式转换</template>

      <UiField variant="settings" control="checkbox">
        <UiCheckbox
          :model-value="draftSettings.imagePreprocess.formatConvert.enabled"
          label="启用格式转换"
          @change="settingsActions.setImageFormatConvertEnabled"
        />
      </UiField>

      <UiField
        v-if="draftSettings.imagePreprocess.formatConvert.enabled"
        variant="settings"
        label="目标格式"
        control-id="webImportImageTargetFormat"
      >
        <UiSelect
          id="webImportImageTargetFormat"
          :model-value="draftSettings.imagePreprocess.formatConvert.targetFormat"
          :options="targetFormatOptions"
          @update:model-value="updateTargetFormat"
        />
      </UiField>
    </ProductFormSection>
  </template>
</template>

<style scoped>
.web-import-preprocess__section {
  --product-form-section-margin-bottom: 16px;
  --product-form-section-title-margin-bottom: 8px;
  --product-form-section-title-padding-bottom: 0;
  --product-form-section-title-border-bottom: 0;
  --product-form-section-title-text: var(--color-text-supporting);
  --product-form-section-title-font-size: 13px;
  --product-form-section-title-font-weight: 500;
}
</style>
