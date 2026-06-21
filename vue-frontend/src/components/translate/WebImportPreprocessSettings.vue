<script setup lang="ts">
import UiPanel from '@/components/ui/UiPanel.vue'
import UiInput from '@/components/ui/UiInput.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useWebImportStore } from '@/stores/webImportStore'
import type { WebImportSettings } from '@/types/webImport'

defineProps<{
  draftSettings: WebImportSettings
}>()

const webImportStore = useWebImportStore()
const targetFormatOptions = [
  { label: '保持原格式', value: 'original' },
  { label: 'JPEG', value: 'jpeg' },
  { label: 'PNG', value: 'png' },
  { label: 'WebP', value: 'webp' },
] as const
</script>

<template>
  <UiPanel variant="settings">
    <div class="web-import-modal__form-row">
      <label class="ui-checkbox-label">
        <UiInput
          type="checkbox"
          :checked="draftSettings.imagePreprocess.enabled"
          @change="webImportStore.setImagePreprocessEnabled(($event.target as HTMLInputElement).checked)"
        />
        启用图片预处理
      </label>
    </div>

    <template v-if="draftSettings.imagePreprocess.enabled">
      <div class="web-import-modal__form-row">
        <label class="ui-checkbox-label">
          <UiInput
            type="checkbox"
            :checked="draftSettings.imagePreprocess.autoRotate"
            @change="webImportStore.setImageAutoRotate(($event.target as HTMLInputElement).checked)"
          />
          根据 EXIF 自动旋转
        </label>
      </div>

      <h5 class="web-import-modal__subsection-title">压缩设置</h5>
      <div class="web-import-modal__form-row">
        <label class="ui-checkbox-label">
          <UiInput
            type="checkbox"
            :checked="draftSettings.imagePreprocess.compression.enabled"
            @change="webImportStore.setImageCompressionEnabled(($event.target as HTMLInputElement).checked)"
          />
          启用压缩
        </label>
      </div>

      <template v-if="draftSettings.imagePreprocess.compression.enabled">
        <div class="web-import-modal__form-grid">
          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">质量 (0-100)</label>
            <UiInput
              type="number"
              class="web-import-modal__form-input web-import-modal__form-input--small"
              :value="draftSettings.imagePreprocess.compression.quality"
              @input="webImportStore.setImageCompressionQuality(Number(($event.target as HTMLInputElement).value))"
              min="1"
              max="100"
            />
          </div>
          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">最大宽度 (0=不限)</label>
            <UiInput
              type="number"
              class="web-import-modal__form-input web-import-modal__form-input--small"
              :value="draftSettings.imagePreprocess.compression.maxWidth"
              @input="webImportStore.setImageMaxWidth(Number(($event.target as HTMLInputElement).value))"
              min="0"
            />
          </div>
          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">最大高度 (0=不限)</label>
            <UiInput
              type="number"
              class="web-import-modal__form-input web-import-modal__form-input--small"
              :value="draftSettings.imagePreprocess.compression.maxHeight"
              @input="webImportStore.setImageMaxHeight(Number(($event.target as HTMLInputElement).value))"
              min="0"
            />
          </div>
        </div>
      </template>

      <h5 class="web-import-modal__subsection-title">格式转换</h5>
      <div class="web-import-modal__form-row">
        <label class="ui-checkbox-label">
          <UiInput
            type="checkbox"
            :checked="draftSettings.imagePreprocess.formatConvert.enabled"
            @change="webImportStore.setImageFormatConvertEnabled(($event.target as HTMLInputElement).checked)"
          />
          启用格式转换
        </label>
      </div>

      <div v-if="draftSettings.imagePreprocess.formatConvert.enabled" class="web-import-modal__form-row">
        <label class="web-import-modal__form-label">目标格式</label>
        <CustomSelect
          :model-value="draftSettings.imagePreprocess.formatConvert.targetFormat"
          :options="targetFormatOptions"
          @change="(value) => webImportStore.setImageTargetFormat(String(value) as 'jpeg' | 'png' | 'webp' | 'original')"
        />
      </div>
    </template>
  </UiPanel>
</template>
