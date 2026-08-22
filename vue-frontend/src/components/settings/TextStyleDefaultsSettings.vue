<template>
  <div class="text-style-defaults-settings">
    <ProductFormSection>
      <template #title>文本默认值</template>
      <p class="text-style-defaults-settings__intro">
        这里修改的是后端数据库中的全局默认文字设置。
        <br />
        保存成功后，新导入页面和后续任务会使用这些默认值。
      </p>
      <ProductActionRow aria-label="文本默认值操作" justify="start">
        <UiButton
          variant="secondary"
          type="button"
          data-testid="reset-text-style-defaults"
          :disabled="isLoading"
          @click="resetDraftToFactory"
        >
          恢复出厂默认
        </UiButton>
      </ProductActionRow>
      <ProductStatusBanner v-if="errorMessage" tone="danger" role="alert">
        {{ errorMessage }}
      </ProductStatusBanner>
    </ProductFormSection>

    <ProductFormSection>
      <template #title>字体排版</template>
      <UiFormGrid>
        <UiField variant="settings" label="字号" control-id="textDefaultsFontSize">
          <UiNumberField
            input-id="textDefaultsFontSize"
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
          control-id="textDefaultsAutoFontSize"
        >
          <UiCheckbox
            input-id="textDefaultsAutoFontSize"
            :model-value="draftDefaults.autoFontSize"
            @change="updateAutoFontSize"
          />
        </UiField>
      </UiFormGrid>

      <UiField variant="settings" label="文本字体" control-id="textDefaultsFontFamily">
        <UiCombobox
          input-id="textDefaultsFontFamily"
          aria-label="文本字体"
          :model-value="draftDefaults.fontFamily"
          :options="fontSelectOptions"
          :disabled="isLoading || isUploadingFont"
          @change="handleFontSelectChange"
        />
        <UiFileInput
          ref="fontUploadInput"
          :accept="FONT_FILE_ACCEPT"
          :disabled="isLoading || isUploadingFont"
          hidden
          @files-change="handleFontUpload"
        />
      </UiField>

      <UiFormGrid>
        <UiField variant="settings" label="排版方向" control-id="textDefaultsLayoutDirection">
          <UiSelect
            id="textDefaultsLayoutDirection"
            :model-value="draftDefaults.layoutDirection"
            :options="layoutDirectionOptions"
            @change="handleLayoutDirectionChange"
          />
        </UiField>
        <UiField variant="settings" label="行内对齐" control-id="textDefaultsInlineAlign">
          <UiSelect
            id="textDefaultsInlineAlign"
            :model-value="draftDefaults.inlineAlign"
            :options="inlineAlignOptions"
            @change="handleInlineAlignChange"
          />
        </UiField>
        <UiField variant="settings" label="文本块对齐" control-id="textDefaultsBlockAlign">
          <UiSelect
            id="textDefaultsBlockAlign"
            :model-value="draftDefaults.blockAlign"
            :options="blockAlignOptions"
            @change="handleBlockAlignChange"
          />
        </UiField>
      </UiFormGrid>

      <UiField
        variant="settings"
        label="行间距"
        control-id="textDefaultsLineSpacing"
        hint="行间距倍数，必须大于 0。"
      >
        <UiNumberField
          input-id="textDefaultsLineSpacing"
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
        control-id="textDefaultsUseAutoTextColor"
      >
        <UiCheckbox
          input-id="textDefaultsUseAutoTextColor"
          :model-value="draftDefaults.useAutoTextColor"
          @change="updateUseAutoTextColor"
        />
      </UiField>
      <UiFormGrid>
        <UiField variant="settings" label="文字颜色" control-id="textDefaultsTextColor">
          <UiColorInput
            input-id="textDefaultsTextColor"
            :model-value="draftDefaults.textColor"
            :disabled="draftDefaults.useAutoTextColor"
            aria-label="文字颜色"
            size="sm"
            @update:model-value="updateTextColor"
          />
        </UiField>
        <UiField variant="settings" label="气泡填充方式" control-id="textDefaultsInpaintMethod">
          <UiSelect
            id="textDefaultsInpaintMethod"
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
        control-id="textDefaultsFillColor"
      >
        <UiColorInput
          input-id="textDefaultsFillColor"
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
        control-id="textDefaultsStrokeEnabled"
      >
        <UiCheckbox
          input-id="textDefaultsStrokeEnabled"
          :model-value="draftDefaults.strokeEnabled"
          @change="updateStrokeEnabled"
        />
      </UiField>
      <UiFormGrid v-if="draftDefaults.strokeEnabled">
        <UiField variant="settings" label="描边颜色" control-id="textDefaultsStrokeColor">
          <UiColorInput
            input-id="textDefaultsStrokeColor"
            :model-value="draftDefaults.strokeColor"
            aria-label="描边颜色"
            @update:model-value="updateStrokeColor"
          />
        </UiField>
        <UiField
          variant="settings"
          label="描边宽度 (px)"
          control-id="textDefaultsStrokeWidth"
          hint="0 表示无描边。"
        >
          <UiNumberField
            input-id="textDefaultsStrokeWidth"
            :model-value="draftDefaults.strokeWidth"
            :min="0"
            size="sm"
            @change="updateStrokeWidth"
          />
        </UiField>
      </UiFormGrid>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { computed, ref } from 'vue'
import type { TextStyleSettings } from '@/types/settings'
import {
  getTextStyleDefaults,
  normalizeTextStyleSettings,
} from '@/defaults/textStyleDefaults'
import { listV2Fonts, uploadV2Font, type V2Font } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import {
  FONT_FILE_ACCEPT,
  FONT_FILE_FORMATS_LABEL,
  isSupportedFontFileName,
} from '@/utils/fontFiles'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import {
  blockAlignOptions,
  inlineAlignOptions,
  inpaintMethodOptions as rawInpaintMethodOptions,
  layoutDirectionOptions,
} from '@/utils/textStyleForm'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'

const toast = useToast()
const settingsStore = useSettingsStore()
const publicAccess = usePublicUserAccess()
const inpaintMethodOptions = computed(() => publicAccess.modelOptions(
  rawInpaintMethodOptions,
  {
    lama_mpe: 'lama_mpe',
    litelama: 'litelama',
  },
))
const draftDefaults = computed(() => settingsStore.textStyleDefaults)
const isLoading = ref(false)
const isUploadingFont = ref(false)
const errorMessage = ref('')
const fontList = computed<V2Font[]>(() => settingsStore.fontCatalog)
const fontUploadInput = ref<InstanceType<typeof UiFileInput> | null>(null)

const fontSelectOptions = computed(() => {
  const options = fontList.value.map(font => ({
    label: font.displayName,
    value: font.id,
  }))
  options.push({ label: '自定义字体...', value: 'custom-font' })
  return options
})

async function loadFontList(): Promise<void> {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const fonts = await listV2Fonts()
    settingsStore.hydrateResourceCatalogs(fonts, settingsStore.promptCatalog)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '获取字体列表失败'
  } finally {
    isLoading.value = false
  }
}

void loadFontList()

function updateDraft(updates: Partial<TextStyleSettings>): void {
  const normalized = normalizeTextStyleSettings({
    ...draftDefaults.value,
    ...updates,
  })
  settingsStore.textStyleDefaults = normalized
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
  if (value !== null && Number.isInteger(value) && value >= 0) {
    updateDraft({ strokeWidth: value })
  }
}

function resetDraftToFactory(): void {
  settingsStore.textStyleDefaults = getTextStyleDefaults()
  errorMessage.value = ''
}

async function handleFontUpload(files: File[]): Promise<void> {
  if (isUploadingFont.value) return
  const file = files[0]
  if (!file) return

  if (!isSupportedFontFileName(file.name)) {
    toast.error(`请选择 ${FONT_FILE_FORMATS_LABEL} 格式的字体文件`)
    fontUploadInput.value?.clear()
    return
  }

  isUploadingFont.value = true
  try {
    const uploadedFont = await uploadV2Font(file)
    settingsStore.upsertFont(uploadedFont)
    updateDraft({ fontFamily: uploadedFont.id })
    toast.success('字体上传成功')
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '字体上传失败')
  } finally {
    isUploadingFont.value = false
    fontUploadInput.value?.clear()
  }
}

function handleFontSelectChange(value: string | number): void {
  if (typeof value !== 'string') return
  if (value === 'custom-font') {
    if (isUploadingFont.value) return
    fontUploadInput.value?.click()
    return
  }
  if (value) {
    updateDraft({ fontFamily: value })
  }
}

</script>

<style scoped>
.text-style-defaults-settings__intro {
  margin: 0 0 14px;
  color: var(--color-text-supporting);
  font-size: 13px;
  line-height: 1.6;
}
</style>
