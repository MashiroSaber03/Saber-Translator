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
            :min="10"
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
          @change="handleFontSelectChange"
        />
        <UiFileInput
          ref="fontUploadInput"
          :accept="FONT_FILE_ACCEPT"
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
        <UiField variant="settings" label="对齐方式" control-id="textDefaultsTextAlign">
          <UiSelect
            id="textDefaultsTextAlign"
            :model-value="draftDefaults.textAlign"
            :options="textAlignOptions"
            @change="handleTextAlignChange"
          />
        </UiField>
      </UiFormGrid>

      <UiField
        variant="settings"
        label="行间距"
        control-id="textDefaultsLineSpacing"
        hint="行间距倍数（0.5 - 3.0）"
      >
        <UiNumberField
          input-id="textDefaultsLineSpacing"
          :model-value="draftDefaults.lineSpacing"
          :min="0.5"
          :max="3"
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
            :max="10"
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
import { computed, ref, watch } from 'vue'
import type { InpaintMethod, TextAlign, TextDirection } from '@/types/bubble'
import type { TextStyleSettings } from '@/types/settings'
import { getFactoryTextStyleDefaults } from '@/defaults/textStyleFactoryDefaults'
import { normalizeTextStyleSettings } from '@/defaults/textStyleDefaults'
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
  clampLineSpacing,
  getFontDisplayName,
  inpaintMethodOptions,
  layoutDirectionOptions,
  textAlignOptions,
} from '@/utils/textStyleForm'

const props = defineProps<{
  isOpen: boolean
}>()

const toast = useToast()
const settingsStore = useSettingsStore()
const draftDefaults = ref<TextStyleSettings>(getFactoryTextStyleDefaults())
const isLoading = ref(false)
const errorMessage = ref('')
const fontList = ref<V2Font[]>([])
const fontUploadInput = ref<InstanceType<typeof UiFileInput> | null>(null)

const fontSelectOptions = computed(() => {
  const backendOptions = fontList.value.map(font => ({
    label: font.displayName,
    value: font.id,
  }))
  const known = new Set(backendOptions.map(option => option.value))
  const options = [...backendOptions]
  const currentFont = draftDefaults.value.fontFamily
  if (currentFont && !known.has(currentFont)) {
    options.unshift({
      label: getFontDisplayName(currentFont),
      value: currentFont,
    })
  }
  options.push({ label: '自定义字体...', value: 'custom-font' })
  return options
})

async function loadFontList(): Promise<void> {
  try {
    fontList.value = await listV2Fonts()
    settingsStore.hydrateResourceCatalogs(
      fontList.value,
      settingsStore.promptCatalog,
    )
  } catch {
    fontList.value = []
  }
}

async function loadDefaults(): Promise<void> {
  isLoading.value = true
  errorMessage.value = ''
  try {
    await Promise.all([
      settingsStore.isBackendReady ? Promise.resolve(true) : settingsStore.loadFromBackend(),
      loadFontList(),
    ])
    if (!settingsStore.isBackendReady) {
      throw new Error(settingsStore.backendError || '获取文本默认值失败')
    }
    const normalized = normalizeTextStyleSettings(settingsStore.textStyleDefaults)
    draftDefaults.value = normalized
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '获取文本默认值失败'
  } finally {
    isLoading.value = false
  }
}

watch(
  () => props.isOpen,
  (isOpen) => {
    if (isOpen) {
      void loadDefaults()
    }
  },
  { immediate: true }
)

function updateDraft(updates: Partial<TextStyleSettings>): void {
  const normalized = normalizeTextStyleSettings({
    ...draftDefaults.value,
    ...updates,
  })
  draftDefaults.value = normalized
  settingsStore.textStyleDefaults = normalized
}

function updateFontSize(value: number | null): void {
  if (value !== null && value > 0) {
    updateDraft({ fontSize: value })
  }
}

function updateAutoFontSize(value: boolean): void {
  updateDraft({ autoFontSize: value })
}

function handleLayoutDirectionChange(value: string | number): void {
  updateDraft({ layoutDirection: String(value) as TextDirection })
}

function handleTextAlignChange(value: string | number): void {
  updateDraft({ textAlign: String(value) as TextAlign })
}

function handleInpaintMethodChange(value: string | number): void {
  updateDraft({ inpaintMethod: String(value) as InpaintMethod })
}

function updateLineSpacing(value: number | null): void {
  const lineSpacing = value === null
    ? draftDefaults.value.lineSpacing
    : clampLineSpacing(value, draftDefaults.value.lineSpacing)
  updateDraft({ lineSpacing })
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
  if (value !== null && value >= 0) {
    updateDraft({ strokeWidth: value })
  }
}

function resetDraftToFactory(): void {
  const normalized = normalizeTextStyleSettings(getFactoryTextStyleDefaults())
  draftDefaults.value = normalized
  settingsStore.textStyleDefaults = normalized
  errorMessage.value = ''
}

async function handleFontUpload(files: File[]): Promise<void> {
  const file = files[0]
  if (!file) return

  if (!isSupportedFontFileName(file.name)) {
    toast.error(`请选择 ${FONT_FILE_FORMATS_LABEL} 格式的字体文件`)
    fontUploadInput.value?.clear()
    return
  }

  try {
    const response = await uploadV2Font(file)
    await loadFontList()
    updateDraft({ fontFamily: response.id })
    toast.success('字体上传成功')
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '字体上传失败')
  } finally {
    fontUploadInput.value?.clear()
  }
}

function handleFontSelectChange(value: string | number): void {
  const nextValue = String(value)
  if (nextValue === 'custom-font') {
    fontUploadInput.value?.click()
    return
  }
  updateDraft({ fontFamily: nextValue })
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
