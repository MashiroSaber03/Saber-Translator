<template>
  <div class="text-style-defaults-settings">
    <ProductFormSection>
      <template #title>文本默认值</template>
      <ProductStatusBanner tone="info" role="note">
        这里修改的是全局默认文字设置，会写入 <code class="text-style-defaults-settings__config-path">config/text_style_defaults.json</code>。
        <br />
        保存成功后会在下次启动时作为新的初始默认值使用。
      </ProductStatusBanner>
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
          accept=".ttf,.otf,.woff,.woff2"
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
import { computed, ref, watch, watchEffect } from 'vue'
import type { InpaintMethod, TextAlign, TextDirection } from '@/types/bubble'
import type { TextStyleSettings } from '@/types/settings'
import { getFactoryTextStyleDefaults } from '@/defaults/textStyleFactoryDefaults'
import { normalizeTextStyleSettings } from '@/defaults/textStyleDefaults'
import { listV2Fonts, uploadV2Font, type V2Font } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import {
  BUILTIN_FONTS,
  clampLineSpacing,
  getFontDisplayName,
  inpaintMethodOptions,
  layoutDirectionOptions,
  textAlignOptions,
} from '@/utils/textStyleForm'

interface TextDefaultsSaveResult {
  success: boolean
  changed: boolean
  error?: string
}

const props = withDefaults(defineProps<{
  isOpen: boolean
  saveRequestId?: number
}>(), {
  saveRequestId: 0,
})

const emit = defineEmits<{
  (e: 'save-complete', result: TextDefaultsSaveResult): void
}>()

const toast = useToast()
const settingsStore = useSettingsStore()
const draftDefaults = ref<TextStyleSettings>(getFactoryTextStyleDefaults())
const loadedDefaults = ref<TextStyleSettings | null>(null)
const resetRequested = ref(false)
const userTouched = ref(false)
const isLoading = ref(false)
const errorMessage = ref('')
const fontList = ref<V2Font[]>([])
const fontUploadInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const handledSaveRequestId = ref(0)

const fontSelectOptions = computed(() => {
  const backendOptions = fontList.value.map(font => ({
    label: font.displayName,
    value: font.id,
  }))
  const known = new Set(backendOptions.map(option => option.value))
  const legacyOptions = BUILTIN_FONTS
    .filter(font => !known.has(font))
    .map(font => ({ label: getFontDisplayName(font), value: font }))
  const options = [...backendOptions, ...legacyOptions]
  options.push({ label: '自定义字体...', value: 'custom-font' })
  return options
})

const hasPendingChanges = computed(() => {
  if (resetRequested.value) return true
  if (!loadedDefaults.value) return false
  return JSON.stringify(draftDefaults.value) !== JSON.stringify(loadedDefaults.value)
})

async function loadFontList(): Promise<void> {
  try {
    fontList.value = await listV2Fonts()
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
    const normalized = normalizeTextStyleSettings(settingsStore.settings.textStyle)
    draftDefaults.value = normalized
    loadedDefaults.value = normalized
    resetRequested.value = false
    userTouched.value = false
  } catch (error) {
    loadedDefaults.value = null
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

watchEffect(async () => {
  const requestId = props.saveRequestId
  if (requestId === 0 || requestId === handledSaveRequestId.value) return
  handledSaveRequestId.value = requestId
  emit('save-complete', await saveDefaults())
})

function updateDraft(updates: Partial<TextStyleSettings>): void {
  draftDefaults.value = {
    ...draftDefaults.value,
    ...updates,
  }
  resetRequested.value = false
  userTouched.value = true
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
  draftDefaults.value = getFactoryTextStyleDefaults()
  resetRequested.value = true
  userTouched.value = true
  errorMessage.value = ''
}

async function handleFontUpload(files: File[]): Promise<void> {
  const file = files[0]
  if (!file) return

  const validExtensions = ['.ttf', '.otf', '.woff', '.woff2']
  const fileName = file.name.toLowerCase()
  const isValidType = validExtensions.some(ext => fileName.endsWith(ext))
  if (!isValidType) {
    toast.error('请选择 .ttf、.otf、.woff 或 .woff2 格式的字体文件')
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

async function saveDefaults(): Promise<TextDefaultsSaveResult> {
  if (resetRequested.value) {
    const normalized = normalizeTextStyleSettings(getFactoryTextStyleDefaults())
    settingsStore.updateTextStyle(normalized)
    draftDefaults.value = normalized
    loadedDefaults.value = normalized
    resetRequested.value = false
    userTouched.value = false
    errorMessage.value = ''
    return { success: true, changed: true }
  }

  if (!loadedDefaults.value) {
    if (!userTouched.value) {
      return { success: true, changed: false }
    }
    const error = '请先成功加载当前默认值，或先点击“恢复出厂默认”再保存'
    errorMessage.value = error
    return { success: false, changed: false, error }
  }

  if (!hasPendingChanges.value) {
    return { success: true, changed: false }
  }

  const normalized = normalizeTextStyleSettings(draftDefaults.value)
  settingsStore.updateTextStyle(normalized)
  draftDefaults.value = normalized
  loadedDefaults.value = normalized
  resetRequested.value = false
  userTouched.value = false
  errorMessage.value = ''
  return { success: true, changed: true }
}

</script>

<style scoped>
.text-style-defaults-settings__config-path {
  font-family: var(--font-mono);
}
</style>
