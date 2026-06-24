<template>
  <div class="text-style-defaults-settings">
    <UiPanel variant="settings">
      <template #title>文本默认值</template>
      <UiField class="ui-settings-field">
        <div class="ui-form-hint">
          这里修改的是全局默认文字设置，会写入 <code>config/text_style_defaults.json</code>。
          <br />
          保存成功后会在下次启动时作为新的初始默认值使用。
        </div>
      </UiField>
      <UiField class="ui-settings-field action-row">
        <UiButton
          variant="secondary"
          type="button"
          data-testid="reset-text-style-defaults"
          :disabled="isLoading"
          @click="resetDraftToFactory"
        >
          恢复出厂默认
        </UiButton>
      </UiField>
      <UiField v-if="errorMessage" class="ui-settings-field">
        <div class="ui-form-hint ui-form-hint--error">{{ errorMessage }}</div>
      </UiField>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>字体排版</template>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="textDefaultsFontSize">字号</label>
          <UiInput
            id="textDefaultsFontSize"
            type="number"
            :value="draftDefaults.fontSize"
            min="10"
            :disabled="draftDefaults.autoFontSize"
            @input="updateFontSize"
          />
        </UiField>
        <UiField class="ui-settings-field ui-settings-field--checkbox">
          <label class="ui-checkbox-label">
            <UiInput
              type="checkbox"
              :checked="draftDefaults.autoFontSize"
              @change="updateAutoFontSize"
            />
            <span class="checkbox-text">自动计算初始字号</span>
          </label>
        </UiField>
      </UiFormGrid>

      <UiField class="ui-settings-field">
        <label for="textDefaultsFontFamily">文本字体</label>
        <CustomSelect
          :model-value="draftDefaults.fontFamily"
          :options="fontSelectOptions"
          @change="handleFontSelectChange"
        />
        <UiFileInput
          ref="fontUploadInput"
          accept=".ttf,.ttc,.otf"
          style="display: none"
          @change="handleFontUpload"
        />
      </UiField>

      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="textDefaultsLayoutDirection">排版方向</label>
          <CustomSelect
            :model-value="draftDefaults.layoutDirection"
            :options="layoutDirectionOptions"
            @change="handleLayoutDirectionChange"
          />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="textDefaultsTextAlign">对齐方式</label>
          <CustomSelect
            :model-value="draftDefaults.textAlign"
            :options="textAlignOptions"
            @change="handleTextAlignChange"
          />
        </UiField>
      </UiFormGrid>

      <UiField class="ui-settings-field">
        <label for="textDefaultsLineSpacing">行间距</label>
        <UiInput
          id="textDefaultsLineSpacing"
          type="number"
          :value="draftDefaults.lineSpacing"
          min="0.5"
          max="3"
          step="0.1"
          @change="updateLineSpacing"
        />
        <div class="ui-form-hint">行间距倍数（0.5 - 3.0）</div>
      </UiField>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>颜色与填充</template>
      <UiField class="ui-settings-field ui-settings-field--checkbox">
        <label class="ui-checkbox-label">
          <UiInput
            type="checkbox"
            :checked="draftDefaults.useAutoTextColor"
            @change="updateUseAutoTextColor"
          />
          <span class="checkbox-text">自动识别文字颜色</span>
        </label>
      </UiField>
      <UiFormGrid>
        <UiField class="ui-settings-field">
          <label for="textDefaultsTextColor">文字颜色</label>
          <UiInput
            id="textDefaultsTextColor"
            type="color"
            :value="draftDefaults.textColor"
            :disabled="draftDefaults.useAutoTextColor"
            @input="updateTextColor"
          />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="textDefaultsInpaintMethod">气泡填充方式</label>
          <CustomSelect
            :model-value="draftDefaults.inpaintMethod"
            :options="inpaintMethodOptions"
            @change="handleInpaintMethodChange"
          />
        </UiField>
      </UiFormGrid>
      <UiField v-if="draftDefaults.inpaintMethod === 'solid'" class="ui-settings-field">
        <label for="textDefaultsFillColor">填充颜色</label>
        <UiInput
          id="textDefaultsFillColor"
          type="color"
          :value="draftDefaults.fillColor"
          @input="updateFillColor"
        />
      </UiField>
    </UiPanel>

    <UiPanel variant="settings">
      <template #title>描边</template>
      <UiField class="ui-settings-field ui-settings-field--checkbox">
        <label class="ui-checkbox-label">
          <UiInput
            type="checkbox"
            :checked="draftDefaults.strokeEnabled"
            @change="updateStrokeEnabled"
          />
          <span class="checkbox-text">启用描边</span>
        </label>
      </UiField>
      <UiFormGrid v-if="draftDefaults.strokeEnabled">
        <UiField class="ui-settings-field">
          <label for="textDefaultsStrokeColor">描边颜色</label>
          <UiInput
            id="textDefaultsStrokeColor"
            type="color"
            :value="draftDefaults.strokeColor"
            @input="updateStrokeColor"
          />
        </UiField>
        <UiField class="ui-settings-field">
          <label for="textDefaultsStrokeWidth">描边宽度 (px)</label>
          <UiInput
            id="textDefaultsStrokeWidth"
            type="number"
            :value="draftDefaults.strokeWidth"
            min="0"
            max="10"
            @input="updateStrokeWidth"
          />
          <div class="ui-form-hint">0 表示无描边。</div>
        </UiField>
      </UiFormGrid>
    </UiPanel>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { computed, ref, watch } from 'vue'
import type { InpaintMethod, TextAlign, TextDirection } from '@/types/bubble'
import type { TextStyleSettings } from '@/types/settings'
import { getFactoryTextStyleDefaults } from '@/defaults/textStyleFactoryDefaults'
import { normalizeTextStyleSettings } from '@/defaults/textStyleDefaults'
import { configApi } from '@/api/config'
import { useToast } from '@/utils/toast'
import CustomSelect from '@/components/common/CustomSelect.vue'
import {
  BUILTIN_FONTS,
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
const draftDefaults = ref<TextStyleSettings>(getFactoryTextStyleDefaults())
const loadedDefaults = ref<TextStyleSettings | null>(null)
const resetRequested = ref(false)
const userTouched = ref(false)
const isLoading = ref(false)
const errorMessage = ref('')
const fontList = ref<string[]>([])
const fontUploadInput = ref<HTMLInputElement | null>(null)

const fontSelectOptions = computed(() => {
  const options = Array.from(new Set([...BUILTIN_FONTS, ...fontList.value])).map(font => ({
    label: getFontDisplayName(font),
    value: font,
  }))
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
    const response = await configApi.getFontList()
    const fonts = response.fonts || []
    fontList.value = fonts.map(font => font.path)
  } catch {
    fontList.value = [...BUILTIN_FONTS]
  }
}

async function loadDefaults(): Promise<void> {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const [defaultsResponse] = await Promise.all([
      configApi.getTextStyleDefaults(),
      loadFontList(),
    ])

    if (!defaultsResponse.success || !defaultsResponse.defaults) {
      throw new Error(defaultsResponse.error || '获取文本默认值失败')
    }

    const normalized = normalizeTextStyleSettings(defaultsResponse.defaults)
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

function updateDraft(updates: Partial<TextStyleSettings>): void {
  draftDefaults.value = {
    ...draftDefaults.value,
    ...updates,
  }
  resetRequested.value = false
  userTouched.value = true
}

function updateFontSize(event: Event): void {
  const value = parseInt((event.target as HTMLInputElement).value, 10)
  if (!Number.isNaN(value) && value > 0) {
    updateDraft({ fontSize: value })
  }
}

function updateAutoFontSize(event: Event): void {
  updateDraft({ autoFontSize: (event.target as HTMLInputElement).checked })
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

function updateLineSpacing(event: Event): void {
  const value = clampLineSpacing(Number((event.target as HTMLInputElement).value), draftDefaults.value.lineSpacing)
  updateDraft({ lineSpacing: value })
}

function updateTextColor(event: Event): void {
  updateDraft({ textColor: (event.target as HTMLInputElement).value })
}

function updateUseAutoTextColor(event: Event): void {
  updateDraft({ useAutoTextColor: (event.target as HTMLInputElement).checked })
}

function updateFillColor(event: Event): void {
  updateDraft({ fillColor: (event.target as HTMLInputElement).value })
}

function updateStrokeEnabled(event: Event): void {
  updateDraft({ strokeEnabled: (event.target as HTMLInputElement).checked })
}

function updateStrokeColor(event: Event): void {
  updateDraft({ strokeColor: (event.target as HTMLInputElement).value })
}

function updateStrokeWidth(event: Event): void {
  const value = parseInt((event.target as HTMLInputElement).value, 10)
  if (!Number.isNaN(value) && value >= 0) {
    updateDraft({ strokeWidth: value })
  }
}

function resetDraftToFactory(): void {
  draftDefaults.value = getFactoryTextStyleDefaults()
  resetRequested.value = true
  userTouched.value = true
  errorMessage.value = ''
}

async function handleFontUpload(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  const validExtensions = ['.ttf', '.ttc', '.otf']
  const fileName = file.name.toLowerCase()
  const isValidType = validExtensions.some(ext => fileName.endsWith(ext))
  if (!isValidType) {
    toast.error('请选择 .ttf、.ttc 或 .otf 格式的字体文件')
    input.value = ''
    return
  }

  try {
    const response = await configApi.uploadFont(file)
    if (response.success && response.fontPath) {
      await loadFontList()
      updateDraft({ fontFamily: response.fontPath })
      toast.success('字体上传成功')
    } else {
      toast.error(response.error || '字体上传失败')
    }
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '字体上传失败')
  } finally {
    input.value = ''
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

async function saveDefaults(): Promise<{ success: boolean; changed: boolean; error?: string }> {
  if (resetRequested.value) {
    try {
      const response = await configApi.resetTextStyleDefaults()
      if (!response.success || !response.defaults) {
        const error = response.error || '重置文本默认值失败'
        errorMessage.value = error
        return { success: false, changed: false, error }
      }

      const normalized = normalizeTextStyleSettings(response.defaults)
      draftDefaults.value = normalized
      loadedDefaults.value = normalized
      resetRequested.value = false
      userTouched.value = false
      errorMessage.value = ''
      return { success: true, changed: true }
    } catch (error) {
      const message = error instanceof Error ? error.message : '重置文本默认值失败'
      errorMessage.value = message
      return { success: false, changed: false, error: message }
    }
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

  try {
    const response = await configApi.saveTextStyleDefaults(draftDefaults.value)
    if (!response.success || !response.defaults) {
      const error = response.error || '保存文本默认值失败'
      errorMessage.value = error
      return { success: false, changed: false, error }
    }

    const normalized = normalizeTextStyleSettings(response.defaults)
    draftDefaults.value = normalized
    loadedDefaults.value = normalized
    resetRequested.value = false
    userTouched.value = false
    errorMessage.value = ''
    return { success: true, changed: true }
  } catch (error) {
    const message = error instanceof Error ? error.message : '保存文本默认值失败'
    errorMessage.value = message
    return { success: false, changed: false, error: message }
  }
}

defineExpose({
  saveDefaults,
})
</script>

<style scoped>
.action-row {
  display: flex;
  justify-content: flex-start;
}

.text-style-defaults-settings code {
  font-family: Consolas, 'Courier New', monospace;
}
</style>
