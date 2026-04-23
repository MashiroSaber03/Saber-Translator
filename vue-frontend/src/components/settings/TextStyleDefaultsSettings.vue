<template>
  <div class="text-style-defaults-settings">
    <div class="settings-group">
      <div class="settings-group-title">文本默认值</div>
      <div class="settings-item">
        <div class="input-hint">
          这里修改的是全局默认文字设置，会写入 <code>config/text_style_defaults.json</code>。
          <br />
          保存成功后会在下次启动时作为新的初始默认值使用。
        </div>
      </div>
      <div class="settings-item action-row">
        <button
          type="button"
          class="btn btn-secondary"
          data-testid="reset-text-style-defaults"
          :disabled="isLoading"
          @click="resetDraftToFactory"
        >
          恢复出厂默认
        </button>
      </div>
      <div v-if="errorMessage" class="settings-item">
        <div class="input-hint error-hint">{{ errorMessage }}</div>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">字体排版</div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="textDefaultsFontSize">字号</label>
          <input
            id="textDefaultsFontSize"
            type="number"
            :value="draftDefaults.fontSize"
            min="10"
            :disabled="draftDefaults.autoFontSize"
            @input="updateFontSize"
          />
        </div>
        <div class="settings-item checkbox-item">
          <label class="checkbox-label">
            <input
              type="checkbox"
              :checked="draftDefaults.autoFontSize"
              @change="updateAutoFontSize"
            />
            <span class="checkbox-text">自动计算初始字号</span>
          </label>
        </div>
      </div>

      <div class="settings-item">
        <label for="textDefaultsFontFamily">文本字体</label>
        <CustomSelect
          :model-value="draftDefaults.fontFamily"
          :options="fontSelectOptions"
          @change="handleFontSelectChange"
        />
        <input
          ref="fontUploadInput"
          type="file"
          accept=".ttf,.ttc,.otf"
          style="display: none"
          @change="handleFontUpload"
        />
      </div>

      <div class="settings-row">
        <div class="settings-item">
          <label for="textDefaultsLayoutDirection">排版方向</label>
          <CustomSelect
            :model-value="draftDefaults.layoutDirection"
            :options="layoutDirectionOptions"
            @change="handleLayoutDirectionChange"
          />
        </div>
        <div class="settings-item">
          <label for="textDefaultsTextAlign">对齐方式</label>
          <CustomSelect
            :model-value="draftDefaults.textAlign"
            :options="textAlignOptions"
            @change="handleTextAlignChange"
          />
        </div>
      </div>

      <div class="settings-item">
        <label for="textDefaultsLineSpacing">行间距</label>
        <input
          id="textDefaultsLineSpacing"
          type="number"
          :value="draftDefaults.lineSpacing"
          min="0.5"
          max="3"
          step="0.1"
          @change="updateLineSpacing"
        />
        <div class="input-hint">行间距倍数（0.5 - 3.0）</div>
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">颜色与填充</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input
            type="checkbox"
            :checked="draftDefaults.useAutoTextColor"
            @change="updateUseAutoTextColor"
          />
          <span class="checkbox-text">自动识别文字颜色</span>
        </label>
      </div>
      <div class="settings-row">
        <div class="settings-item">
          <label for="textDefaultsTextColor">文字颜色</label>
          <input
            id="textDefaultsTextColor"
            type="color"
            :value="draftDefaults.textColor"
            :disabled="draftDefaults.useAutoTextColor"
            @input="updateTextColor"
          />
        </div>
        <div class="settings-item">
          <label for="textDefaultsInpaintMethod">气泡填充方式</label>
          <CustomSelect
            :model-value="draftDefaults.inpaintMethod"
            :options="inpaintMethodOptions"
            @change="handleInpaintMethodChange"
          />
        </div>
      </div>
      <div v-if="draftDefaults.inpaintMethod === 'solid'" class="settings-item">
        <label for="textDefaultsFillColor">填充颜色</label>
        <input
          id="textDefaultsFillColor"
          type="color"
          :value="draftDefaults.fillColor"
          @input="updateFillColor"
        />
      </div>
    </div>

    <div class="settings-group">
      <div class="settings-group-title">描边</div>
      <div class="settings-item checkbox-item">
        <label class="checkbox-label">
          <input
            type="checkbox"
            :checked="draftDefaults.strokeEnabled"
            @change="updateStrokeEnabled"
          />
          <span class="checkbox-text">启用描边</span>
        </label>
      </div>
      <div v-if="draftDefaults.strokeEnabled" class="settings-row">
        <div class="settings-item">
          <label for="textDefaultsStrokeColor">描边颜色</label>
          <input
            id="textDefaultsStrokeColor"
            type="color"
            :value="draftDefaults.strokeColor"
            @input="updateStrokeColor"
          />
        </div>
        <div class="settings-item">
          <label for="textDefaultsStrokeWidth">描边宽度 (px)</label>
          <input
            id="textDefaultsStrokeWidth"
            type="number"
            :value="draftDefaults.strokeWidth"
            min="0"
            max="10"
            @input="updateStrokeWidth"
          />
          <div class="input-hint">0 表示无描边。</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
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
    if (fonts.length > 0 && typeof fonts[0] === 'object') {
      fontList.value = fonts.map(font => (typeof font === 'object' ? font.path : String(font)))
    } else {
      fontList.value = fonts as string[]
    }
  } catch (error) {
    console.warn('[TextStyleDefaultsSettings] 加载字体列表失败:', error)
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

.error-hint {
  color: var(--color-danger, #d14343);
}

.text-style-defaults-settings code {
  font-family: Consolas, 'Courier New', monospace;
}
</style>
