<template>
  <!-- 文字样式配置组件 -->
  <div class="text-style-settings">
    <!-- 字号设置 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">字号</label>
        <div class="setting-control font-size-control">
          <!-- 自动/手动切换 -->
          <label class="auto-toggle">
            <input
              type="checkbox"
              v-model="autoFontSize"
              @change="handleAutoFontSizeChange"
            />
            <span>自动</span>
          </label>
          
          <!-- 字号滑块 -->
          <input
            type="range"
            v-model.number="fontSize"
            :min="FONT_SIZE_MIN"
            :max="FONT_SIZE_MAX"
            :step="FONT_SIZE_STEP"
            :disabled="autoFontSize"
            @input="handleFontSizeChange"
            class="font-size-slider"
          />
          
          <!-- 字号数值输入 -->
          <input
            type="number"
            v-model.number="fontSize"
            :min="FONT_SIZE_MIN"
            :max="FONT_SIZE_MAX"
            :disabled="autoFontSize"
            @change="handleFontSizeChange"
            class="font-size-input"
          />
        </div>
      </div>
      
      <!-- 字号预设 -->
      <div class="setting-row presets-row">
        <div class="font-presets">
          <button
            v-for="preset in allFontPresets"
            :key="preset"
            class="preset-btn"
            :class="{ active: fontSize === preset }"
            :disabled="autoFontSize"
            @click="setFontSize(preset)"
          >
            {{ preset }}
          </button>
          <!-- 添加自定义预设按钮 -->
          <button
            class="preset-btn add-preset-btn"
            :disabled="autoFontSize"
            @click="showAddPresetDialog"
            title="添加自定义预设"
          >
            +
          </button>
        </div>
      </div>
    </div>

    <!-- 字体选择 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">字体</label>
        <div class="setting-control">
          <CustomSelect
            v-model="fontFamily"
            :options="fontSelectOptions"
            @change="handleFontFamilySelectChange"
          />
          <!-- 上传自定义字体按钮 -->
          <button class="upload-font-btn" @click="triggerFontUpload" title="上传自定义字体">
            📁
          </button>
          <input
            ref="fontUploadInput"
            type="file"
            accept=".ttf,.ttc,.otf"
            style="display: none"
            @change="handleFontUpload"
          />
        </div>
      </div>
    </div>

    <!-- 排版方向 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">排版</label>
        <div class="setting-control">
          <CustomSelect
            v-model="layoutDirection"
            :options="layoutDirectionOptions"
            @change="handleLayoutDirectionSelectChange"
          />
        </div>
      </div>
    </div>

    <!-- 文字颜色 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">文字颜色</label>
        <div class="setting-control color-control">
          <input
            type="color"
            v-model="textColor"
            @input="handleTextColorChange"
            class="color-picker"
          />
          <input
            type="text"
            v-model="textColor"
            @change="handleTextColorChange"
            class="color-input"
            maxlength="7"
          />
        </div>
      </div>
    </div>

    <!-- 填充颜色 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">填充颜色</label>
        <div class="setting-control color-control">
          <input
            type="color"
            v-model="fillColor"
            @input="handleFillColorChange"
            class="color-picker"
          />
          <input
            type="text"
            v-model="fillColor"
            @change="handleFillColorChange"
            class="color-input"
            maxlength="7"
          />
        </div>
      </div>
    </div>

    <!-- 描边设置 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">描边</label>
        <div class="setting-control stroke-control">
          <label class="stroke-toggle">
            <input
              type="checkbox"
              v-model="strokeEnabled"
              @change="handleStrokeEnabledChange"
            />
            <span>启用</span>
          </label>
        </div>
      </div>
      
      <!-- 描边详细设置（启用时显示） -->
      <div v-if="strokeEnabled" class="stroke-options">
        <div class="setting-row">
          <label class="setting-label sub-label">颜色</label>
          <div class="setting-control color-control">
            <input
              type="color"
              v-model="strokeColor"
              @input="handleStrokeColorChange"
              class="color-picker"
            />
            <input
              type="text"
              v-model="strokeColor"
              @change="handleStrokeColorChange"
              class="color-input"
              maxlength="7"
            />
          </div>
        </div>
        <div class="setting-row">
          <label class="setting-label sub-label">宽度</label>
          <div class="setting-control">
            <input
              type="range"
              v-model.number="strokeWidth"
              min="1"
              max="10"
              step="1"
              @input="handleStrokeWidthChange"
              class="stroke-width-slider"
            />
            <span class="stroke-width-value">{{ strokeWidth }}px</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 修复方式 -->
    <div class="setting-group">
      <div class="setting-row">
        <label class="setting-label">修复方式</label>
        <div class="setting-control">
          <CustomSelect
            v-model="inpaintMethod"
            :options="inpaintMethodOptions"
            @change="handleInpaintMethodSelectChange"
          />
        </div>
      </div>
    </div>

    <!-- 应用到全部按钮 -->
    <div class="setting-group apply-all-group">
      <div class="apply-all-row">
        <button class="apply-all-btn" @click="showApplyOptions = !showApplyOptions">
          应用到全部
          <span class="dropdown-icon">{{ showApplyOptions ? '▲' : '▼' }}</span>
        </button>
        <button
          class="apply-options-btn"
          @click="showApplyOptions = !showApplyOptions"
          title="选择要应用的参数"
        >
          ⚙️
        </button>
      </div>
      
      <!-- 应用选项下拉面板 -->
      <div v-if="showApplyOptions" class="apply-options-panel">
        <label class="apply-option">
          <input type="checkbox" v-model="applyOptions.fontSize" />
          <span>字号</span>
        </label>
        <label class="apply-option">
          <input type="checkbox" v-model="applyOptions.fontFamily" />
          <span>字体</span>
        </label>
        <label class="apply-option">
          <input type="checkbox" v-model="applyOptions.layoutDirection" />
          <span>排版方向</span>
        </label>
        <label class="apply-option">
          <input type="checkbox" v-model="applyOptions.textColor" />
          <span>文字颜色</span>
        </label>
        <label class="apply-option">
          <input type="checkbox" v-model="applyOptions.fillColor" />
          <span>填充颜色</span>
        </label>
        <label class="apply-option">
          <input type="checkbox" v-model="applyOptions.stroke" />
          <span>描边设置</span>
        </label>
        <div class="apply-actions">
          <button class="select-all-btn" @click="selectAllOptions">全选</button>
          <button class="apply-confirm-btn" @click="applyToAll">确认应用</button>
        </div>
      </div>
    </div>
  </div>
</template>


<script setup lang="ts">
/**
 * 文字样式配置组件
 * 提供字体、字号、颜色、描边、修复方式等设置
 */

import { ref, computed, onMounted } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import { useImageStore } from '@/stores/imageStore'
import { useToast } from '@/utils/toast'
import { getFontList, uploadFont } from '@/api/config'
import { applySettingsToAllImages } from '@/api/translate'
import {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_MAX,
  FONT_SIZE_STEP
} from '@/constants'
import type { TextDirection, InpaintMethod } from '@/types/bubble'
import CustomSelect from '@/components/common/CustomSelect.vue'

// ============================================================
// Store 和工具
// ============================================================

const settingsStore = useSettingsStore()
const imageStore = useImageStore()
const toast = useToast()

// ============================================================
// 状态
// ============================================================

/** 字体列表 */
const fontList = ref<(string | import('@/types').FontInfo)[]>([])

/** 字体上传输入框引用 */
const fontUploadInput = ref<HTMLInputElement | null>(null)

/** 是否显示应用选项面板 */
const showApplyOptions = ref(false)

/** 应用选项 */
const applyOptions = ref({
  fontSize: true,
  fontFamily: true,
  layoutDirection: true,
  textColor: true,
  fillColor: true,
  stroke: true
})

/** 排版方向选项（用于CustomSelect） */
const layoutDirectionOptions = [
  { label: '自动', value: 'auto' },
  { label: '垂直', value: 'vertical' },
  { label: '水平', value: 'horizontal' }
]

/** 修复方式选项（用于CustomSelect） */
const inpaintMethodOptions = [
  { label: '纯色填充', value: 'solid' },
  { label: 'LAMA MPE', value: 'lama_mpe' },
  { label: 'LiteLAMA', value: 'litelama' }
]

/** 字体选择选项（用于CustomSelect） */
const fontSelectOptions = computed(() => {
  return fontList.value.map(font => {
    // 兼容 FontInfo 对象和字符串两种格式
    if (typeof font === 'string') {
      return {
        label: getFontDisplayName(font),
        value: font
      }
    } else {
      return {
        label: font.display_name || font.file_name,
        value: font.path || font.file_name
      }
    }
  })
})

// ============================================================
// 计算属性 - 双向绑定设置
// ============================================================

/** 字号 */
const fontSize = computed({
  get: () => settingsStore.settings.textStyle.fontSize,
  set: (value: number) => settingsStore.updateTextStyle({ fontSize: value })
})

/** 自动字号 */
const autoFontSize = computed({
  get: () => settingsStore.settings.textStyle.autoFontSize,
  set: (value: boolean) => settingsStore.updateTextStyle({ autoFontSize: value })
})

/** 字体 */
const fontFamily = computed({
  get: () => settingsStore.settings.textStyle.fontFamily,
  set: (value: string) => settingsStore.updateTextStyle({ fontFamily: value })
})

/** 排版方向 */
const layoutDirection = computed({
  get: () => settingsStore.settings.textStyle.layoutDirection,
  set: (value: TextDirection) => settingsStore.updateTextStyle({ layoutDirection: value })
})

/** 文字颜色 */
const textColor = computed({
  get: () => settingsStore.settings.textStyle.textColor,
  set: (value: string) => settingsStore.updateTextStyle({ textColor: value })
})

/** 填充颜色 */
const fillColor = computed({
  get: () => settingsStore.settings.textStyle.fillColor,
  set: (value: string) => settingsStore.updateTextStyle({ fillColor: value })
})

/** 描边启用 */
const strokeEnabled = computed({
  get: () => settingsStore.settings.textStyle.strokeEnabled,
  set: (value: boolean) => settingsStore.updateTextStyle({ strokeEnabled: value })
})

/** 描边颜色 */
const strokeColor = computed({
  get: () => settingsStore.settings.textStyle.strokeColor,
  set: (value: string) => settingsStore.updateTextStyle({ strokeColor: value })
})

/** 描边宽度 */
const strokeWidth = computed({
  get: () => settingsStore.settings.textStyle.strokeWidth,
  set: (value: number) => settingsStore.updateTextStyle({ strokeWidth: value })
})

/** 修复方式 */
const inpaintMethod = computed({
  get: () => settingsStore.settings.textStyle.inpaintMethod,
  set: (value: InpaintMethod) => settingsStore.updateTextStyle({ inpaintMethod: value })
})

/** 所有字号预设（内置 + 自定义） */
const allFontPresets = computed(() => {
  const custom = settingsStore.customFontPresets
  const all = [...FONT_SIZE_PRESETS, ...custom]
  return [...new Set(all)].sort((a, b) => a - b)
})

// ============================================================
// 方法
// ============================================================

/**
 * 获取字体显示名称
 */
function getFontDisplayName(fontPath: string): string {
  // 从路径中提取文件名
  const parts = fontPath.split('/')
  const fileName = parts[parts.length - 1] || fontPath
  // 移除扩展名
  return fileName.replace(/\.(ttf|ttc|otf)$/i, '')
}

/**
 * 加载字体列表
 */
async function loadFontList(): Promise<void> {
  try {
    const response = await getFontList()
    if (response.success && response.fonts) {
      fontList.value = response.fonts
    }
  } catch (error) {
    console.error('加载字体列表失败:', error)
  }
}

/**
 * 触发字体上传
 */
function triggerFontUpload(): void {
  fontUploadInput.value?.click()
}

/**
 * 处理字体上传
 */
async function handleFontUpload(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  try {
    const response = await uploadFont(file)
    if (response.success) {
      toast.success('字体上传成功')
      await loadFontList()
      // 选择新上传的字体
      if (response.fontPath) {
        fontFamily.value = response.fontPath
      }
    } else {
      toast.error(response.error || '字体上传失败')
    }
  } catch (error) {
    toast.error('字体上传失败')
  } finally {
    // 清空输入框
    input.value = ''
  }
}

/**
 * 设置字号
 */
function setFontSize(size: number): void {
  fontSize.value = size
}

/**
 * 处理字体选择变化（CustomSelect）
 */
function handleFontFamilySelectChange(value: string | number): void {
  fontFamily.value = String(value)
}

/**
 * 处理排版方向变化（CustomSelect）
 */
function handleLayoutDirectionSelectChange(value: string | number): void {
  layoutDirection.value = String(value) as TextDirection
}

/**
 * 处理修复方式变化（CustomSelect）
 */
function handleInpaintMethodSelectChange(value: string | number): void {
  inpaintMethod.value = String(value) as InpaintMethod
}

/**
 * 显示添加预设对话框
 */
function showAddPresetDialog(): void {
  const input = prompt('请输入自定义字号预设值:', String(fontSize.value))
  if (input) {
    const size = parseInt(input, 10)
    if (!isNaN(size) && size >= FONT_SIZE_MIN && size <= FONT_SIZE_MAX) {
      settingsStore.addCustomFontPreset(size)
      toast.success(`已添加字号预设: ${size}`)
    } else {
      toast.error(`字号必须在 ${FONT_SIZE_MIN} - ${FONT_SIZE_MAX} 之间`)
    }
  }
}

/**
 * 全选应用选项
 */
function selectAllOptions(): void {
  applyOptions.value = {
    fontSize: true,
    fontFamily: true,
    layoutDirection: true,
    textColor: true,
    fillColor: true,
    stroke: true
  }
}

/**
 * 应用设置到所有图片
 */
async function applyToAll(): Promise<void> {
  const images = imageStore.images
  if (images.length === 0) {
    toast.error('没有图片可以应用设置')
    return
  }

  // 过滤出有翻译结果的图片
  const translatedImages = images.filter(
    (img) => img.translatedDataURL && img.cleanImageData && img.bubbleStates
  )

  if (translatedImages.length === 0) {
    toast.error('没有已翻译的图片可以应用设置')
    return
  }

  // 构建要应用的设置
  const settings: Record<string, unknown> = {}
  const opts = applyOptions.value

  if (opts.fontSize) settings.font_size = fontSize.value
  if (opts.fontFamily) settings.font_family = fontFamily.value
  if (opts.layoutDirection) settings.text_direction = layoutDirection.value
  if (opts.textColor) settings.text_color = textColor.value
  if (opts.fillColor) settings.fill_color = fillColor.value
  if (opts.stroke) {
    settings.stroke_enabled = strokeEnabled.value
    settings.stroke_color = strokeColor.value
    settings.stroke_width = strokeWidth.value
  }

  try {
    toast.info('正在应用设置到所有图片...')

    const imageData = translatedImages.map((img) => ({
      original_image: img.originalDataURL,
      clean_image: img.cleanImageData!,
      bubble_states: img.bubbleStates!
    }))

    const response = await applySettingsToAllImages(imageData, settings)

    if (response.success && response.data?.translated_images) {
      // 更新图片
      const translatedImagesResult = response.data.translated_images
      translatedImages.forEach((img, idx) => {
        const newTranslatedImage = translatedImagesResult[idx]
        if (newTranslatedImage) {
          const originalIndex = images.indexOf(img)
          imageStore.updateImageByIndex(originalIndex, {
            translatedDataURL: newTranslatedImage,
            hasUnsavedChanges: true
          })
        }
      })
      toast.success('设置已应用到所有图片')
    } else {
      toast.error('应用设置失败')
    }
  } catch (error) {
    toast.error('应用设置失败')
  }

  showApplyOptions.value = false
}

// ============================================================
// 事件处理函数（用于触发保存）
// ============================================================

function handleAutoFontSizeChange(): void {
  console.log('自动字号设置已更改:', autoFontSize.value)
}

function handleFontSizeChange(): void {
  console.log('字号已更改:', fontSize.value)
}

function handleTextColorChange(): void {
  // v-model 自动同步，此处可用于额外处理
}

function handleFillColorChange(): void {
  // v-model 自动同步，此处可用于额外处理
}

function handleStrokeEnabledChange(): void {
  // v-model 自动同步，此处可用于额外处理
}

function handleStrokeColorChange(): void {
  // v-model 自动同步，此处可用于额外处理
}

function handleStrokeWidthChange(): void {
  // v-model 自动同步，此处可用于额外处理
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  loadFontList()
})
</script>


<style scoped>
/* 文字样式设置组件样式 */
.text-style-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* 设置组 */
.setting-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* 设置行 */
.setting-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* 设置标签 */
.setting-label {
  min-width: 60px;
  font-size: 13px;
  color: var(--text-primary, #333);
}

.setting-label.sub-label {
  min-width: 40px;
  padding-left: 10px;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

/* 设置控件容器 */
.setting-control {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 字号控件 */
.font-size-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.auto-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}

.auto-toggle input {
  cursor: pointer;
}

.font-size-slider {
  flex: 1;
  min-width: 80px;
}

.font-size-input {
  width: 50px;
  padding: 4px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  text-align: center;
  font-size: 12px;
}

/* 字号预设 */
.presets-row {
  padding-left: 70px;
}

.font-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.preset-btn {
  padding: 4px 8px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: var(--bg-primary, #fff);
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.preset-btn:hover:not(:disabled) {
  border-color: var(--primary-color, #4a90d9);
  color: var(--primary-color, #4a90d9);
}

.preset-btn.active {
  background: var(--primary-color, #4a90d9);
  border-color: var(--primary-color, #4a90d9);
  color: white;
}

.preset-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.add-preset-btn {
  font-weight: bold;
}

/* 字体选择 */
.font-select {
  flex: 1;
  padding: 6px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  font-size: 13px;
}

.upload-font-btn {
  padding: 6px 10px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: var(--bg-primary, #fff);
  cursor: pointer;
  transition: all 0.2s ease;
}

.upload-font-btn:hover {
  border-color: var(--primary-color, #4a90d9);
}

/* 颜色控件 */
.color-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.color-picker {
  width: 32px;
  height: 32px;
  padding: 0;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  cursor: pointer;
}

.color-input {
  width: 70px;
  padding: 6px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  font-size: 12px;
  font-family: monospace;
}

/* 描边控件 */
.stroke-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  cursor: pointer;
}

.stroke-options {
  padding-left: 10px;
  border-left: 2px solid var(--border-color, #ddd);
  margin-left: 10px;
}

.stroke-width-slider {
  flex: 1;
  min-width: 60px;
}

.stroke-width-value {
  min-width: 35px;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

/* 应用到全部 */
.apply-all-group {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border-color, #ddd);
}

.apply-all-row {
  display: flex;
  gap: 8px;
}

.apply-all-btn {
  flex: 1;
  padding: 8px 12px;
  border: none;
  border-radius: 4px;
  background: var(--primary-color, #4a90d9);
  color: white;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: background 0.2s ease;
}

.apply-all-btn:hover {
  background: var(--primary-color-dark, #3a7bc8);
}

.dropdown-icon {
  font-size: 10px;
}

.apply-options-btn {
  padding: 8px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: var(--bg-primary, #fff);
  cursor: pointer;
  transition: all 0.2s ease;
}

.apply-options-btn:hover {
  border-color: var(--primary-color, #4a90d9);
}

/* 应用选项面板 */
.apply-options-panel {
  margin-top: 10px;
  padding: 10px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.apply-option {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  cursor: pointer;
}

.apply-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border-color, #ddd);
}

.select-all-btn {
  padding: 6px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: var(--bg-primary, #fff);
  font-size: 12px;
  cursor: pointer;
}

.apply-confirm-btn {
  flex: 1;
  padding: 6px 12px;
  border: none;
  border-radius: 4px;
  background: var(--success-color, #5cb85c);
  color: white;
  font-size: 12px;
  cursor: pointer;
}

.apply-confirm-btn:hover {
  background: var(--success-color-dark, #449d44);
}

</style>
