<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 图片结果显示组件
 * 显示翻译后的图片，支持原图/翻译图切换、图片大小调整
 * 
 * 功能：
 * - 翻译后图片显示
 * - 切换原图/翻译图按钮
 * - 切换编辑模式按钮
 * - 图片大小滑块（50%-200%）
 * - 重新翻译失败按钮
 * - 检测文本信息显示（原文 → 译文对照）
 * - 导出/导入文本功能
 * - 下载图片功能
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useExportImport, type DownloadFormat } from '@/composables/useExportImport'
import CustomSelect from '@/components/common/CustomSelect.vue'
import ProgressBar from '@/components/common/ProgressBar.vue'

/** 下载格式选项 */
const downloadFormatOptions = [
  { label: 'ZIP压缩包', value: 'zip' },
  { label: 'PDF文档', value: 'pdf' },
  { label: 'CBZ漫画', value: 'cbz' }
]

// Props 定义
interface Props {
  /** 是否处于编辑模式 */
  isEditMode?: boolean
}

withDefaults(defineProps<Props>(), {
  isEditMode: false
})

// Emits 定义
const emit = defineEmits<{
  /** 切换编辑模式 */
  (e: 'toggle-edit-mode'): void
  /** 重新翻译失败图片 */
  (e: 'retry-failed'): void
}>()

// Stores
const imageStore = useImageStore()
const settingsStore = useSettingsStore()

// 导出导入功能
const exportImport = useExportImport()

// ============================================================
// 状态定义
// ============================================================

/** 图片大小百分比 */
const imageSize = ref(100)

/** 【修复6】是否显示原图（从当前图片状态读取，按图片持久化） */
const showOriginal = computed({
  get: () => currentImage.value?.showOriginal ?? false,
  set: (val: boolean) => {
    if (currentImage.value) {
      imageStore.updateCurrentImage({ showOriginal: val })
    }
  }
})

/** 下载格式 */
const downloadFormat = ref<DownloadFormat>('zip')

/** 导入文件输入框引用 */
const importFileInput = ref<HTMLInputElement | null>(null)

/** 是否正在下载 */
const isDownloading = computed(() => exportImport.isDownloading.value)

/** 下载进度文本 */
const downloadProgressText = computed(() => exportImport.downloadProgressText.value)

/** 下载进度百分比 - 当前行为 */
const downloadProgress = computed(() => exportImport.downloadProgress.value)

/** 是否有图片 */
const hasImages = computed(() => imageStore.hasImages)

// ============================================================
// 计算属性
// ============================================================

/** 当前图片 */
const currentImage = computed(() => imageStore.currentImage)

/** 是否有翻译结果 */
const hasTranslatedImage = computed(() => !!currentImage.value?.translatedDataURL)

/** 是否有可下载的图片（原图或翻译图） */
const hasDownloadableImage = computed(() => 
  !!(currentImage.value?.translatedDataURL || currentImage.value?.originalDataURL)
)

/** 当前显示的图片URL */
const displayImageUrl = computed(() => {
  if (!currentImage.value) return ''
  if (showOriginal.value || !currentImage.value.translatedDataURL) {
    return currentImage.value.originalDataURL
  }
  return currentImage.value.translatedDataURL
})

/** 是否有翻译失败的图片 */
const hasFailedImages = computed(() => imageStore.failedImageCount > 0)

/** 图片样式 */
const imageStyle = computed(() => ({
  width: `${imageSize.value}%`
}))

/** 是否使用文本框提示词（决定显示 textboxText 还是 translatedText） */
const useTextboxPrompt = computed(() => settingsStore.settings.useTextboxPrompt)

/** 检测到的文本列表（原文和译文对照） */
const detectedTexts = computed<Array<{ original: string; translated: string }>>(() => {
  if (!currentImage.value) return []
  
  // 优先从 bubbleStates 获取文本
  if (currentImage.value.bubbleStates && currentImage.value.bubbleStates.length > 0) {
    return currentImage.value.bubbleStates.map(state => ({
      original: state.originalText || '',
      translated: useTextboxPrompt.value 
        ? (state.textboxText || state.translatedText || '')
        : (state.translatedText || '')
    }))
  }
  
  // 读取历史数据格式
  const originalTexts = currentImage.value.originalTexts || []
  const translatedTexts = useTextboxPrompt.value
    ? (currentImage.value.textboxTexts || currentImage.value.bubbleTexts || [])
    : (currentImage.value.bubbleTexts || [])
  
  if (originalTexts.length === 0) return []
  
  return originalTexts.map((original, index) => ({
    original: original || '',
    translated: translatedTexts[index] || ''
  }))
})

/** 是否有检测到的文本 */
const hasDetectedTexts = computed(() => detectedTexts.value.length > 0)

// ============================================================
// 常量
// ============================================================

/** 文本自动换行的最大行长度 */
const MAX_LINE_LENGTH = 60

// ============================================================
// 方法
// ============================================================

/**
 * 文本自动换行
 * @param text - 输入文本
 * @returns 处理换行后的文本
 */
function wrapText(text: string): string {
  if (!text || text.length <= MAX_LINE_LENGTH) return text
  
  let result = ''
  let currentLine = ''
  
  for (let i = 0; i < text.length; i++) {
    currentLine += text[i]
    if (currentLine.length >= MAX_LINE_LENGTH) {
      // 查找合适的断点（标点符号）
      let breakPoint = -1
      for (let j = currentLine.length - 1; j >= 0; j--) {
        const char = currentLine[j]
        if (char && ['。', '！', '？', '.', '!', '?', '；', ';', '，', ','].includes(char)) {
          breakPoint = j + 1
          break
        }
      }
      
      if (breakPoint > MAX_LINE_LENGTH * 0.6) {
        result += currentLine.substring(0, breakPoint) + '\n'
        currentLine = currentLine.substring(breakPoint)
      } else {
        result += currentLine + '\n'
        currentLine = ''
      }
    }
  }
  
  if (currentLine) {
    result += currentLine
  }
  
  return result
}

/**
 * 格式化原文文本
 * @param text - 原文
 * @returns 格式化后的文本
 */
function formatOriginalText(text: string): string {
  return wrapText((text || '').trim())
}

/**
 * 格式化译文文本
 * @param text - 译文
 * @returns 格式化后的文本
 */
function formatTranslatedText(text: string): string {
  const trimmed = (text || '').trim()
  return wrapText(trimmed)
}

/**
 * 检查译文是否为翻译失败
 * @param text - 译文
 * @returns 是否为翻译失败（匹配 【翻译失败】 或包含"翻译失败"的格式）
 */
function isTranslationError(text: string): boolean {
  const t = text || ''
  return t.includes('【翻译失败】') || t.includes('[翻译失败]') || t.includes('翻译失败')
}

/**
 * 切换原图/翻译图
 */
function toggleImageView(): void {
  showOriginal.value = !showOriginal.value
}

/**
 * 切换编辑模式
 */
function toggleEditMode(): void {
  emit('toggle-edit-mode')
}

/**
 * 更新图片大小
 */
function updateImageSize(event: Event): void {
  const input = event.target as HTMLInputElement
  imageSize.value = parseInt(input.value, 10)
}

/**
 * 重新翻译失败图片
 */
function retryFailed(): void {
  emit('retry-failed')
}

/**
 * 下载当前图片
 */
function handleDownloadCurrent(): void {
  exportImport.downloadCurrentImage()
}

/**
 * 下载所有图片
 */
function handleDownloadAll(): void {
  exportImport.downloadAllImages(downloadFormat.value)
}

/**
 * 导出文本
 */
function handleExportText(): void {
  exportImport.exportText()
}

/**
 * 触发导入文本文件选择
 */
function triggerImportText(): void {
  importFileInput.value?.click()
}

/**
 * 处理导入文件选择
 */
async function handleImportFile(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) {
    await exportImport.importText(file)
    // 清空文件输入框，以便可以再次选择同一文件
    input.value = ''
  }
}

</script>

<template>
  <section v-if="currentImage" class="image-result-display result-section result-card">
    <!-- 控制栏 -->
    <div class="image-controls">
      <!-- 切换原图/翻译图按钮 -->
      <UiButton
        variant="toolbar" 
        v-if="hasTranslatedImage"
        id="toggleImageButton"
        class="control-btn"
        @click="toggleImageView"
      >
        {{ showOriginal ? '查看翻译图' : '查看原图' }}
      </UiButton>
      
      <!-- 切换编辑模式按钮 -->
      <UiButton
        variant="toolbar" 
        id="toggleEditModeButton"
        class="control-btn"
        :class="{ active: isEditMode }"
        @click="toggleEditMode"
      >
        {{ isEditMode ? '退出编辑' : '切换编辑模式' }}
      </UiButton>
      
      <!-- 图片大小控制 -->
      <div class="image-size-control">
        <label for="imageSize">图片大小:</label>
        <UiInput 
          type="range" 
          id="imageSize" 
          min="50" 
          max="200" 
          :value="imageSize"
          class="slider range-slider"
          @input="updateImageSize"
        />
        <span class="image-size-value">{{ imageSize }}%</span>
      </div>
      
      <!-- 重新翻译失败按钮 -->
      <UiButton
        variant="toolbar" 
        v-if="hasFailedImages"
        id="retranslateFailedButton"
        class="retry-failed-btn"
        @click="retryFailed"
        title="重新翻译所有失败的图片"
      >
        重新翻译失败图片 ({{ imageStore.failedImageCount }})
      </UiButton>
    </div>
    
    <!-- 图片内容区域 -->
    <div class="content-container">
      <div class="image-container">
        <!-- 翻译后图片 -->
        <img
          class="translated-image"
          :src="displayImageUrl" 
          alt="翻译后图片"
          :style="imageStyle"
        >
      </div>
    </div>
    
    <!-- 检测文本信息区域 -->
    <div 
      id="detectedTextInfo"
      class="text-info"
    >
      <h3>检测到的文本（原文 → 译文）</h3>
      <pre class="detected-text-list"><template v-if="hasDetectedTexts"><span v-for="(item, index) in detectedTexts" :key="index" class="text-item"><span class="original-text">{{ formatOriginalText(item.original) }}</span>
<span :class="['translated-text', { 'translation-error': isTranslationError(item.translated) }]">{{ formatTranslatedText(item.translated) }}</span>
<span class="separator">──────────────────────────</span>

</span></template><template v-else>未检测到文本或尚未翻译</template></pre>
    </div>
    
    <!-- 下载和导出按钮区域 -->
    <div class="download-section">
      <!-- 下载进度条 - 当前行为 #translationProgressBar -->
      <ProgressBar
        v-if="isDownloading"
        :visible="true"
        :percentage="downloadProgress"
        :label="downloadProgressText || '下载中，请稍候...'"
      />
      <div class="download-buttons">
        <UiButton
          variant="primary" 
          id="downloadButton" 
          class="download-btn"
          :disabled="!hasDownloadableImage"
          @click="handleDownloadCurrent"
        >
          下载当前图片
        </UiButton>
        <div class="download-all-container">
          <UiButton
            variant="primary" 
            id="downloadAllImagesButton" 
            class="download-btn"
            :disabled="!hasImages"
            @click="handleDownloadAll"
          >
            下载所有图片
          </UiButton>
          <div class="download-format-selector">
            <CustomSelect
              v-model="downloadFormat"
              :options="downloadFormatOptions"
            />
          </div>
        </div>
        <UiButton
          variant="toolbar" 
          id="exportTextButton" 
          class="download-btn success"
          :disabled="!hasImages"
          @click="handleExportText"
        >
          导出文本
        </UiButton>
        <UiButton
          variant="toolbar" 
          id="importTextButton" 
          class="download-btn success"
          :disabled="!hasImages"
          @click="triggerImportText"
        >
          导入文本
        </UiButton>
        <!-- 隐藏的文件输入框，用于导入文本 -->
        <UiFileInput 
          ref="importFileInput"
          id="importTextFileInput" 
          style="display: none;" 
          accept=".json"
          @change="handleImportFile"
        />
      </div>
    </div>
  </section>
  
  <!-- 空状态提示 - 仅在没有图片时显示简洁提示 -->
  <section v-else class="empty-state-section">
    <!-- 空状态不显示额外卡片，保持与当前行为一致 -->
  </section>
</template>

<style scoped>/* 结果区域卡片 */
.image-result-display.result-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px var(--image-result-display-shadow-default);
  padding: 25px;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.image-result-display.result-section:hover {
  box-shadow: 0 8px 16px var(--image-result-display-shadow-raised);
}

/* 图片控制栏 */
.image-result-display .image-controls {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 20px;
  width: 100%;
}

/* 控制按钮 */
.image-result-display .control-btn {
  padding: 10px 18px;
  background: linear-gradient(135deg, var(--image-result-display-surface-base) 0%, var(--color-surface-accent) 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95em;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px var(--image-result-display-shadow-floating);
}

.image-result-display .control-btn:hover {
  background: linear-gradient(135deg, var(--image-result-display-surface-raised) 0%, var(--color-surface-accent) 100%);
  box-shadow: 0 4px 10px var(--image-result-display-shadow-strong);
  transform: translateY(-2px);
}

.image-result-display .control-btn.active {
  background: linear-gradient(135deg, var(--color-surface-success) 0%, var(--image-result-display-surface-muted) 100%);
  box-shadow: 0 2px 6px var(--image-result-display-shadow-soft);
}

/* 图片大小控制 */
.image-result-display .image-size-control {
  display: flex;
  align-items: center;
  gap: 10px;
}

.image-result-display .image-size-control label {
  font-size: 14px;
  color: var(--image-result-display-text-primary);
}

.image-result-display .image-size-control .slider {
  width: 120px;
  cursor: pointer;
}

.image-result-display .image-size-value {
  min-width: 45px;
  text-align: right;
  font-size: 14px;
  color: var(--image-result-display-text-primary);
}

/* 重试按钮 */
.image-result-display .retry-failed-btn {
  background: linear-gradient(135deg, var(--image-result-display-surface-subtle) 0%, var(--image-result-display-surface-hover) 100%);
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95em;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px var(--image-result-display-shadow-focus);
}

.image-result-display .retry-failed-btn:hover {
  background: linear-gradient(135deg, var(--image-result-display-surface-active) 0%, var(--image-result-display-surface-hover) 100%);
  box-shadow: 0 4px 10px var(--image-result-display-shadow-glow);
  transform: translateY(-2px);
}

/* 内容容器 */
.image-result-display .content-container {
  width: 100%;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  background-color: var(--color-surface-app, var(--color-surface-quiet));
  border: 1px solid var(--color-border-muted, var(--color-border-muted));
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px var(--image-result-display-shadow-inset);
  text-align: center;
}

/* 图片容器 */
.image-result-display .image-container {
  position: relative;
  max-width: 100%;
  text-align: center;
}

/* 翻译后图片 */
.image-result-display .translated-image {
  position: relative;
  max-width: 100%;
  height: auto;
  object-fit: contain;
  border: none;
  transition: width 0.3s ease;
  display: block;
  margin: 0 auto;
}

/* 空状态区域 - 保持与当前行为一致，不显示额外卡片 */
.image-result-display .empty-state-section {
  display: none;
}

/* 检测文本信息区域 */
.image-result-display .text-info {
  width: 100%;
  margin-top: 20px;
  padding: 15px;
  background-color: var(--secondary-bg, var(--image-result-display-surface-selected));
  border: 1px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 4px;
  white-space: pre-wrap;
  font-family: var(--font-mono);
  font-size: 0.9em;
  text-align: left;
  overflow-x: auto;
  height: 300px;
  overflow-y: auto;
}

.image-result-display .text-info h3 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: var(--color-text-default, var(--color-text-default));
  font-weight: 600;
}

.image-result-display .detected-text-list {
  margin: 0;
  padding: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.image-result-display .text-item {
  display: block;
}

.image-result-display .original-text {
  color: var(--color-text-default, var(--color-text-default));
}

.image-result-display .translated-text {
  color: var(--color-action-primary, var(--image-result-display-text-secondary));
}

.image-result-display .translated-text.translation-error {
  color: var(--color-status-error, var(--color-text-danger-strong));
}

.image-result-display .separator {
  color: var(--color-text-supporting, var(--color-text-muted));
}

/* 下载区域 */
.image-result-display .download-section {
  width: 100%;
  margin-top: 20px;
  padding: 15px;
  background-color: var(--secondary-bg, var(--image-result-display-surface-selected));
  border: 1px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 8px;
}

.image-result-display .download-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  align-items: center;
}

.image-result-display .download-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95em;
  font-weight: 500;
  white-space: normal;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px var(--image-result-display-shadow-overlay);
}

.image-result-display .download-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.image-result-display .download-btn.primary {
  background: linear-gradient(135deg, var(--image-result-display-surface-base) 0%, var(--color-surface-accent) 100%);
  color: white;
}

.image-result-display .download-btn.primary:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--image-result-display-surface-raised) 0%, var(--color-surface-accent) 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 10px var(--image-result-display-shadow-strong);
}

.image-result-display .download-btn.success {
  background: linear-gradient(135deg, var(--color-surface-success) 0%, var(--image-result-display-surface-muted) 100%);
  color: white;
}

.image-result-display .download-btn.success:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--image-result-display-surface-overlay) 0%, var(--image-result-display-surface-muted) 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 10px var(--image-result-display-shadow-brand);
}

.image-result-display .download-all-container {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
}

.image-result-display .download-format-selector {
  width: auto;
  max-width: 150px;
}

.image-result-display .download-format-selector select {
  width: 100%;
  padding: 10px;
  border: 1px solid var(--image-result-display-border-default);
  border-radius: 8px;
  font-size: 0.9em;
  background-color: var(--color-surface-quiet);
  margin-top: 5px;
  transition: border-color 0.3s, box-shadow 0.3s;
  cursor: pointer;
}

.image-result-display .download-format-selector select:hover {
  border-color: var(--color-action-primary, var(--color-border-info));
}

.image-result-display .download-format-selector select:focus {
  border-color: var(--color-border-accent);
  box-shadow: 0 0 0 3px var(--image-result-display-shadow-floating);
  outline: none;
}

.image-result-display .highlight-bubble {
  position: absolute;
  border: 1px solid var(--image-result-display-border-strong);
  pointer-events: auto;
  z-index: var(--z-overlay);
  border-radius: 5px;
  overflow: visible;
  cursor: pointer;
}

.image-result-display .highlight-bubble.selected {
  border: 1px solid var(--image-result-display-border-muted);
}

@media (--breakpoint-md-down) {
  .image-result-display .image-container {
    max-width: 280px;
    margin-top: 25px;
  }
}
</style>
