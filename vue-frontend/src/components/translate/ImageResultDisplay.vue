<script setup lang="ts">
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
import { useSettingsStore } from '@/stores/settingsStore'
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

const props = withDefaults(defineProps<Props>(), {
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

/** 下载进度百分比 - 复刻原版 */
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
  
  // 兼容旧数据格式
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
 * @returns 是否为翻译失败
 */
function isTranslationError(text: string): boolean {
  return (text || '').includes('翻译失败')
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
  <section v-if="currentImage" class="result-section card result-card">
    <!-- 控制栏 -->
    <div class="image-controls">
      <!-- 切换原图/翻译图按钮 -->
      <button 
        v-if="hasTranslatedImage"
        id="toggleImageButton"
        class="control-btn"
        @click="toggleImageView"
      >
        {{ showOriginal ? '查看翻译图' : '查看原图' }}
      </button>
      
      <!-- 切换编辑模式按钮 -->
      <button 
        id="toggleEditModeButton"
        class="control-btn"
        :class="{ active: isEditMode }"
        @click="toggleEditMode"
      >
        {{ isEditMode ? '退出编辑' : '切换编辑模式' }}
      </button>
      
      <!-- 图片大小控制 -->
      <div class="image-size-control">
        <label for="imageSize">图片大小:</label>
        <input 
          type="range" 
          id="imageSize" 
          min="50" 
          max="200" 
          :value="imageSize"
          class="slider range-slider"
          @input="updateImageSize"
        >
        <span id="imageSizeValue">{{ imageSize }}%</span>
      </div>
      
      <!-- 重新翻译失败按钮 -->
      <button 
        v-if="hasFailedImages"
        id="retranslateFailedButton"
        class="retry-failed-btn"
        @click="retryFailed"
        title="重新翻译所有失败的图片"
      >
        重新翻译失败图片 ({{ imageStore.failedImageCount }})
      </button>
    </div>
    
    <!-- 图片内容区域 -->
    <div class="content-container">
      <div class="image-container">
        <!-- 翻译后图片 -->
        <img 
          id="translatedImageDisplay" 
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
      <pre id="detectedTextList">
        <template v-if="hasDetectedTexts">
          <span 
            v-for="(item, index) in detectedTexts" 
            :key="index"
            class="text-item"
          >
            <span class="original-text">{{ formatOriginalText(item.original) }}</span>
            <br />
            <span 
              :class="['translated-text', { 'translation-error': isTranslationError(item.translated) }]"
            >{{ formatTranslatedText(item.translated) }}</span>
            <br />
            <span class="separator">──────────────────────────</span>
            <br /><br />
          </span>
        </template>
        <template v-else>未检测到文本或尚未翻译</template>
      </pre>
    </div>
    
    <!-- 下载和导出按钮区域 -->
    <div class="download-section">
      <!-- 下载进度条 - 复刻原版 #translationProgressBar -->
      <ProgressBar
        v-if="isDownloading"
        :visible="true"
        :percentage="downloadProgress"
        :label="downloadProgressText || '下载中，请稍候...'"
      />
      <div class="download-buttons">
        <button 
          id="downloadButton" 
          class="download-btn primary"
          :disabled="!hasDownloadableImage"
          @click="handleDownloadCurrent"
        >
          下载当前图片
        </button>
        <div class="download-all-container">
          <button 
            id="downloadAllImagesButton" 
            class="download-btn primary"
            :disabled="!hasImages"
            @click="handleDownloadAll"
          >
            下载所有图片
          </button>
          <div class="download-format-selector">
            <CustomSelect
              v-model="downloadFormat"
              :options="downloadFormatOptions"
            />
          </div>
        </div>
        <button 
          id="exportTextButton" 
          class="download-btn success"
          :disabled="!hasImages"
          @click="handleExportText"
        >
          导出文本
        </button>
        <button 
          id="importTextButton" 
          class="download-btn success"
          :disabled="!hasImages"
          @click="triggerImportText"
        >
          导入文本
        </button>
        <!-- 隐藏的文件输入框，用于导入文本 -->
        <input 
          type="file" 
          ref="importFileInput"
          id="importTextFileInput" 
          style="display: none;" 
          accept=".json"
          @change="handleImportFile"
        >
      </div>
    </div>
  </section>
  
  <!-- 空状态提示 - 仅在没有图片时显示简洁提示 -->
  <section v-else class="empty-state-section">
    <!-- 空状态不显示额外卡片，保持与原版一致 -->
  </section>
</template>

<style scoped>
/* 结果区域卡片 - 匹配原版 #result-section 样式 */
.result-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.result-section:hover {
  box-shadow: 0 8px 16px rgba(0,0,0,0.12);
}

/* 控制栏 - 匹配原版 .image-controls 样式 */
.image-controls {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 20px;
  width: 100%;
}

/* 控制按钮 */
.control-btn {
  padding: 10px 18px;
  background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95em;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px rgba(52, 152, 219, 0.2);
}

.control-btn:hover {
  background: linear-gradient(135deg, #1f6aa6 0%, #3498db 100%);
  box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
  transform: translateY(-2px);
}

.control-btn.active {
  background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
  box-shadow: 0 2px 6px rgba(39, 174, 96, 0.2);
}

/* 图片大小控制 */
.image-size-control {
  display: flex;
  align-items: center;
  gap: 10px;
}

.image-size-control label {
  font-size: 14px;
  color: #555;
}

.image-size-control .slider {
  width: 120px;
  cursor: pointer;
}

#imageSizeValue {
  min-width: 45px;
  text-align: right;
  font-size: 14px;
  color: #555;
}

/* 重试按钮 */
.retry-failed-btn {
  background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95em;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px rgba(243, 156, 18, 0.2);
}

.retry-failed-btn:hover {
  background: linear-gradient(135deg, #d35400 0%, #f39c12 100%);
  box-shadow: 0 4px 10px rgba(243, 156, 18, 0.3);
  transform: translateY(-2px);
}

/* 内容容器 */
.content-container {
  width: 100%;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  background-color: var(--bg-color, #f8fafc);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  text-align: center;
}

/* 图片容器 */
.image-container {
  position: relative;
  max-width: 100%;
  text-align: center;
}

/* 翻译后图片 - 匹配原版 #translatedImageDisplay 样式 */
#translatedImageDisplay {
  position: relative;
  max-width: 100%;
  height: auto;
  object-fit: contain;
  border: none;
  transition: width 0.3s ease;
  display: block;
  margin: 0 auto;
}

/* 空状态区域 - 保持与原版一致，不显示额外卡片 */
.empty-state-section {
  display: none;
}

/* 检测文本信息区域 */
.text-info {
  width: 100%;
  margin-top: 20px;
  padding: 15px;
  background-color: var(--secondary-bg, #f9f9f9);
  border: 1px solid var(--border-color, #eee);
  border-radius: 4px;
  white-space: pre-wrap;
  font-family: monospace;
  font-size: 0.9em;
  text-align: left;
  overflow-x: auto;
  height: 300px;
  overflow-y: auto;
}

.text-info h3 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: var(--text-primary, #333);
  font-weight: 600;
}

#detectedTextList {
  margin: 0;
  padding: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.text-item {
  display: block;
}

.original-text {
  color: var(--text-primary, #333);
}

.translated-text {
  color: var(--primary-color, #4a90d9);
}

.translated-text.translation-error {
  color: var(--error-color, #e74c3c);
}

.separator {
  color: var(--text-secondary, #999);
}

/* 下载区域样式 - 匹配原版 */
.download-section {
  width: 100%;
  margin-top: 20px;
  padding: 15px;
  background-color: var(--secondary-bg, #f9f9f9);
  border: 1px solid var(--border-color, #eee);
  border-radius: 8px;
}

.download-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  align-items: center;
}

.download-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95em;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.download-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.download-btn.primary {
  background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
  color: white;
}

.download-btn.primary:hover:not(:disabled) {
  background: linear-gradient(135deg, #1f6aa6 0%, #3498db 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
}

.download-btn.success {
  background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
  color: white;
}

.download-btn.success:hover:not(:disabled) {
  background: linear-gradient(135deg, #1e8449 0%, #2ecc71 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 10px rgba(39, 174, 96, 0.3);
}

.download-all-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.download-format-selector select {
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  background-color: white;
  font-size: 0.9em;
  cursor: pointer;
  transition: border-color 0.2s;
}

.download-format-selector select:hover {
  border-color: var(--primary-color, #4a90d9);
}

.download-format-selector select:focus {
  outline: none;
  border-color: var(--primary-color, #4a90d9);
  box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.2);
}

/* ===================================
   图像展示区样式 - 完整迁移自 image-display.css
   =================================== */

#image-display-area {
  flex-grow: 2.4;
  padding: 20px;
  margin-left: 340px;
  margin-right: 240px;
  max-width: none;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

#image-display-area .card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  text-align: center;
  flex-grow: 0;
  margin-bottom: 25px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

#image-display-area .card:hover {
  box-shadow: 0 8px 16px rgba(0,0,0,0.12);
}

#image-display-area #upload-section {
  flex: 0 0 auto;
  min-height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-bottom: 15px;
}

#image-display-area #upload-section #drop-area {
  border: 2px dashed #b0bec5;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  color: #546e7a;
  margin-bottom: 15px;
  width: 85%;
  margin-left: auto;
  margin-right: auto;
  min-height: 100px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  background-color: #f7fafc;
}

#image-display-area #upload-section #drop-area:hover {
  border-color: #3498db;
  background-color: #ecf5fe;
  transform: translateY(-3px);
}

#image-display-area #upload-section #drop-area.highlight {
  border-color: #3498db;
  background-color: #ecf5fe;
  box-shadow: 0 0 15px rgba(52, 152, 219, 0.3);
}

#image-display-area #upload-section #drop-area p {
  font-size: 1.1em;
  margin: 10px 0;
}

#image-display-area #result-section {
  flex: 1 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
}

#image-display-area #result-section .image-container {
  width: 100%;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  background-color: var(--bg-color);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  text-align: center;
}

#image-display-area #result-section #translatedImageDisplay {
  position: relative;
  max-width: 100%;
  height: auto;
  object-fit: contain;
  border: none;
  transition: width 0.3s ease;
}

#image-display-area #result-section .image-controls {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 20px;
  width: 100%;
}

#image-display-area #result-section .text-info {
  width: 100%;
  margin-top: 20px;
  padding: 15px;
  background-color: #f9f9f9;
  border: 1px solid #eee;
  border-radius: 4px;
  white-space: pre-wrap;
  font-family: monospace;
  font-size: 0.9em;
  text-align: left;
  overflow-x: auto;
  height: 300px;
  overflow-y: auto;
}

#image-display-area #result-section .download-buttons {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 25px;
  width: 100%;
  flex-wrap: wrap;
}

.download-all-container {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
}

.download-format-selector {
  width: auto;
  max-width: 150px;
}

.download-format-selector select {
  width: 100%;
  padding: 10px;
  border: 1px solid #e0e6ed;
  border-radius: 8px;
  font-size: 0.9em;
  background-color: #f8fafc;
  margin-top: 5px;
  transition: border-color 0.3s, box-shadow 0.3s;
  cursor: pointer;
}

.download-format-selector select:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
  outline: none;
}

#image-display-area #result-section .download-buttons button {
  display: none;
  padding: 12px 20px;
  background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1em;
  margin-top: 0;
  transition: all 0.3s ease;
  min-width: 150px;
  font-weight: 600;
  letter-spacing: 0.5px;
  box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
  position: relative;
  overflow: hidden;
}

#image-display-area #result-section .download-buttons button:hover {
  background: linear-gradient(135deg, #1f6aa6 0%, #3498db 100%);
  box-shadow: 0 6px 10px rgba(52, 152, 219, 0.3);
  transform: translateY(-2px);
}

#image-display-area #result-section .download-buttons button:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

#image-display-area #result-section .download-buttons button:hover:before {
  left: 100%;
}

#exportTextButton, #importTextButton {
  background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
  box-shadow: 0 4px 6px rgba(46, 204, 113, 0.2) !important;
}

#exportTextButton:hover, #importTextButton:hover {
  background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
  box-shadow: 0 6px 10px rgba(46, 204, 113, 0.3) !important;
}

#image-display-area #upload-section #loadingMessage,
#image-display-area #upload-section .error-message {
  margin-top: 15px;
  padding: 10px 15px;
  border-radius: 8px;
  text-align: center;
  font-size: 1em;
}

#image-display-area #upload-section #loadingMessage:before {
  content: '处理中...';
}

#image-display-area #upload-section .error-message {
  color: #c53030;
  background-color: #fff5f5;
  border-left: 4px solid #fc8181;
  font-weight: bold;
}

#image-display-area #upload-section .loader {
  display: none;
}

#translatingMessage, #downloadingMessage {
  display: none !important;
}

#result-section p#translatingMessage {
  margin-top: 10px;
  font-style: italic;
  color: #777;
  text-align: center;
  display: none;
}

#result-section p#downloadingMessage {
  margin-top: 10px;
  font-style: italic;
  color: #777;
  text-align: center;
  display: none;
}

.highlight-bubble {
  position: absolute;
  border: 1px solid #2980b9;
  pointer-events: auto;
  z-index: 1000;
  border-radius: 5px;
  box-sizing: border-box;
  overflow: visible !important;
  cursor: pointer;
}

.highlight-bubble.selected {
  border: 1px solid #e74c3c;
}

@keyframes pulse-border {
  0% { border-color: #e74c3c; }
  50% { border-color: #ff6b6b; }
  100% { border-color: #e74c3c; }
}
</style>
