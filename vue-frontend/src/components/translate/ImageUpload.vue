<script setup lang="ts">
/**
 * 图片上传组件
 * 支持多图片批量上传、PDF解析、MOBI/AZW解析、拖拽上传
 * 
 * 功能：
 * - 图片上传（支持 jpg/png/webp 等格式）
 * - PDF 文件解析（前端 pdf.js 或后端 PyMuPDF）
 * - MOBI/AZW/AZW3 电子书解析（后端分批解析）
 * - 拖拽上传支持
 * - 文件名自然排序
 * - 上传进度显示
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { showToast } from '@/utils/toast'
import ProgressBar from '@/components/common/ProgressBar.vue'
import {
  parsePdfStart,
  parsePdfBatch,
  parsePdfCleanup,
  parseMobiStart,
  parseMobiBatch,
  parseMobiCleanup,
} from '@/api/system'

// ============================================================
// Props 和 Emits
// ============================================================

const emit = defineEmits<{
  /** 上传完成 */
  (e: 'uploadComplete', count: number): void
}>()

// ============================================================
// Stores
// ============================================================

const imageStore = useImageStore()
const settingsStore = useSettingsStore()

// ============================================================
// 状态定义
// ============================================================

/** 文件输入框引用 */
const fileInputRef = ref<HTMLInputElement | null>(null)

/** 是否正在加载 */
const isLoading = ref(false)

/** 是否拖拽中 */
const isDragging = ref(false)

/** 错误消息 */
const errorMessage = ref('')

/** 上传进度（0-100） */
const uploadProgress = ref(0)

/** 当前处理的文件名 */
const currentFileName = ref('')

/** 是否显示进度条 */
const showProgress = ref(false)

// ============================================================
// 计算属性
// ============================================================

/** PDF 处理方式（前端/后端） */
const pdfProcessingMethod = computed(() => settingsStore.settings.pdfProcessingMethod)

// ============================================================
// 方法
// ============================================================

/**
 * 触发文件选择对话框
 */
function triggerFileSelect() {
  fileInputRef.value?.click()
}

/**
 * 处理文件选择
 */
async function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  
  await processFiles(Array.from(input.files))
  
  // 清空 input 以便重复选择同一文件
  input.value = ''
}

/**
 * 处理拖拽放置
 */
async function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false
  
  if (!event.dataTransfer?.files || event.dataTransfer.files.length === 0) return
  
  await processFiles(Array.from(event.dataTransfer.files))
}

/**
 * 处理拖拽进入
 */
function handleDragOver(event: DragEvent) {
  event.preventDefault()
  isDragging.value = true
}

/**
 * 处理拖拽离开
 */
function handleDragLeave(event: DragEvent) {
  // 检查是否真的离开了拖拽区域（而不是进入子元素）
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const x = event.clientX
  const y = event.clientY
  
  if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
    isDragging.value = false
  }
}

/**
 * 处理文件列表
 * @param files 文件列表
 */
async function processFiles(files: File[]) {
  if (files.length === 0) return
  
  isLoading.value = true
  errorMessage.value = ''
  showProgress.value = true
  uploadProgress.value = 0
  
  try {
    // 复刻原版：不在此处预排序，由 TranslateView.handleUploadComplete 统一排序
    let processedCount = 0
    const totalFiles = files.length
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (!file) continue
      
      currentFileName.value = file.name
      
      const fileType = file.type
      const fileName = file.name.toLowerCase()
      
      if (fileType.startsWith('image/')) {
        // 处理图片文件
        await processImageFile(file)
        processedCount++
      } else if (fileType === 'application/pdf' || fileName.endsWith('.pdf')) {
        // 处理 PDF 文件
        const count = await processPdfFile(file)
        processedCount += count
      } else if (fileName.endsWith('.mobi') || fileName.endsWith('.azw') || fileName.endsWith('.azw3')) {
        // 处理 MOBI/AZW 文件
        const count = await processMobiFile(file)
        processedCount += count
      } else {
        console.warn(`不支持的文件类型: ${file.name}`)
        showToast(`不支持的文件类型: ${file.name}`, 'warning')
      }
      
      // 更新进度
      uploadProgress.value = Math.round(((i + 1) / totalFiles) * 100)
    }
    
    if (processedCount > 0) {
      showToast(`已添加 ${processedCount} 张图片`, 'success')
      emit('uploadComplete', processedCount)
    }
  } catch (error) {
    console.error('处理文件失败:', error)
    const errMsg = error instanceof Error ? error.message : '处理文件失败，请重试'
    errorMessage.value = errMsg
    showToast(errMsg, 'error')
  } finally {
    isLoading.value = false
    showProgress.value = false
    currentFileName.value = ''
  }
}

/**
 * 处理图片文件
 * @param file 图片文件
 */
async function processImageFile(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const dataURL = e.target?.result as string
      imageStore.addImage(file.name, dataURL)
      resolve()
    }
    reader.onerror = () => reject(new Error(`读取图片文件失败: ${file.name}`))
    reader.readAsDataURL(file)
  })
}


/**
 * 处理 PDF 文件
 * 支持前端 pdf.js 和后端 PyMuPDF 两种方式
 * @param file PDF 文件
 * @returns 处理的图片数量
 */
async function processPdfFile(file: File): Promise<number> {
  if (pdfProcessingMethod.value === 'frontend') {
    // 前端 pdf.js 解析
    return await processPdfFrontend(file)
  } else {
    // 后端 PyMuPDF 分批解析
    return await processPdfBackend(file)
  }
}

/**
 * 将 Blob 转换为 DataURL（复刻原版 blobToDataURL）
 * @param blob - Blob 对象
 * @returns DataURL 字符串
 */
function blobToDataURL(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

/**
 * 前端 pdf.js 解析 PDF
 * 复刻原版 main.js processPDFFilesFrontend 逻辑：
 * - 使用 OffscreenCanvas 后台渲染（页面不可见时也能继续渲染）
 * - 输出 JPEG 格式（quality 1.0），与原版保持一致
 * @param file PDF 文件
 * @returns 处理的图片数量
 */
async function processPdfFrontend(file: File): Promise<number> {
  try {
    // 动态导入 pdf.js
    const pdfjsLib = await import('pdfjs-dist')
    
    // 设置 worker（使用 CDN）
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`
    
    // 读取文件为 ArrayBuffer
    const arrayBuffer = await file.arrayBuffer()
    
    // 加载 PDF 文档
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
    const numPages = pdf.numPages
    
    console.log(`PDF ${file.name} 共 ${numPages} 页，开始本地渲染...`)
    showToast(`正在解析 PDF，共 ${numPages} 页...`, 'info')
    
    // 检测是否支持 OffscreenCanvas（后台渲染不受页面可见性影响）
    const useOffscreen = typeof OffscreenCanvas !== 'undefined'
    if (useOffscreen) {
      console.log('使用 OffscreenCanvas 后台渲染模式')
    }
    
    let processedCount = 0
    
    for (let pageNum = 1; pageNum <= numPages; pageNum++) {
      currentFileName.value = `${file.name} - 第 ${pageNum}/${numPages} 页`
      uploadProgress.value = Math.round((pageNum / numPages) * 100)
      
      try {
        const page = await pdf.getPage(pageNum)
        
        // 设置渲染比例（2.0 可以获得较高清晰度，与原版一致）
        const scale = 2.0
        const viewport = page.getViewport({ scale })
        
        let dataURL: string
        
        if (useOffscreen) {
          // 使用 OffscreenCanvas - 后台也能继续渲染（复刻原版）
          const offscreen = new OffscreenCanvas(viewport.width, viewport.height)
          const context = offscreen.getContext('2d')
          
          await page.render({
            canvasContext: context as unknown as CanvasRenderingContext2D,
            viewport: viewport
          }).promise
          
          // OffscreenCanvas 转 Blob 再转 DataURL (JPEG 最高质量，复刻原版)
          const blob = await offscreen.convertToBlob({ type: 'image/jpeg', quality: 1.0 })
          dataURL = await blobToDataURL(blob)
        } else {
          // 回退：使用普通 Canvas（复刻原版）
          const canvas = document.createElement('canvas')
          const context = canvas.getContext('2d')!
          canvas.width = viewport.width
          canvas.height = viewport.height
          
          await page.render({
            canvasContext: context,
            viewport: viewport
          }).promise
          
          // 输出 JPEG 格式（与原版一致）
          dataURL = canvas.toDataURL('image/jpeg', 1.0)
        }
        
        // 文件名格式与原版一致
        const pageName = `${file.name}_页面${pageNum}`
        
        imageStore.addImage(pageName, dataURL)
        processedCount++
        
        console.log(`  页面 ${pageNum}/${numPages} 处理完成`)
      } catch (pageError) {
        console.warn(`PDF ${file.name} 第 ${pageNum} 页渲染失败:`, pageError)
      }
    }
    
    console.log(`PDF ${file.name} 全部 ${numPages} 页处理完成`)
    return processedCount
  } catch (error) {
    console.error('前端 PDF 解析失败:', error)
    showToast('前端 PDF 解析失败，尝试使用后端解析...', 'warning')
    // 回退到后端解析
    return await processPdfBackend(file)
  }
}

/**
 * 后端 PyMuPDF 分批解析 PDF
 * 复刻原版 main.js processPDFFilesBackend 逻辑
 * @param file PDF 文件
 * @returns 处理的图片数量
 */
async function processPdfBackend(file: File): Promise<number> {
  const BATCH_SIZE = 5
  let sessionId: string | null = null
  
  try {
    // 步骤1: 开始解析会话
    showToast(`正在上传 PDF 文件...`, 'info')
    const startResponse = await parsePdfStart(file, BATCH_SIZE)
    
    if (!startResponse.success || !startResponse.session_id) {
      throw new Error(startResponse.error || 'PDF 解析启动失败')
    }
    
    sessionId = startResponse.session_id
    const totalPages = startResponse.total_pages || 0
    
    console.log(`PDF ${file.name} 共 ${totalPages} 页，开始后端分批解析...`)
    showToast(`正在解析 PDF，共 ${totalPages} 页...`, 'info')
    
    let loadedCount = 0
    
    // 步骤2: 分批获取页面（复刻原版的 for 循环方式）
    for (let startIndex = 0; startIndex < totalPages; startIndex += BATCH_SIZE) {
      currentFileName.value = `${file.name} - 处理中 ${Math.min(startIndex + BATCH_SIZE, totalPages)}/${totalPages} 页`
      uploadProgress.value = totalPages > 0 ? Math.round((startIndex / totalPages) * 100) : 0
      
      const batchResponse = await parsePdfBatch(sessionId, startIndex, BATCH_SIZE)
      
      if (!batchResponse.success) {
        console.warn(`批次 ${startIndex} 获取失败:`, batchResponse.error)
        continue
      }
      
      // 处理返回的图片（复刻原版：images 是对象数组 {page_index, data_url}）
      if (batchResponse.images && batchResponse.images.length > 0) {
        for (const imgData of batchResponse.images) {
          if (!imgData || !imgData.data_url) continue
          
          // 文件名格式与原版一致
          const pageName = `${file.name}_页面${String(imgData.page_index + 1).padStart(4, '0')}`
          
          imageStore.addImage(pageName, imgData.data_url)
          loadedCount++
        }
      }
      
      console.log(`  已加载 ${loadedCount}/${totalPages} 页`)
    }
    
    console.log(`PDF ${file.name} 全部 ${loadedCount} 页处理完成`)
    return loadedCount
  } catch (error) {
    console.error('后端 PDF 解析失败:', error)
    throw error
  } finally {
    // 步骤3: 清理会话
    if (sessionId) {
      try {
        await parsePdfCleanup(sessionId)
      } catch (cleanupError) {
        console.warn('PDF 会话清理失败:', cleanupError)
      }
    }
  }
}

/**
 * 处理 MOBI/AZW 文件（后端分批解析）
 * @param file MOBI/AZW 文件
 * @returns 处理的图片数量
 */
async function processMobiFile(file: File): Promise<number> {
  let sessionId: string | null = null
  
  try {
    // 开始解析会话
    showToast(`正在上传电子书文件...`, 'info')
    const startResponse = await parseMobiStart(file, 5)
    
    if (!startResponse.success || !startResponse.session_id) {
      throw new Error(startResponse.error || 'MOBI/AZW 解析启动失败')
    }
    
    sessionId = startResponse.session_id
    const totalImages = startResponse.total_images || 0
    
    showToast(`正在解析电子书，共 ${totalImages} 张图片...`, 'info')
    
    let processedCount = 0
    let hasMore = true
    
    // 分批获取图片
    while (hasMore) {
      currentFileName.value = `${file.name} - 已处理 ${processedCount}/${totalImages} 张`
      uploadProgress.value = totalImages > 0 ? Math.round((processedCount / totalImages) * 100) : 0
      
      const batchResponse = await parseMobiBatch(sessionId)
      
      if (!batchResponse.success) {
        throw new Error(batchResponse.error || 'MOBI/AZW 批次解析失败')
      }
      
      // 处理返回的图片
      if (batchResponse.images && batchResponse.images.length > 0) {
        for (let i = 0; i < batchResponse.images.length; i++) {
          const imageData = batchResponse.images[i]
          if (!imageData) continue
          
          const imageNum = processedCount + i + 1
          const imageName = `${file.name.replace(/\.(mobi|azw|azw3)$/i, '')}_image_${String(imageNum).padStart(3, '0')}.png`
          
          // 确保是完整的 DataURL
          const dataURL = imageData.startsWith('data:') ? imageData : `data:image/png;base64,${imageData}`
          imageStore.addImage(imageName, dataURL)
        }
        processedCount += batchResponse.images.length
      }
      
      hasMore = batchResponse.has_more ?? false
    }
    
    return processedCount
  } catch (error) {
    console.error('MOBI/AZW 解析失败:', error)
    throw error
  } finally {
    // 清理会话
    if (sessionId) {
      try {
        await parseMobiCleanup(sessionId)
      } catch (cleanupError) {
        console.warn('MOBI/AZW 会话清理失败:', cleanupError)
      }
    }
  }
}

/**
 * 清除错误消息
 */
function clearError() {
  errorMessage.value = ''
}

// 暴露方法供父组件调用
defineExpose({
  triggerFileSelect,
  processFiles,
  clearError,
})
</script>

<template>
  <div class="image-upload">
    <!-- 拖拽上传区域 -->
    <div 
      id="drop-area"
      class="drop-area"
      :class="{ 'drag-over': isDragging, 'loading': isLoading }"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @drop="handleDrop"
    >
      <div class="drop-content">
        <p class="drop-text">
          拖拽图片、PDF或MOBI文件到这里，或 
          <span class="select-link" @click="triggerFileSelect">
            点击选择文件
          </span>
        </p>
      </div>
      
      <!-- 隐藏的文件输入框 -->
      <input 
        ref="fileInputRef"
        type="file" 
        id="imageUpload" 
        accept="image/*,application/pdf,.mobi,.azw,.azw3" 
        multiple 
        class="file-input"
        @change="handleFileSelect"
      >
    </div>
    
    <!-- 上传进度条 - 使用 ProgressBar 组件 -->
    <ProgressBar
      v-if="showProgress"
      :visible="true"
      :percentage="uploadProgress"
      :label="currentFileName || '处理中...'"
    />
    
    <!-- 错误消息 -->
    <div v-if="errorMessage" class="error-message" @click="clearError">
      <span class="error-icon">⚠️</span>
      <span class="error-text">{{ errorMessage }}</span>
      <span class="error-close">×</span>
    </div>
    
    <!-- 加载动画 -->
    <div v-if="isLoading && !showProgress" class="loading-overlay">
      <div class="spinner"></div>
      <span class="loading-text">处理中...</span>
    </div>
  </div>
</template>

<style scoped>
/* 图片上传组件样式 - 匹配原版 style.css */
.image-upload {
  position: relative;
  width: 100%;
}

/* 拖拽区域 - 匹配原版 #drop-area 样式 */
.drop-area {
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

.drop-area:hover {
  border-color: #3498db;
  background-color: #ecf5fe;
  transform: translateY(-3px);
}

.drop-area.drag-over {
  border-color: #3498db;
  background-color: #ecf5fe;
  box-shadow: 0 0 15px rgba(52, 152, 219, 0.3);
}

.drop-area.loading {
  pointer-events: none;
  opacity: 0.7;
}

.drop-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.drop-text {
  font-size: 1.1em;
  color: #546e7a;
  margin: 10px 0;
}

.select-link {
  color: #3498db !important;
  cursor: pointer;
  text-decoration: underline;
  font-weight: bold;
  transition: color 0.3s;
}

.select-link:hover {
  color: #2572a4 !important;
}

/* 隐藏的文件输入框 */
.file-input {
  display: none;
}

/* 错误消息 - 匹配原版 .error-message 样式 */
.error-message {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 15px;
  padding: 10px 15px;
  background-color: #fff5f5;
  border-left: 4px solid #fc8181;
  border-radius: 8px;
  color: #c53030;
  font-size: 1em;
  font-weight: bold;
  cursor: pointer;
}

.error-icon {
  flex-shrink: 0;
}

.error-text {
  flex: 1;
}

.error-close {
  flex-shrink: 0;
  font-size: 18px;
  opacity: 0.6;
}

.error-close:hover {
  opacity: 1;
}

/* 加载动画 */
.loading-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--border-color, #e0e0e0);
  border-top-color: var(--primary-color, #4a90d9);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-text {
  font-size: 14px;
  color: var(--text-secondary, #666);
}

/* 暗色主题适配 */
[data-theme="dark"] .drop-area {
  background-color: #2d3748;
  border-color: #4a5568;
  color: #a0aec0;
}

[data-theme="dark"] .drop-area:hover {
  border-color: #4a90d9;
  background-color: rgba(74, 144, 217, 0.1);
}

[data-theme="dark"] .drop-text {
  color: #a0aec0;
}

[data-theme="dark"] .drop-hint {
  color: #718096;
}
</style>
