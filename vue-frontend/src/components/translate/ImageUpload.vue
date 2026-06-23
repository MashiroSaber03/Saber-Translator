<script setup lang="ts">

import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { showToast } from '@/utils/toast'
import { naturalSort } from '@/utils'
import { useWebImportStore } from '@/stores/webImportStore'
import ProgressBar from '@/components/common/ProgressBar.vue'
import {
  parsePdfStart,
  parsePdfBatch,
  parsePdfCleanup,
  parseMobiStart,
  parseMobiBatch,
  parseMobiCleanup,
} from '@/api/system'
const emit = defineEmits<{
  (e: 'uploadComplete', count: number): void
}>()
const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const webImportStore = useWebImportStore()
const fileInputRef = ref<HTMLInputElement | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)
const isLoading = ref(false)
const isDragging = ref(false)
const errorMessage = ref('')
const uploadProgress = ref(0)
const currentFileName = ref('')
const showProgress = ref(false)
const pdfProcessingMethod = computed(() => settingsStore.settings.pdfProcessingMethod)
function triggerFileSelect() {
  fileInputRef.value?.click()
}
function triggerWebImport() {
  webImportStore.openModal()
}
function triggerFolderSelect() {
  folderInputRef.value?.click()
}
async function handleFolderSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  const allFiles = Array.from(input.files)
  const imageFiles = allFiles.filter(file => file.type.startsWith('image/'))
  if (imageFiles.length === 0) {
    showToast('所选文件夹中没有找到图片文件', 'warning')
    input.value = ''
    return
  }
  const sortedFiles = naturalSort(imageFiles, (file) => file.webkitRelativePath)
  await processFilesWithFolderInfo(sortedFiles)
  input.value = ''
}
async function processFilesWithFolderInfo(files: File[]) {
  if (files.length === 0) return
  isLoading.value = true
  showProgress.value = true
  uploadProgress.value = 0
  try {
    let processedCount = 0
    const totalFiles = files.length
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (!file || !file.type.startsWith('image/')) continue
      currentFileName.value = file.name
      const relativePath = file.webkitRelativePath || ''
      // 提取文件夹路径（去掉文件名）
      const folderPath = relativePath.includes('/')
        ? relativePath.substring(0, relativePath.lastIndexOf('/'))
        : ''
      await new Promise<void>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => {
          const dataURL = e.target?.result as string
          imageStore.addImage(file.name, dataURL, {
            relativePath,
            folderPath
          })
          resolve()
        }
        reader.onerror = () => reject(new Error(`读取图片失败: ${file.name}`))
        reader.readAsDataURL(file)
      })
      processedCount++
      uploadProgress.value = Math.round(((i + 1) / totalFiles) * 100)
    }
    if (processedCount > 0) {
      showToast(`已添加 ${processedCount} 张图片`, 'success')
      emit('uploadComplete', processedCount)
    }
  } catch (error) {
    console.error('处理文件失败:', error)
    const errMsg = error instanceof Error ? error.message : '处理文件失败'
    showToast(errMsg, 'error')
  } finally {
    isLoading.value = false
    showProgress.value = false
  }
}
async function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  await processFiles(Array.from(input.files))
  input.value = ''
}
async function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false
  if (!event.dataTransfer?.files || event.dataTransfer.files.length === 0) return
  await processFiles(Array.from(event.dataTransfer.files))
}
function handleDragOver(event: DragEvent) {
  event.preventDefault()
  isDragging.value = true
}
function handleDragLeave(event: DragEvent) {
  // 检查是否真的离开了拖拽区域（而不是进入子元素）
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const x = event.clientX
  const y = event.clientY
  if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
    isDragging.value = false
  }
}
async function processFiles(files: File[]) {
  if (files.length === 0) return
  isLoading.value = true
  errorMessage.value = ''
  showProgress.value = true
  uploadProgress.value = 0
  try {
    // 业务契约：不在此处预排序，由 TranslateView.handleUploadComplete 统一排序
    let processedCount = 0
    const totalFiles = files.length
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (!file) continue
      currentFileName.value = file.name
      const fileType = file.type
      const fileName = file.name.toLowerCase()
      if (fileType.startsWith('image/')) {
        await processImageFile(file)
        processedCount++
      } else if (fileType === 'application/pdf' || fileName.endsWith('.pdf')) {
        const count = await processPdfFile(file)
        processedCount += count
      } else if (fileName.endsWith('.mobi') || fileName.endsWith('.azw') || fileName.endsWith('.azw3')) {
        const count = await processMobiFile(file)
        processedCount += count
      } else {
        console.warn(`不支持的文件类型: ${file.name}`)
        showToast(`不支持的文件类型: ${file.name}`, 'warning')
      }
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
async function processPdfFile(file: File): Promise<number> {
  if (pdfProcessingMethod.value === 'frontend') {
    // 前端 pdf.js 解析
    return await processPdfFrontend(file)
  } else {
    return await processPdfBackend(file)
  }
}
function blobToDataURL(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}
async function processPdfFrontend(file: File): Promise<number> {
  try {
    // 动态导入 pdf.js
    const pdfjsLib = await import('pdfjs-dist')
    // 设置 worker（使用 CDN）
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`
    const arrayBuffer = await file.arrayBuffer()
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
    const numPages = pdf.numPages
    showToast(`正在解析 PDF，共 ${numPages} 页...`, 'info')
    // 检测是否支持 OffscreenCanvas（后台渲染不受页面可见性影响）
    const useOffscreen = typeof OffscreenCanvas !== 'undefined'
    let processedCount = 0
    for (let pageNum = 1; pageNum <= numPages; pageNum++) {
      currentFileName.value = `${file.name} - 第 ${pageNum}/${numPages} 页`
      uploadProgress.value = Math.round((pageNum / numPages) * 100)
      try {
        const page = await pdf.getPage(pageNum)
        // 设置渲染比例（2.0 可以获得较高清晰度，按业务契约）
        const scale = 2.0
        const viewport = page.getViewport({ scale })
        let dataURL: string
        if (useOffscreen) {
          // 使用 OffscreenCanvas - 后台也能继续渲染（业务契约）
          const offscreen = new OffscreenCanvas(viewport.width, viewport.height)
          const context = offscreen.getContext('2d')
          await page.render({
            canvasContext: context as unknown as CanvasRenderingContext2D,
            viewport: viewport
          }).promise
          // OffscreenCanvas 转 Blob 再转 DataURL (JPEG 最高质量，业务契约)
          const blob = await offscreen.convertToBlob({ type: 'image/jpeg', quality: 1.0 })
          dataURL = await blobToDataURL(blob)
        } else {
          // 回退：使用普通 Canvas（业务契约）
          const canvas = document.createElement('canvas')
          const context = canvas.getContext('2d')!
          canvas.width = viewport.width
          canvas.height = viewport.height
          await page.render({
            canvasContext: context,
            viewport: viewport
          }).promise
          // 输出 JPEG 格式（按业务契约）
          dataURL = canvas.toDataURL('image/jpeg', 1.0)
        }
        const pageName = `${file.name}_页面${pageNum}`
        imageStore.addImage(pageName, dataURL)
        processedCount++
      } catch (pageError) {
        console.warn(`PDF ${file.name} 第 ${pageNum} 页渲染失败:`, pageError)
      }
    }
    return processedCount
  } catch (error) {
    console.error('前端 PDF 解析失败:', error)
    showToast('前端 PDF 解析失败，尝试使用后端解析...', 'warning')
    return await processPdfBackend(file)
  }
}
async function processPdfBackend(file: File): Promise<number> {
  const BATCH_SIZE = 5
  let sessionId: string | null = null
  try {
    showToast(`正在上传 PDF 文件...`, 'info')
    const startResponse = await parsePdfStart(file, BATCH_SIZE)
    if (!startResponse.success || !startResponse.session_id) {
      throw new Error(startResponse.error || 'PDF 解析启动失败')
    }
    sessionId = startResponse.session_id
    const totalPages = startResponse.total_pages || 0
    showToast(`正在解析 PDF，共 ${totalPages} 页...`, 'info')
    let loadedCount = 0
    // 步骤2: 分批获取页面（业务契约的 for 循环方式）
    for (let startIndex = 0; startIndex < totalPages; startIndex += BATCH_SIZE) {
      currentFileName.value = `${file.name} - 处理中 ${Math.min(startIndex + BATCH_SIZE, totalPages)}/${totalPages} 页`
      uploadProgress.value = totalPages > 0 ? Math.round((startIndex / totalPages) * 100) : 0
      const batchResponse = await parsePdfBatch(sessionId, startIndex, BATCH_SIZE)
      if (!batchResponse.success) {
        console.warn(`批次 ${startIndex} 获取失败:`, batchResponse.error)
        continue
      }
      // 处理返回的图片（业务契约：images 是对象数组 {page_index, data_url}）
      if (batchResponse.images && batchResponse.images.length > 0) {
        for (const imgData of batchResponse.images) {
          if (!imgData || !imgData.data_url) continue
          const pageName = `${file.name}_页面${String(imgData.page_index + 1).padStart(4, '0')}`
          imageStore.addImage(pageName, imgData.data_url)
          loadedCount++
        }
      }
    }
    return loadedCount
  } catch (error) {
    console.error('后端 PDF 解析失败:', error)
    throw error
  } finally {
    if (sessionId) {
      try {
        await parsePdfCleanup(sessionId)
      } catch (cleanupError) {
        console.warn('PDF 会话清理失败:', cleanupError)
      }
    }
  }
}
async function processMobiFile(file: File): Promise<number> {
  let sessionId: string | null = null
  try {
    showToast(`正在上传电子书文件...`, 'info')
    const startResponse = await parseMobiStart(file, 5)
    if (!startResponse.success || !startResponse.session_id) {
      throw new Error(startResponse.error || 'MOBI/AZW 解析启动失败')
    }
    sessionId = startResponse.session_id
    const totalImages = startResponse.total_pages || startResponse.total_images || 0
    showToast(`正在解析电子书，共 ${totalImages} 张图片...`, 'info')
    let processedCount = 0
    let hasMore = true
    while (hasMore) {
      currentFileName.value = `${file.name} - 已处理 ${processedCount}/${totalImages} 张`
      uploadProgress.value = totalImages > 0 ? Math.round((processedCount / totalImages) * 100) : 0
      const batchResponse = await parseMobiBatch(sessionId, processedCount, 5)
      if (!batchResponse.success) {
        throw new Error(batchResponse.error || 'MOBI/AZW 批次解析失败')
      }
      if (batchResponse.images && batchResponse.images.length > 0) {
        for (let i = 0; i < batchResponse.images.length; i++) {
          const imageObj = batchResponse.images[i]
          // 后端返回结构：{ success, data_url, width, height, ... }
          if (!imageObj || !imageObj.data_url) continue
          const imageNum = processedCount + i + 1
          const imageName = `${file.name.replace(/\.(mobi|azw|azw3)$/i, '')}_image_${String(imageNum).padStart(3, '0')}.png`
          imageStore.addImage(imageName, imageObj.data_url)
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
    if (sessionId) {
      try {
        await parseMobiCleanup(sessionId)
      } catch (cleanupError) {
        console.warn('MOBI/AZW 会话清理失败:', cleanupError)
      }
    }
  }
}
function clearError() {
  errorMessage.value = ''
}
defineExpose({
  triggerFileSelect,
  triggerFolderSelect,
  processFiles,
  clearError,
})
</script>
<template>
  <div class="image-upload">
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
          <UiButton variant="link" class="select-link" @click="triggerFileSelect">
            选择文件
          </UiButton>
          <span class="separator"> | </span>
          <UiButton variant="link" class="select-link folder-link" @click="triggerFolderSelect">
            📁 选择文件夹
          </UiButton>
          <span class="separator"> | </span>
          <UiButton variant="link" class="select-link web-import-link" @click="triggerWebImport">
            🌐 从网页导入
          </UiButton>
        </p>
      </div>
      <UiFileInput 
        ref="fileInputRef" 
        id="imageUpload" 
        accept="image/*,application/pdf,.mobi,.azw,.azw3" 
        multiple 
        class="file-input"
        @change="handleFileSelect"
      />
      <UiFileInput 
        ref="folderInputRef" 
        webkitdirectory
        class="file-input"
        @change="handleFolderSelect"
      />
    </div>
    <ProgressBar
      v-if="showProgress"
      :visible="true"
      :percentage="uploadProgress"
      :label="currentFileName || '处理中...'"
    />
    <UiButton
      v-if="errorMessage"
      variant="toolbar"
      class="error-message"
      aria-label="关闭上传错误提示"
      @click="clearError"
    >
      <span class="error-icon">⚠️</span>
      <span class="error-text">{{ errorMessage }}</span>
      <span class="error-close">×</span>
    </UiButton>
    <div v-if="isLoading && !showProgress" class="loading-overlay">
      <div class="spinner"></div>
      <span class="loading-text">处理中...</span>
    </div>
  </div>
</template>

<style scoped>
/* 图片上传组件样式 */
.image-upload {
  /* owner tokens: image-upload */
  --image-upload-border-default: #b0bec5;
  --image-upload-border-strong: #fc8181;
  --image-upload-shadow-default: rgba(52, 152, 219, .3);
  --image-upload-surface-base: #f7fafc;
  --image-upload-surface-raised: #ecf5fe;
  --image-upload-text-primary: #546e7a;
  --image-upload-text-secondary: #2572a4;
  --image-upload-text-muted: #b0bec5;
  --image-upload-text-subtle: #c53030;

  position: relative;
  width: 100%;
}
/* 拖拽上传区域 */
.drop-area {
  border: 2px dashed var(--image-upload-border-default);
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  color: var(--image-upload-text-primary);
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
  background-color: var(--image-upload-surface-base);
}

.drop-area:hover {
  border-color: var(--color-border-accent);
  background-color: var(--image-upload-surface-raised);
  transform: translateY(-3px);
}

.drop-area.drag-over {
  border-color: var(--color-border-accent);
  background-color: var(--image-upload-surface-raised);
  box-shadow: 0 0 15px var(--image-upload-shadow-default);
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
  color: var(--image-upload-text-primary);
  margin: 10px 0;
}

.select-link {
  display: inline-flex;
  align-items: center;
  border: 0;
  background: transparent;
  color: var(--color-text-link);
  cursor: pointer;
  font: inherit;
  text-decoration: underline;
  font-weight: bold;
  transition: color 0.3s;
}

.select-link:hover {
  color: var(--image-upload-text-secondary);
}

.separator {
  margin: 0 4px;
  color: var(--image-upload-text-muted);
}

.web-import-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.folder-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
/* 隐藏的文件输入框 */
.file-input {
  display: none;
}
/* 上传错误消息 */
.error-message {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  margin-top: 15px;
  padding: 10px 15px;
  border: 0;
  background-color: var(--color-surface-neutral-soft);
  border-left: 4px solid var(--image-upload-border-strong);
  border-radius: 8px;
  color: var(--image-upload-text-subtle);
  font-size: 1em;
  font-weight: bold;
  text-align: left;
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
  border: 3px solid var(--color-border-muted, var(--color-border-default));
  border-top-color: var(--color-action-primary, var(--color-border-info));
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
</style>
