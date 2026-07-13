<script setup lang="ts">

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.js?url'
import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { showToast } from '@/utils/toast'
import {
  buildDocumentParseBatches,
  calculateDocumentParseProgress,
  createDocumentPageFileName,
  naturalSort,
} from '@/utils'
import { readBlobAsDataUrl } from '@/utils/dataUrl'
import { useWebImportStore } from '@/stores/webImportStore'
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
const folderInputRef = ref<InstanceType<typeof UiFileInput> | null>(null)
const isLoading = ref(false)
const errorMessage = ref('')
const uploadProgress = ref(0)
const currentFileName = ref('')
const showProgress = ref(false)
const pdfProcessingMethod = computed(() => settingsStore.settings.pdfProcessingMethod)
function triggerWebImport() {
  webImportStore.openModal()
}
function triggerFolderSelect() {
  folderInputRef.value?.click()
}
async function handleFolderSelect(files: File[]) {
  if (files.length === 0) return
  try {
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    if (imageFiles.length === 0) {
      showToast('所选文件夹中没有找到图片文件', 'warning')
      return
    }
    const sortedFiles = naturalSort(imageFiles, (file) => file.webkitRelativePath)
    await processFilesWithFolderInfo(sortedFiles)
  } finally {
    folderInputRef.value?.clear()
  }
}
async function handleFileSelect(files: File[]) {
  if (files.length === 0) {
    return
  }

  await processFiles(files)
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
      const folderPath = relativePath.includes('/')
        ? relativePath.substring(0, relativePath.lastIndexOf('/'))
        : ''
      const dataURL = await readBlobAsDataUrl(file, `读取图片失败: ${file.name}`)
      imageStore.addImage(file.name, dataURL, {
        relativePath,
        folderPath
      })
      processedCount++
      uploadProgress.value = Math.round(((i + 1) / totalFiles) * 100)
    }
    if (processedCount > 0) {
      showToast(`已添加 ${processedCount} 张图片`, 'success')
      emit('uploadComplete', processedCount)
    }
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : '处理文件失败'
    showToast(errMsg, 'error')
  } finally {
    isLoading.value = false
    showProgress.value = false
  }
}
async function processFiles(files: File[]) {
  if (files.length === 0) return
  isLoading.value = true
  errorMessage.value = ''
  showProgress.value = true
  uploadProgress.value = 0
  try {
    // Preserve the import order here; TranslateView applies the final natural sort.
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
        showToast(`不支持的文件类型: ${file.name}`, 'warning')
      }
      uploadProgress.value = Math.round(((i + 1) / totalFiles) * 100)
    }
    if (processedCount > 0) {
      showToast(`已添加 ${processedCount} 张图片`, 'success')
      emit('uploadComplete', processedCount)
    }
  } catch (error) {
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
  const dataURL = await readBlobAsDataUrl(file, `读取图片文件失败: ${file.name}`)
  imageStore.addImage(file.name, dataURL)
}
async function processPdfFile(file: File): Promise<number> {
  if (pdfProcessingMethod.value === 'frontend') {
    return await processPdfFrontend(file)
  } else {
    return await processPdfBackend(file)
  }
}
type PdfCanvasContext = CanvasRenderingContext2D
type BrowserCanvasContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null

function getPdfCanvasContext(context: BrowserCanvasContext): PdfCanvasContext {
  if (!context) {
    throw new Error('无法创建 PDF 渲染上下文')
  }
  return context as PdfCanvasContext
}
async function processPdfFrontend(file: File): Promise<number> {
  try {
    const pdfjsLib = await import('pdfjs-dist')
    pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl
    const arrayBuffer = await file.arrayBuffer()
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
    const numPages = pdf.numPages
    showToast(`正在解析 PDF，共 ${numPages} 页...`, 'info')
    const useOffscreen = typeof OffscreenCanvas !== 'undefined'
    let processedCount = 0
    for (let pageNum = 1; pageNum <= numPages; pageNum++) {
      currentFileName.value = `${file.name} - 第 ${pageNum}/${numPages} 页`
      uploadProgress.value = calculateDocumentParseProgress(pageNum, numPages)
      try {
        const page = await pdf.getPage(pageNum)
        const scale = 2.0
        const viewport = page.getViewport({ scale })
        let dataURL: string
        if (useOffscreen) {
          const offscreen = new OffscreenCanvas(viewport.width, viewport.height)
          const context = getPdfCanvasContext(offscreen.getContext('2d'))
          await page.render({
            canvasContext: context,
            viewport: viewport
          }).promise
          const blob = await offscreen.convertToBlob({ type: 'image/jpeg', quality: 1.0 })
          dataURL = await readBlobAsDataUrl(blob, 'PDF 页面转图片失败')
        } else {
          const canvas = document.createElement('canvas')
          canvas.width = viewport.width
          canvas.height = viewport.height
          const context = getPdfCanvasContext(canvas.getContext('2d'))
          await page.render({
            canvasContext: context,
            viewport: viewport
          }).promise
          dataURL = canvas.toDataURL('image/jpeg', 1.0)
        }
        const pageName = createDocumentPageFileName(file.name, pageNum)
        imageStore.addImage(pageName, dataURL)
        processedCount++
      } catch {
        // A single page can fail without invalidating the rest of the document import.
      }
    }
    return processedCount
  } catch {
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
    for (const batch of buildDocumentParseBatches(totalPages, BATCH_SIZE)) {
      currentFileName.value = `${file.name} - 处理中 ${batch.processedPages}/${totalPages} 页`
      uploadProgress.value = calculateDocumentParseProgress(batch.startIndex, totalPages)
      const batchResponse = await parsePdfBatch(sessionId, batch.startIndex, batch.count)
      if (!batchResponse.success) {
        continue
      }
      if (batchResponse.images && batchResponse.images.length > 0) {
        for (const imgData of batchResponse.images) {
          if (!imgData || !imgData.data_url) continue
          const pageName = createDocumentPageFileName(file.name, imgData.page_index + 1)
          imageStore.addImage(pageName, imgData.data_url)
          loadedCount++
        }
      }
      uploadProgress.value = calculateDocumentParseProgress(batch.processedPages, totalPages)
    }
    return loadedCount
  } finally {
    if (sessionId) {
      try {
        await parsePdfCleanup(sessionId)
      } catch {
        // Temporary backend sessions are cleaned up on a best-effort basis.
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
      uploadProgress.value = calculateDocumentParseProgress(processedCount, totalImages)
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
          const imageName = createDocumentPageFileName(file.name, imageNum)
          imageStore.addImage(imageName, imageObj.data_url)
        }
        processedCount += batchResponse.images.length
        uploadProgress.value = calculateDocumentParseProgress(processedCount, totalImages)
      }
      hasMore = batchResponse.has_more ?? false
    }
    return processedCount
  } finally {
    if (sessionId) {
      try {
        await parseMobiCleanup(sessionId)
      } catch {
        // Temporary backend sessions are cleaned up on a best-effort basis.
      }
    }
  }
}
function clearError() {
  errorMessage.value = ''
}
</script>
<template>
  <div class="image-upload">
    <ProductFileDropzone
      input-id="imageUpload"
      class="image-upload__dropzone"
      label="上传翻译源文件"
      accept="image/*,application/pdf,.mobi,.azw,.azw3"
      multiple
      :disabled="isLoading"
      @select="handleFileSelect"
    >
      <template #default="{ isDragging: dropzoneDragging }">
        <div class="image-upload__drop-content">
          <UiIcon name="upload" class="image-upload__drop-icon" size="30" />
          <p class="image-upload__drop-title">
            {{ dropzoneDragging ? '释放文件开始导入' : '拖拽图片、PDF 或 MOBI 文件到这里' }}
          </p>
          <p class="image-upload__drop-hint">点击此区域选择文件</p>
        </div>
      </template>
    </ProductFileDropzone>
    <ProductActionRow
      class="image-upload__actions"
      aria-label="其他导入方式"
      justify="center"
      variant="toolbar"
    >
      <UiButton
        variant="secondary"
        size="sm"
        :disabled="isLoading"
        aria-label="选择本地图片文件夹"
        @click="triggerFolderSelect"
      >
        <UiIcon name="folder-open" size="16" />
        <span>选择文件夹</span>
      </UiButton>
      <UiButton
        variant="secondary"
        size="sm"
        :disabled="isLoading"
        aria-label="从网页导入漫画图片"
        @click="triggerWebImport"
      >
        <UiIcon name="globe" size="16" />
        <span>从网页导入</span>
      </UiButton>
    </ProductActionRow>
    <UiFileInput
      ref="folderInputRef"
      hidden
      webkitdirectory
      @files-change="handleFolderSelect"
    />
    <UiProgressBar
      v-if="showProgress"
      :label="currentFileName || '处理中...'"
      :value="uploadProgress"
    >
      <span class="image-upload__progress-label">{{ currentFileName || '处理中...' }}</span>
    </UiProgressBar>
    <ProductStatusBanner
      v-if="errorMessage"
      class="image-upload__error-banner"
      tone="danger"
      aria-live="assertive"
    >
      <span class="image-upload__error-text">{{ errorMessage }}</span>
      <template #actions>
        <UiIconButton
          variant="soft"
          size="sm"
          label="关闭上传错误提示"
          @click="clearError"
        >
          <UiIcon name="x" size="14" />
        </UiIconButton>
      </template>
    </ProductStatusBanner>
  </div>
</template>

<style scoped>
.image-upload {
  /* owner tokens: image-upload */
  --image-upload-drop-title: var(--color-text-default);
  --image-upload-drop-hint: var(--color-text-secondary);
  --image-upload-drop-icon: var(--color-action-primary);
  --product-file-dropzone-padding: 38px 24px;
  --product-file-dropzone-radius: 12px;
  --product-file-dropzone-background: var(--color-surface-app);
  --product-file-dropzone-background-hover: var(--color-surface-interactive-hover);
  --product-file-dropzone-border: var(--color-border-muted);
  --product-file-dropzone-border-hover: var(--color-border-accent);
  --product-file-dropzone-color: var(--color-text-secondary);

  position: relative;
  width: 100%;
}

.image-upload__dropzone {
  width: min(100%, 720px);
  min-height: 150px;
  margin: 0 auto 12px;
}

.image-upload__drop-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 86px;
}

.image-upload__drop-icon {
  color: var(--image-upload-drop-icon);
}

.image-upload__drop-title {
  margin: 0;
  color: var(--image-upload-drop-title);
  font-size: 1.02rem;
  font-weight: 700;
}

.image-upload__drop-hint {
  margin: 0;
  color: var(--image-upload-drop-hint);
  font-size: 0.9rem;
}

.image-upload__actions {
  margin-bottom: 15px;
}

.image-upload__error-banner {
  width: 100%;
  margin-top: 15px;
}

.image-upload__error-text {
  font-weight: 600;
}
</style>
