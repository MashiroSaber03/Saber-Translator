<script setup lang="ts">

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import { ref } from 'vue'
import { showToast } from '@/utils/toast'
import { useWebImportStore } from '@/stores/webImportStore'
import {
  createContainerImportJob,
  importImagesSequentially,
} from '@/api/v2/content'

const props = defineProps<{
  chapterId: string | null
}>()
const emit = defineEmits<{
  (e: 'uploadComplete', count: number): void
}>()
const webImportStore = useWebImportStore()
const folderInputRef = ref<InstanceType<typeof UiFileInput> | null>(null)
const isLoading = ref(false)
const errorMessage = ref('')
const uploadProgress = ref(0)
const currentFileName = ref('')
const showProgress = ref(false)
const CONTAINER_SUFFIXES = new Set(['.pdf', '.zip', '.cbz', '.mobi', '.azw', '.azw3'])

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
    await importImageFiles(imageFiles)
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

function fileSuffix(file: File): string {
  const index = file.name.lastIndexOf('.')
  return index >= 0 ? file.name.slice(index).toLowerCase() : ''
}

async function importImageFiles(files: File[]): Promise<number> {
  if (!props.chapterId || files.length === 0) return 0
  const results = await importImagesSequentially(props.chapterId, files, {
    onProgress: state => {
      currentFileName.value = state.currentPath
      uploadProgress.value = Math.round(state.completed / state.total * 100)
    },
  })
  return results.length
}

async function processFiles(files: File[]) {
  if (files.length === 0) return
  if (!props.chapterId) {
    showToast('后端章节尚未初始化，请稍后重试', 'error')
    return
  }
  isLoading.value = true
  errorMessage.value = ''
  showProgress.value = true
  uploadProgress.value = 0
  try {
    const images = files.filter(file => file.type.startsWith('image/'))
    const containers = files.filter(file => CONTAINER_SUFFIXES.has(fileSuffix(file)))
    const unsupported = files.filter(
      file => !file.type.startsWith('image/') && !CONTAINER_SUFFIXES.has(fileSuffix(file)),
    )
    for (const file of unsupported) {
      showToast(`不支持的文件类型: ${file.name}`, 'warning')
    }

    let importedCount = await importImageFiles(images)
    for (const [index, file] of containers.entries()) {
      currentFileName.value = `上传到后端任务：${file.name}`
      await createContainerImportJob(props.chapterId, file)
      uploadProgress.value = Math.round((index + 1) / containers.length * 100)
    }

    if (importedCount > 0) {
      showToast(`已写入后端 ${importedCount} 张图片`, 'success')
      emit('uploadComplete', importedCount)
    }
    if (containers.length > 0) {
      showToast(
        `已创建 ${containers.length} 个后端解析任务，可安全关闭页面`,
        'success',
      )
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
      accept="image/*,application/pdf,.zip,.cbz,.mobi,.azw,.azw3"
      multiple
      :disabled="isLoading"
      @select="handleFileSelect"
    >
      <template #default="{ isDragging: dropzoneDragging }">
        <div class="image-upload__drop-content">
          <UiIcon name="upload" class="image-upload__drop-icon" size="30" />
          <p class="image-upload__drop-title">
            {{ dropzoneDragging ? '释放文件开始导入' : '拖拽图片、PDF、CBZ 或电子书到这里' }}
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
