<script setup lang="ts">

import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
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
      <template #default>
        <span aria-hidden="true" />
      </template>
    </ProductFileDropzone>
    <ProductActionRow
      class="image-upload__drop-title"
      aria-label="其他导入方式"
      justify="center"
    >
      拖拽图片、PDF或MOBI文件到这里，或
      <span class="image-upload__select-link">选择文件</span>
      <span class="image-upload__separator">|</span>
      <UiButton
        class="image-upload__inline-action"
        variant="link"
        size="sm"
        :disabled="isLoading"
        aria-label="选择本地图片文件夹"
        @click.stop="triggerFolderSelect"
      >
        <UiIcon name="folder-open" size="15" />
        <span>选择文件夹</span>
      </UiButton>
      <span class="image-upload__separator" aria-hidden="true">|</span>
      <UiButton
        class="image-upload__inline-action"
        variant="link"
        size="sm"
        :disabled="isLoading"
        aria-label="从网页导入漫画图片"
        @click.stop="triggerWebImport"
      >
        <UiIcon name="globe" size="15" />
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
  --product-file-dropzone-padding: 40px;
  --product-file-dropzone-radius: 12px;
  --product-file-dropzone-background: var(--color-surface-app);
  --product-file-dropzone-background-hover: var(--color-surface-interactive-hover);
  --product-file-dropzone-border: var(--color-border-default);
  --product-file-dropzone-border-hover: var(--color-border-accent);
  --product-file-dropzone-color: var(--color-text-secondary);

  position: relative;
  width: 100%;
}

.image-upload__dropzone {
  width: 85%;
  min-height: 160px;
  margin: 0 auto 15px;
}

.image-upload__drop-title {
  position: absolute;
  top: 80px;
  left: 50%;
  z-index: var(--z-local);
  width: calc(85% - 80px);
  margin: 0;
  color: var(--image-upload-drop-title);
  font-size: 1.1em;
  font-weight: 400;
  line-height: 1.6;
  transform: translate(-50%, -50%);
  pointer-events: none;
}

.image-upload__select-link,
.image-upload__inline-action {
  color: var(--color-action-primary);
  font-weight: 700;
}

.image-upload__inline-action {
  --ui-button-link-color: var(--color-action-primary);
  --ui-button-link-font-size: 1em;
  --ui-button-link-font-weight: 700;
  --ui-button-link-text-decoration: underline;

  display: inline-flex;
  gap: 4px;
  vertical-align: baseline;
  pointer-events: auto;
}

.image-upload__select-link {
  text-decoration: underline;
}

.image-upload__separator {
  margin: 0 4px;
  color: var(--color-border-default);
  font-weight: 400;
}

.image-upload__error-banner {
  width: 100%;
  margin-top: 15px;
}

.image-upload__error-text {
  font-weight: 600;
}

@media (--breakpoint-md-down) {
  .image-upload__dropzone {
    width: 100%;
    min-height: 150px;
  }

  .image-upload__drop-title {
    top: 75px;
    width: calc(100% - 80px);
  }
}
</style>
