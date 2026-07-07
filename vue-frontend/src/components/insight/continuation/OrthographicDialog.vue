<template>
  <ContinuationDialogShell
    :title="dialogTitle"
    width-variant="wide"
    @close="close"
  >
    <div class="orthographic-dialog__body">
      <div class="orthographic-dialog__upload-section">
        <ProductFileDropzone
          input-id="orthographicSourceImages"
          class="orthographic-dialog__dropzone"
          :label="`上传 ${characterName} ${formName} 三视图源图`"
          accept="image/*"
          multiple
          @select="selectImages"
        >
          <template #default="{ isDragging }">
            <div class="orthographic-dialog__upload-placeholder">
              <UiIcon
                class="orthographic-dialog__upload-icon"
                :name="isDragging ? 'upload' : 'folder-open'"
                size="48"
                stroke-width="1.5"
              />
              <p v-if="isDragging" class="orthographic-dialog__upload-placeholder-message">释放以上传图片</p>
              <p v-else class="orthographic-dialog__upload-placeholder-message">点击选择或拖拽角色图片（1-5张）</p>
              <p class="orthographic-dialog__upload-hint">可上传多张图片帮助AI理解角色特征</p>
            </div>
          </template>
        </ProductFileDropzone>

        <ProductThumbnailGrid
          v-if="sourceImagePreviewItems.length > 0"
          class="orthographic-dialog__source-grid"
          aria-label="三视图源图预览"
          :items="sourceImagePreviewItems"
        />
      </div>

      <ProductStatusBanner
        v-if="isGenerating"
        class="orthographic-dialog__generating-state"
        tone="info"
        aria-live="polite"
        title="生成中"
        icon-name="palette"
      >
        <div class="orthographic-dialog__generating-content">
          <UiSpinner :decorative="false" label="三视图生成中" size="32" />
          <p class="orthographic-dialog__progress-message">{{ progressMessage }}</p>
        </div>
        <p class="orthographic-dialog__progress-tip">
          <UiIcon name="clock" size="14" />
          <span>AI 生成通常需要 30-60 秒</span>
        </p>
      </ProductStatusBanner>

      <ProductRecordCard v-else-if="resultImagePath" class="orthographic-dialog__result">
        <template #meta>
          <strong>生成结果</strong>
        </template>
        <div class="orthographic-dialog__result-preview">
          <img
            class="orthographic-dialog__result-image"
            :src="getResultUrl()"
            :alt="resultImageAlt"
          >
        </div>
      </ProductRecordCard>
    </div>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          v-if="!resultImagePath"
          variant="primary"
          :disabled="sourceImages.length === 0 || isGenerating"
          @click="generate"
        >
          <UiIcon v-if="!isGenerating" name="palette" size="15" />
          <span>{{ isGenerating ? '生成中...' : '生成三视图' }}</span>
        </UiButton>
        <template v-else>
          <UiButton variant="secondary" @click="generate">重新生成</UiButton>
          <UiButton variant="primary" @click="useResult">
            <UiIcon name="check" size="15" />
            <span>使用三视图</span>
          </UiButton>
        </template>
      </ContinuationDialogActions>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const props = defineProps<{
  characterName: string
  formId: string
  formName: string
  bookId: string
  isGenerating: boolean
  resultImagePath: string | null
}>()

const emit = defineEmits<{
  close: []
  generate: [sourceImages: File[]]
  'use-result': [imagePath: string]
}>()

const sourceImages = ref<File[]>([])
const sourceImagePreviews = ref<Array<{ url: string }>>([])
const progressMessage = ref('')
let progressTimers: Array<ReturnType<typeof setTimeout>> = []
const close = () => emit('close')
const dialogTitle = computed(() => {
  const suffix = props.formName && props.formName !== '默认' ? ` (${props.formName})` : ''
  return `生成三视图 - ${props.characterName}${suffix}`
})
const resultImageAlt = computed(() => {
  const formSuffix = props.formName && props.formName !== '默认' ? `（${props.formName}）` : ''
  return `${props.characterName}${formSuffix}三视图生成结果`
})
const sourceImagePreviewItems = computed<ProductThumbnailGridItem[]>(() => {
  return sourceImagePreviews.value.map((preview, index) => {
    const number = index + 1

    return {
      id: preview.url,
      alt: `源图${number}`,
      cornerLabel: String(number),
      interactive: false,
      label: `源图 ${number}`,
      src: preview.url,
    }
  })
})

function revokeSourceImagePreviews(): void {
  sourceImagePreviews.value.forEach(preview => {
    window.URL.revokeObjectURL(preview.url)
  })
  sourceImagePreviews.value = []
}

function setSourceImages(files: File[]): void {
  revokeSourceImagePreviews()
  sourceImages.value = files
  sourceImagePreviews.value = files.map(file => ({
    url: window.URL.createObjectURL(file),
  }))
}

function clearProgressTimers(): void {
  progressTimers.forEach(timer => clearTimeout(timer))
  progressTimers = []
}

function scheduleProgressMessage(delay: number, message: string): void {
  const timer = setTimeout(() => {
    progressTimers = progressTimers.filter(item => item !== timer)
    if (props.isGenerating) {
      progressMessage.value = message
    }
  }, delay)
  progressTimers.push(timer)
}

function selectImages(selectedFiles: File[]) {
  const files = selectedFiles
    .filter(file => file.type.startsWith('image/'))
    .slice(0, 5)
  setSourceImages(files)
}

async function generate() {
  if (sourceImages.value.length === 0) return

  progressMessage.value = `正在上传 ${sourceImages.value.length} 张图片...`

  clearProgressTimers()
  scheduleProgressMessage(500, 'AI 正在分析角色特征...')
  scheduleProgressMessage(2000, '正在生成三视图，请耐心等待...')

  emit('generate', sourceImages.value)
}

function useResult() {
  if (props.resultImagePath) {
    emit('use-result', props.resultImagePath)
  }
}

function getResultUrl(): string {
  if (!props.bookId || !props.resultImagePath) return ''
  return `/api/manga-insight/${props.bookId}/continuation/generated-image?path=${encodeURIComponent(props.resultImagePath)}`
}

watch(() => props.isGenerating, (generating) => {
  if (!generating) {
    clearProgressTimers()
    return
  }
  if (!progressMessage.value) {
    progressMessage.value = '正在准备生成三视图...'
  }
})

watch(() => props.resultImagePath, (imagePath) => {
  if (imagePath) {
    clearProgressTimers()
  }
})

onBeforeUnmount(() => {
  clearProgressTimers()
  revokeSourceImagePreviews()
})
</script>

<style scoped>
.orthographic-dialog__body {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.orthographic-dialog__upload-section {
  min-width: 0;
}

.orthographic-dialog__dropzone {
  --product-file-dropzone-padding: 40px 20px;
  --product-file-dropzone-radius: 12px;
  --product-file-dropzone-background-hover: var(--color-focus-brand-soft);
}

.orthographic-dialog__upload-placeholder {
  pointer-events: none;
}

.orthographic-dialog__upload-icon {
  display: block;
  margin-right: auto;
  margin-left: auto;
  margin-bottom: 12px;
}

.orthographic-dialog__upload-placeholder-message {
  margin: 8px 0;
  font-size: 14px;
}

.orthographic-dialog__upload-hint {
  margin: 8px 0;
  color: var(--color-text-supporting);
  font-size: 12px;
}

.orthographic-dialog__source-grid {
  --product-thumbnail-grid-min-size: 100px;
  --product-thumbnail-grid-aspect-ratio: 1;

  margin-top: 16px;
}

.orthographic-dialog__generating-state {
  align-items: center;
}

.orthographic-dialog__generating-content {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.orthographic-dialog__progress-message {
  margin: 0;
}

.orthographic-dialog__progress-tip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 8px 0 0;
  color: var(--color-text-supporting);
  font-size: 14px;
}

.orthographic-dialog__result {
  --product-record-card-background: var(--color-surface-subtle);
  --product-record-card-border: var(--color-border-muted);
  --product-record-card-padding: 16px;
  --product-record-card-gap: 12px;
  --product-record-card-shadow-hover: none;
}

.orthographic-dialog__result-preview {
  overflow: hidden;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
}

.orthographic-dialog__result-image {
  display: block;
  width: 100%;
}
</style>
