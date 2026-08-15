<template>
  <ContinuationDialogShell
    :title="dialogTitle"
    width-variant="wide"
    :dismissible="!isGenerating"
    @close="close"
  >
    <div class="orthographic-dialog__body">
      <div class="orthographic-dialog__upload-section">
        <ProductFileDropzone
          input-id="orthographicSourceImages"
          class="orthographic-dialog__dropzone"
          :label="`上传 ${characterName} ${formName} 三视图源图`"
          accept="image/*"
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
              <p v-else class="orthographic-dialog__upload-placeholder-message">点击选择或拖拽一张角色参考图</p>
              <p class="orthographic-dialog__upload-hint">将使用这张参考图生成角色三视图</p>
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
        <UiButton variant="secondary" :disabled="isGenerating" @click="close">取消</UiButton>
        <UiButton
          v-if="!resultImagePath"
          variant="primary"
          :disabled="!sourceImage || isGenerating"
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
  generate: [sourceImage: File]
  'use-result': [imagePath: string]
}>()

const sourceImage = ref<File | null>(null)
const sourceImagePreviewUrl = ref<string | null>(null)
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
  if (!sourceImagePreviewUrl.value) return []

  return [{
    id: sourceImagePreviewUrl.value,
    alt: '角色参考图',
    interactive: false,
    label: '角色参考图',
    src: sourceImagePreviewUrl.value,
  }]
})

function revokeSourceImagePreviews(): void {
  if (sourceImagePreviewUrl.value) {
    window.URL.revokeObjectURL(sourceImagePreviewUrl.value)
    sourceImagePreviewUrl.value = null
  }
}

function setSourceImage(file: File | null): void {
  revokeSourceImagePreviews()
  sourceImage.value = file
  sourceImagePreviewUrl.value = file ? window.URL.createObjectURL(file) : null
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
  const file = selectedFiles.find(item => item.type.startsWith('image/')) ?? null
  setSourceImage(file)
}

async function generate() {
  const file = sourceImage.value
  if (!file) return

  progressMessage.value = '正在上传参考图...'

  clearProgressTimers()
  scheduleProgressMessage(500, 'AI 正在分析角色特征...')
  scheduleProgressMessage(2000, '正在生成三视图，请耐心等待...')

  emit('generate', file)
}

function useResult() {
  if (props.resultImagePath) {
    emit('use-result', props.resultImagePath)
  }
}

function getResultUrl(): string {
  return props.resultImagePath ?? ''
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
