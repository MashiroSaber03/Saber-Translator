<template>
  <ContinuationDialogShell
    :title="dialogTitle"
    custom-class="continuation-dialog-modal--wide"
    @close="close"
  >
    <div class="orthographic-dialog-body">
      <div class="ortho-upload-section">
        <label
          class="upload-area"
          :class="{ 'drag-over': isDragging }"
          @dragenter="handleDragEnter"
          @dragover="handleDragOver"
          @dragleave="handleDragLeave"
          @drop="handleDrop"
        >
          <UiFileInput
            accept="image/*"
            multiple
            hidden
            @change="selectImages"
          />
          <div class="upload-placeholder">
            <span class="upload-icon">{{ isDragging ? '📥' : '📁' }}</span>
            <p v-if="isDragging">释放以上传图片</p>
            <p v-else>点击选择或拖拽角色图片（1-5张）</p>
            <p class="hint">可上传多张图片帮助AI理解角色特征</p>
          </div>
        </label>

        <div v-if="sourceImages.length > 0" class="source-images">
          <div v-for="(file, index) in sourceImages" :key="index" class="source-image">
            <img :src="createObjectURL(file)" :alt="`源图${index + 1}`">
            <span class="image-index">{{ index + 1 }}</span>
          </div>
        </div>
      </div>

      <div v-if="isGenerating" class="generating-state">
        <div class="spinner"></div>
        <p class="progress-message">{{ progressMessage }}</p>
        <p class="progress-tip">⏱️ AI 生成通常需要 30-60 秒</p>
      </div>

      <div v-else-if="resultImagePath" class="ortho-result">
        <h4>生成结果：</h4>
        <div class="result-preview">
          <img :src="getResultUrl()" alt="三视图">
        </div>
      </div>
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
          {{ isGenerating ? '生成中...' : '🎨 生成三视图' }}
        </UiButton>
        <template v-else>
          <UiButton variant="secondary" @click="generate">重新生成</UiButton>
          <UiButton variant="primary" @click="useResult">✓ 使用三视图</UiButton>
        </template>
      </ContinuationDialogActions>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiFileInput from '@/components/ui/UiFileInput.vue'
import { computed, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const props = defineProps<{
  characterName: string
  formId: string
  formName: string
  bookId: string
}>()

const emit = defineEmits<{
  close: []
  generate: [sourceImages: File[]]
  'use-result': [imagePath: string]
}>()

const sourceImages = ref<File[]>([])
const isDragging = ref(false)
const isGenerating = ref(false)
const progressMessage = ref('')
const resultImagePath = ref<string | null>(null)
const close = () => emit('close')
const dialogTitle = computed(() => {
  const suffix = props.formName && props.formName !== '默认' ? ` (${props.formName})` : ''
  return `🎨 生成三视图 - ${props.characterName}${suffix}`
})

function selectImages(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files) return

  const files = Array.from(input.files).slice(0, 5)
  sourceImages.value = files
}

function handleDragEnter(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = true
}

function handleDragOver(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = true
}

function handleDragLeave(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = false
}

function handleDrop(event: DragEvent) {
  event.preventDefault()
  event.stopPropagation()
  isDragging.value = false

  const files = event.dataTransfer?.files
  if (!files || files.length === 0) return

  const imageFiles = Array.from(files)
    .filter(file => file.type.startsWith('image/'))
    .slice(0, 5)

  if (imageFiles.length > 0) {
    sourceImages.value = imageFiles
  }
}

async function generate() {
  if (sourceImages.value.length === 0) return

  isGenerating.value = true
  progressMessage.value = `正在上传 ${sourceImages.value.length} 张图片...`

  setTimeout(() => {
    if (isGenerating.value) {
      progressMessage.value = 'AI 正在分析角色特征...'
    }
  }, 500)

  setTimeout(() => {
    if (isGenerating.value) {
      progressMessage.value = '正在生成三视图，请耐心等待...'
    }
  }, 2000)

  emit('generate', sourceImages.value)
}

function useResult() {
  if (resultImagePath.value) {
    emit('use-result', resultImagePath.value)
  }
}

function createObjectURL(file: File): string {
  return window.URL.createObjectURL(file)
}

function getResultUrl(): string {
  if (!props.bookId || !resultImagePath.value) return ''
  return `/api/manga-insight/${props.bookId}/continuation/generated-image?path=${encodeURIComponent(resultImagePath.value)}`
}

function setResult(imagePath: string) {
  resultImagePath.value = imagePath
  isGenerating.value = false
}

function setGenerating(generating: boolean) {
  isGenerating.value = generating
}

defineExpose({
  setResult,
  setGenerating,
})
</script>

<style scoped>
.orthographic-dialog-body {
  --orthographic-dialog-surface-base: rgba(99, 102, 241, .05);
  --orthographic-dialog-surface-raised: rgba(0, 0, 0, .7);
}

.orthographic-dialog-body {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.upload-area {
  display: block;
  padding: 40px 20px;
  border: 2px dashed var(--color-border-muted, var(--color-border-subtle));
  border-radius: 12px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
}

.upload-area:hover,
.upload-area.drag-over {
  border-color: var(--color-border-brand);
  background: var(--orthographic-dialog-surface-base);
}

.upload-placeholder {
  pointer-events: none;
}

.upload-icon {
  display: block;
  margin-bottom: 12px;
  font-size: 48px;
}

.upload-placeholder p {
  margin: 8px 0;
  font-size: 14px;
}

.hint {
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 12px;
}

.source-images {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.source-image {
  position: relative;
  overflow: hidden;
  aspect-ratio: 1;
  border: 2px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 8px;
}

.source-image img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-index {
  position: absolute;
  top: 4px;
  right: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--orthographic-dialog-surface-raised);
  color: white;
  font-size: 12px;
  font-weight: bold;
}

.generating-state {
  padding: 40px 20px;
  text-align: center;
}

.spinner {
  width: 48px;
  height: 48px;
  margin: 0 auto 16px;
  border: 4px solid var(--color-border-muted, var(--color-border-default));
  border-top-color: var(--color-border-brand);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.progress-tip {
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 14px;
}

.ortho-result h4 {
  margin: 0 0 16px;
  font-size: 16px;
}

.result-preview {
  overflow: hidden;
  border: 1px solid var(--color-border-muted, var(--color-border-default));
  border-radius: 8px;
}

.result-preview img {
  display: block;
  width: 100%;
}
</style>
