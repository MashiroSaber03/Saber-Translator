<script setup lang="ts">
import DetectedTextPanel, { type DetectedTextItem } from '@/components/translate/DetectedTextPanel.vue'
import ExportActions from '@/components/translate/result/ExportActions.vue'
import ResultImageCanvas from '@/components/translate/result/ResultImageCanvas.vue'
import ResultToolbar from '@/components/translate/result/ResultToolbar.vue'
import { useExportImport, type DownloadFormat } from '@/composables/useExportImport'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { computed, ref } from 'vue'

interface Props {
  isEditMode?: boolean
}

withDefaults(defineProps<Props>(), {
  isEditMode: false,
})

const emit = defineEmits<{
  (e: 'toggle-edit-mode'): void
  (e: 'retry-failed'): void
}>()

const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const exportImport = useExportImport()

const imageSize = ref(100)

const showOriginal = computed({
  get: () => currentImage.value?.showOriginal ?? false,
  set: (val: boolean) => {
    if (currentImage.value) {
      imageStore.updateCurrentImage({ showOriginal: val })
    }
  },
})

const downloadFormat = ref<DownloadFormat>('zip')

const isDownloading = computed(() => exportImport.isDownloading.value)
const downloadProgressText = computed(() => exportImport.downloadProgressText.value)
const downloadProgress = computed(() => exportImport.downloadProgress.value)
const hasImages = computed(() => imageStore.hasImages)

const currentImage = computed(() => imageStore.currentImage)

const hasTranslatedImage = computed(() => !!currentImage.value?.translatedDataURL)

const hasDownloadableImage = computed(
  () => !!(currentImage.value?.translatedDataURL || currentImage.value?.originalDataURL)
)

const displayImageUrl = computed(() => {
  if (!currentImage.value) return ''
  if (showOriginal.value || !currentImage.value.translatedDataURL) {
    return currentImage.value.originalDataURL
  }
  return currentImage.value.translatedDataURL
})

const hasFailedImages = computed(() => imageStore.failedImageCount > 0)
const failedImageCount = computed(() => imageStore.failedImageCount)

const displayImageAlt = computed(() => {
  const fileName = currentImage.value?.fileName || currentImage.value?.name || '当前图片'
  return showOriginal.value || !currentImage.value?.translatedDataURL
    ? `原图：${fileName}`
    : `翻译图：${fileName}`
})

const useTextboxPrompt = computed(() => settingsStore.settings.useTextboxPrompt)

const detectedTexts = computed<DetectedTextItem[]>(() => {
  if (!currentImage.value) return []

  if (currentImage.value.bubbleStates && currentImage.value.bubbleStates.length > 0) {
    return currentImage.value.bubbleStates.map(state => ({
      original: state.originalText || '',
      translated: useTextboxPrompt.value
        ? state.textboxText || state.translatedText || ''
        : state.translatedText || '',
    }))
  }

  const originalTexts = currentImage.value.originalTexts || []
  const translatedTexts = useTextboxPrompt.value
    ? currentImage.value.textboxTexts || currentImage.value.bubbleTexts || []
    : currentImage.value.bubbleTexts || []

  if (originalTexts.length === 0) return []

  return originalTexts.map((original, index) => ({
    original: original || '',
    translated: translatedTexts[index] || '',
  }))
})

function toggleImageView(): void {
  showOriginal.value = !showOriginal.value
}

function toggleEditMode(): void {
  emit('toggle-edit-mode')
}

function updateImageSize(value: string | number | boolean): void {
  const nextSize = parseInt(String(value), 10)
  if (Number.isNaN(nextSize)) return
  imageSize.value = Math.min(200, Math.max(50, nextSize))
}

function updateDownloadFormat(value: string | number): void {
  if (value === 'zip' || value === 'pdf' || value === 'cbz') {
    downloadFormat.value = value
  }
}

function retryFailed(): void {
  emit('retry-failed')
}

function handleDownloadCurrent(): void {
  exportImport.downloadCurrentImage()
}

function handleDownloadAll(): void {
  exportImport.downloadAllImages(downloadFormat.value)
}

function handleExportText(): void {
  exportImport.exportText()
}

function handleImportText(file: File): void {
  void exportImport.importText(file)
}
</script>

<template>
  <section
    v-if="currentImage"
    class="image-result-display"
    data-testid="translation-result-display"
  >
    <ResultToolbar
      :failed-image-count="failedImageCount"
      :has-failed-images="hasFailedImages"
      :has-translated-image="hasTranslatedImage"
      :image-size="imageSize"
      :is-edit-mode="isEditMode"
      :show-original="showOriginal"
      @retry-failed="retryFailed"
      @toggle-edit-mode="toggleEditMode"
      @toggle-image-view="toggleImageView"
      @update-image-size="updateImageSize"
    />

    <ResultImageCanvas
      :image-alt="displayImageAlt"
      :image-size="imageSize"
      :image-url="displayImageUrl"
    />

    <DetectedTextPanel :items="detectedTexts" />

    <ExportActions
      :download-format="downloadFormat"
      :download-progress="downloadProgress"
      :download-progress-text="downloadProgressText"
      :has-downloadable-image="hasDownloadableImage"
      :has-images="hasImages"
      :is-downloading="isDownloading"
      @download-all="handleDownloadAll"
      @download-current="handleDownloadCurrent"
      @export-text="handleExportText"
      @import-text="handleImportText"
      @update:download-format="updateDownloadFormat"
    />
  </section>
</template>

<style scoped>
.image-result-display {
  /* owner tokens: image-result-display */
  --image-result-display-panel-shadow: var(--shadow-soft);

  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--color-surface-card);
  border-radius: 12px;
  box-shadow: 0 4px 12px var(--image-result-display-panel-shadow);
  padding: 25px;
  text-align: center;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

.image-result-display:hover {
  box-shadow: var(--shadow-lg);
}
</style>
