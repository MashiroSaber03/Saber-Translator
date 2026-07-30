<script setup lang="ts">
import { ref } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { DownloadFormat } from '@/composables/useExportImport'

const downloadFormatOptions = [
  { label: 'ZIP压缩包', value: 'zip' },
  { label: 'PDF文档', value: 'pdf' },
  { label: 'CBZ漫画', value: 'cbz' },
]

defineProps<{
  downloadFormat: DownloadFormat
  downloadProgress: number
  downloadProgressText: string
  hasDownloadableImage: boolean
  hasImages: boolean
  isDownloading: boolean
}>()

const emit = defineEmits<{
  (e: 'downloadAll'): void
  (e: 'downloadCurrent'): void
  (e: 'exportText'): void
  (e: 'importText', file: File): void
  (e: 'update:downloadFormat', value: DownloadFormat): void
}>()

const importFileInput = ref<InstanceType<typeof UiFileInput> | null>(null)

function updateDownloadFormat(value: string | number): void {
  if (value === 'zip' || value === 'pdf' || value === 'cbz') {
    emit('update:downloadFormat', value)
  }
}

function triggerImportText(): void {
  importFileInput.value?.click()
}

function handleImportFile(files: File[]): void {
  const file = files[0]
  if (!file) return

  emit('importText', file)
  importFileInput.value?.clear()
}
</script>

<template>
  <section class="result-export-actions" aria-label="翻译结果导出">
    <UiProgressBar
      v-if="isDownloading"
      :label="downloadProgressText || '下载中，请稍候...'"
      :value="downloadProgress"
    >
      <span class="result-export-actions__progress-label">{{ downloadProgressText || '下载中，请稍候...' }}</span>
    </UiProgressBar>

    <ProductActionRow
      class="result-export-actions__row"
      aria-label="翻译结果导出操作"
      justify="center"
    >
      <UiButton
        variant="primary"
        :disabled="!hasDownloadableImage"
        @click="$emit('downloadCurrent')"
      >
        <UiIcon name="download" />
        下载当前图片
      </UiButton>

      <div class="result-export-actions__download-all">
        <UiButton
          variant="primary"
          :disabled="!hasImages || isDownloading"
          @click="$emit('downloadAll')"
        >
          <UiIcon name="download" />
          下载所有图片
        </UiButton>
        <div class="result-export-actions__format">
          <UiSelect
            :model-value="downloadFormat"
            :options="downloadFormatOptions"
            :disabled="isDownloading"
            size="sm"
            aria-label="下载格式"
            @update:model-value="updateDownloadFormat"
          />
        </div>
      </div>

      <UiButton
        variant="secondary"
        :disabled="!hasImages"
        @click="$emit('exportText')"
      >
        <UiIcon name="file-text" />
        导出文本
      </UiButton>

      <UiButton
        variant="secondary"
        :disabled="!hasImages"
        @click="triggerImportText"
      >
        <UiIcon name="upload" />
        导入文本
      </UiButton>

      <UiFileInput
        ref="importFileInput"
        hidden
        accept=".json"
        @files-change="handleImportFile"
      />
    </ProductActionRow>
  </section>
</template>

<style scoped>
.result-export-actions {
  width: 100%;
  margin-top: 20px;
  padding: 15px;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  background-color: var(--color-surface-quiet);
}

.result-export-actions__row {
  width: 100%;
}

.result-export-actions__download-all {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
}

.result-export-actions__format {
  width: auto;
  max-width: 150px;
}
</style>
