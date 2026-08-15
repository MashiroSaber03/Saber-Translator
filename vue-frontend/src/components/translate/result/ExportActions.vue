<script setup lang="ts">
import { ref } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
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
  isImporting: boolean
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
        class="result-export-actions__button result-export-actions__button--primary"
        variant="primary"
        :disabled="!hasDownloadableImage || isDownloading || isImporting"
        @click="$emit('downloadCurrent')"
      >
        下载当前图片
      </UiButton>

      <div class="result-export-actions__download-all">
        <UiButton
          class="result-export-actions__button result-export-actions__button--primary"
          variant="primary"
          :disabled="!hasImages || isDownloading || isImporting"
          @click="$emit('downloadAll')"
        >
          下载所有图片
        </UiButton>
        <div class="result-export-actions__format">
          <UiSelect
            :model-value="downloadFormat"
            :options="downloadFormatOptions"
            :disabled="isDownloading || isImporting"
            size="sm"
            aria-label="下载格式"
            @update:model-value="updateDownloadFormat"
          />
        </div>
      </div>

      <UiButton
        class="result-export-actions__button result-export-actions__button--success"
        variant="primary"
        :disabled="!hasImages || isDownloading || isImporting"
        @click="$emit('exportText')"
      >
        导出文本
      </UiButton>

      <UiButton
        class="result-export-actions__button result-export-actions__button--success"
        variant="primary"
        :disabled="!hasImages || isDownloading || isImporting"
        @click="triggerImportText"
      >
        {{ isImporting ? '导入中…' : '导入文本' }}
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
  gap: 12px;
}

.result-export-actions__button {
  --ui-button-padding: 12px 24px;
  --ui-button-font-size: 0.95em;
  --ui-button-primary-disabled-opacity: 0.5;
  --ui-button-primary-shadow: 0 2px 6px color-mix(in srgb, var(--color-overlay-backdrop-solid) 10%, transparent);
}

.result-export-actions__button--primary {
  --ui-button-primary-background: linear-gradient(135deg, var(--color-action-primary-hover) 0%, var(--color-action-primary) 100%);
  --ui-button-primary-hover-background: linear-gradient(135deg, color-mix(in srgb, var(--color-action-primary-hover) 82%, var(--color-overlay-backdrop-solid)) 0%, var(--color-action-primary) 100%);
  --ui-button-primary-hover-shadow: 0 4px 10px color-mix(in srgb, var(--color-action-primary) 30%, transparent);
}

.result-export-actions__button--success {
  --ui-button-primary-background: linear-gradient(135deg, var(--color-surface-success) 0%, var(--color-action-success-strong) 100%);
  --ui-button-primary-hover-background: linear-gradient(135deg, color-mix(in srgb, var(--color-surface-success) 82%, var(--color-overlay-backdrop-solid)) 0%, var(--color-action-success-strong) 100%);
  --ui-button-primary-hover-shadow: 0 4px 10px color-mix(in srgb, var(--color-surface-success) 30%, transparent);
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
