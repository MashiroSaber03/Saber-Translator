<script setup lang="ts">
import { ref } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { DownloadFormat, TextExportFormat } from '@/composables/useExportImport'
import { IMAGE_EXPORT_FORMAT_OPTIONS } from '@/constants'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'

const downloadFormatOptions = [
  { label: 'ZIP 压缩包', value: 'zip' },
  { label: 'PDF 文档', value: 'pdf' },
  { label: 'CBZ 漫画', value: 'cbz' },
]

const textExportFormatOptions = [
  { label: 'Saber JSON', value: 'json' },
  { label: 'LabelPlus TXT', value: 'labelplus' },
]

defineProps<{
  downloadFormat: DownloadFormat
  downloadProgress: number
  downloadProgressText: string
  hasDownloadableImage: boolean
  hasImages: boolean
  isDownloading: boolean
  isImporting: boolean
  textExportFormat: TextExportFormat
  editModeAvailable?: boolean
}>()

const emit = defineEmits<{
  (e: 'downloadAll'): void
  (e: 'downloadCurrent'): void
  (e: 'downloadPsd'): void
  (e: 'exportText'): void
  (e: 'importText', file: File): void
  (e: 'update:downloadFormat', value: DownloadFormat): void
  (e: 'update:textExportFormat', value: TextExportFormat): void
}>()

const importFileInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const settingsStore = useSettingsStore()
const toast = useToast()
const isSavingImageFormat = ref(false)

async function updateImageFormat(value: string | number): Promise<void> {
  if ((value !== 'jpeg' && value !== 'png') || isSavingImageFormat.value) return
  const previousFormat = settingsStore.exportPreferences.imageFormat ?? 'jpeg'
  settingsStore.exportPreferences.imageFormat = value
  isSavingImageFormat.value = true
  try {
    if (!await settingsStore.saveToBackend()) {
      settingsStore.exportPreferences.imageFormat = previousFormat
      toast.error(settingsStore.backendError || '导出格式保存失败')
    }
  } finally {
    isSavingImageFormat.value = false
  }
}

function updateDownloadFormat(value: string | number): void {
  if (value === 'zip' || value === 'pdf' || value === 'cbz') {
    emit('update:downloadFormat', value)
  }
}

function updateTextExportFormat(value: string | number): void {
  if (value === 'json' || value === 'labelplus') {
    emit('update:textExportFormat', value)
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
    <header class="result-export-actions__header">
      <h3>下载与导出</h3>
    </header>

    <UiProgressBar
      v-if="isDownloading"
      class="result-export-actions__progress"
      :label="downloadProgressText || '下载中，请稍候...'"
      :value="downloadProgress"
    >
      <span class="result-export-actions__progress-label">{{ downloadProgressText || '下载中，请稍候...' }}</span>
    </UiProgressBar>

    <div class="result-export-actions__content">
      <ProductActionRow
        class="result-export-actions__row"
        aria-label="图片文件导出操作"
        justify="start"
      >
        <div class="result-export-actions__field">
          <span>图片格式</span>
          <UiSelect
            :model-value="settingsStore.exportPreferences.imageFormat ?? 'jpeg'"
            :options="IMAGE_EXPORT_FORMAT_OPTIONS"
            :disabled="isDownloading || isSavingImageFormat"
            aria-label="导出图片格式"
            @update:model-value="updateImageFormat"
          />
        </div>
        <div class="result-export-actions__buttons">
          <UiButton
            class="result-export-actions__button"
            size="sm"
            variant="secondary"
            :disabled="!hasDownloadableImage || isDownloading || isImporting || isSavingImageFormat"
            @click="$emit('downloadCurrent')"
          >
            下载当前页
          </UiButton>
          <UiButton
            class="result-export-actions__button"
            size="sm"
            variant="secondary"
            :disabled="!hasImages || isDownloading || isImporting || editModeAvailable === false"
            :title="editModeAvailable === false ? '当前账户未开放编辑模式，无法导出 PSD' : '导出当前页，保留文字及修复背景图层，可在 PS 中继续编辑'"
            @click="emit('downloadPsd')"
          >
            当前页 PSD
          </UiButton>
        </div>
      </ProductActionRow>

      <ProductActionRow
        class="result-export-actions__row"
        aria-label="批量图片导出操作"
        justify="start"
      >
        <div class="result-export-actions__field">
          <span>打包格式</span>
          <UiSelect
            :model-value="downloadFormat"
            :options="downloadFormatOptions"
            :disabled="isDownloading || isImporting"
            aria-label="图片导出格式"
            @update:model-value="updateDownloadFormat"
          />
        </div>
        <div class="result-export-actions__buttons result-export-actions__buttons--single">
          <UiButton
            class="result-export-actions__button"
            size="sm"
            variant="primary"
            :disabled="!hasImages || isDownloading || isImporting || isSavingImageFormat"
            @click="$emit('downloadAll')"
          >
            下载全部
          </UiButton>
        </div>
      </ProductActionRow>
      <p class="result-export-actions__hint">JPG 高质量压缩，PNG 无损保存；均保持原图尺寸。</p>

      <div class="result-export-actions__text">
        <ProductActionRow class="result-export-actions__row" aria-label="文本交换操作" justify="start">
          <div class="result-export-actions__field">
            <span>文本格式</span>
            <UiSelect
              :model-value="textExportFormat"
              :options="textExportFormatOptions"
              :disabled="isDownloading || isImporting"
              aria-label="文本导出格式"
              @update:model-value="updateTextExportFormat"
            />
          </div>
          <div class="result-export-actions__buttons">
            <UiButton
              class="result-export-actions__button"
              size="sm"
              variant="secondary"
              :disabled="!hasImages || isDownloading || isImporting"
              @click="$emit('exportText')"
            >
              导出文本
            </UiButton>
            <UiButton
              class="result-export-actions__button"
              size="sm"
              variant="secondary"
              :disabled="!hasImages || isDownloading || isImporting"
              aria-label="导入 Saber JSON"
              title="导入 Saber JSON 翻译文本"
              @click="triggerImportText"
            >
              {{ isImporting ? '导入中…' : '导入 JSON' }}
            </UiButton>
          </div>
        </ProductActionRow>
        <p class="result-export-actions__hint">Saber JSON 可重新导入；LabelPlus TXT 用于外部嵌字。</p>
      </div>
    </div>

    <UiFileInput
      ref="importFileInput"
      hidden
      accept=".json,application/json"
      @files-change="handleImportFile"
    />
  </section>
</template>

<style scoped>
.result-export-actions {
  --ui-select-min-height: 38px;
  --ui-select-padding: 8px 10px;
  --ui-select-font-size: 0.85rem;

  container: result-export / inline-size;
  box-sizing: border-box;
  width: 100%;
  margin-top: 20px;
  padding: 18px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background-color: var(--color-surface-quiet);
  text-align: left;
}

.result-export-actions__header h3 {
  margin: 0;
  color: var(--color-text-heading);
  font-size: 1.05rem;
  line-height: 1.4;
}

.result-export-actions__hint {
  margin: 0;
  color: var(--color-text-supporting);
  font-size: 0.84rem;
  line-height: 1.55;
}

.result-export-actions__progress {
  margin-top: 14px;
}

.result-export-actions__content {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.result-export-actions__row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 272px;
  gap: 12px;
}

.result-export-actions__buttons {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  min-width: 0;
}

.result-export-actions__buttons--single {
  grid-template-columns: minmax(0, 1fr);
}

.result-export-actions__field {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  align-items: center;
  gap: 8px;
  min-width: 0;
  color: var(--color-text-supporting);
  font-size: 0.84rem;
}

.result-export-actions__text {
  display: grid;
  gap: 10px;
  padding-top: 16px;
  border-top: 1px solid var(--color-border-muted);
}

.result-export-actions__button {
  --ui-button-sm-padding: 8px 10px;
  --ui-button-primary-background: var(--color-action-brand-fill, var(--color-action-brand));
  --ui-button-primary-hover-background: var(--color-action-brand-fill-strong, var(--color-action-brand-strong));
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;

  min-width: 0;
  height: 38px;
}

@container result-export (width < 480px) {
  .result-export-actions__row {
    grid-template-columns: minmax(0, 1fr);
    gap: 8px;
  }
}
</style>
