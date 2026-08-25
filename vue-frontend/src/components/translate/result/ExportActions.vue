<script setup lang="ts">
import { ref } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { DownloadFormat, TextExportFormat } from '@/composables/useExportImport'

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
}>()

const emit = defineEmits<{
  (e: 'downloadAll'): void
  (e: 'downloadCurrent'): void
  (e: 'exportText'): void
  (e: 'importText', file: File): void
  (e: 'update:downloadFormat', value: DownloadFormat): void
  (e: 'update:textExportFormat', value: TextExportFormat): void
}>()

const importFileInput = ref<InstanceType<typeof UiFileInput> | null>(null)

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
      <h3>导出与交换</h3>
      <p>下载处理后的图片，或导出可交接的翻译文本。</p>
    </header>

    <UiProgressBar
      v-if="isDownloading"
      class="result-export-actions__progress"
      :label="downloadProgressText || '下载中，请稍候...'"
      :value="downloadProgress"
    >
      <span class="result-export-actions__progress-label">{{ downloadProgressText || '下载中，请稍候...' }}</span>
    </UiProgressBar>

    <div class="result-export-actions__grid">
      <article class="result-export-actions__card">
        <div class="result-export-actions__card-heading">
          <h4>图片文件</h4>
          <p>下载当前页，或把本章节图片统一打包。</p>
        </div>
        <ProductActionRow
          class="result-export-actions__row"
          aria-label="图片文件导出操作"
          justify="start"
        >
          <UiButton
            class="result-export-actions__button"
            variant="secondary"
            :disabled="!hasDownloadableImage || isDownloading || isImporting"
            @click="$emit('downloadCurrent')"
          >
            下载当前页
          </UiButton>
          <div class="result-export-actions__format-action">
            <div class="result-export-actions__select">
              <UiSelect
                :model-value="downloadFormat"
                :options="downloadFormatOptions"
                :disabled="isDownloading || isImporting"
                size="sm"
                aria-label="图片导出格式"
                @update:model-value="updateDownloadFormat"
              />
            </div>
            <UiButton
              class="result-export-actions__button"
              variant="primary"
              :disabled="!hasImages || isDownloading || isImporting"
              @click="$emit('downloadAll')"
            >
              下载全部
            </UiButton>
          </div>
        </ProductActionRow>
      </article>

      <article class="result-export-actions__card">
        <div class="result-export-actions__card-heading">
          <h4>翻译文本</h4>
          <p>Saber JSON 可重新导入；LabelPlus TXT 用于外部嵌字。</p>
        </div>
        <ProductActionRow
          class="result-export-actions__row"
          aria-label="文本交换操作"
          justify="start"
        >
          <div class="result-export-actions__format-action">
            <div class="result-export-actions__select">
              <UiSelect
                :model-value="textExportFormat"
                :options="textExportFormatOptions"
                :disabled="isDownloading || isImporting"
                size="sm"
                aria-label="文本导出格式"
                @update:model-value="updateTextExportFormat"
              />
            </div>
            <UiButton
              class="result-export-actions__button"
              variant="primary"
              :disabled="!hasImages || isDownloading || isImporting"
              @click="$emit('exportText')"
            >
              导出文本
            </UiButton>
          </div>
          <UiButton
            class="result-export-actions__button"
            variant="secondary"
            :disabled="!hasImages || isDownloading || isImporting"
            @click="triggerImportText"
          >
            {{ isImporting ? '导入中…' : '导入 Saber JSON' }}
          </UiButton>
        </ProductActionRow>
      </article>
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
  width: 100%;
  margin-top: 20px;
  padding: 18px;
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background-color: var(--color-surface-quiet);
  text-align: left;
}

.result-export-actions__header h3,
.result-export-actions__card-heading h4,
.result-export-actions__header p,
.result-export-actions__card-heading p {
  margin: 0;
}

.result-export-actions__header h3 {
  color: var(--color-text-heading);
  font-size: 1.05rem;
  line-height: 1.4;
}

.result-export-actions__header p,
.result-export-actions__card-heading p {
  margin-top: 4px;
  color: var(--color-text-supporting);
  font-size: 0.84rem;
  line-height: 1.55;
}

.result-export-actions__progress {
  margin-top: 14px;
}

.result-export-actions__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 14px;
}

.result-export-actions__card {
  min-width: 0;
  padding: 16px;
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  background: var(--color-surface-card);
}

.result-export-actions__card-heading h4 {
  color: var(--color-text-heading);
  font-size: 0.95rem;
  line-height: 1.4;
}

.result-export-actions__row {
  width: 100%;
  margin-top: 14px;
  gap: 10px;
}

.result-export-actions__format-action {
  display: grid;
  grid-template-columns: minmax(132px, 1fr) auto;
  flex: 1 1 260px;
  gap: 10px;
  min-width: 0;
}

.result-export-actions__select {
  min-width: 0;
}

.result-export-actions__button {
  --ui-button-padding: 9px 15px;
  --ui-button-font-size: 0.88rem;
  --ui-button-primary-background: var(--color-action-brand);
  --ui-button-primary-hover-background: var(--color-action-brand-strong);
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;
}

@media (--breakpoint-lg-down) {
  .result-export-actions__grid {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (--breakpoint-sm-down) {
  .result-export-actions {
    padding: 14px;
  }

  .result-export-actions__card {
    padding: 14px;
  }

  .result-export-actions__format-action {
    grid-template-columns: minmax(0, 1fr);
    flex-basis: 100%;
  }

  .result-export-actions__button {
    --ui-button-width: 100%;
  }
}
</style>
