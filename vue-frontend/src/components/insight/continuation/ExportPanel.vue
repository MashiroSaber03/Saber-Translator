<template>
  <div class="continuation-export-panel">
    <ProductSectionHeader title="导出成品" icon-name="download" />

    <div class="continuation-export-panel__options">
      <div class="continuation-export-panel__summary">
        <p class="continuation-export-panel__summary-text">共生成 <strong class="continuation-export-panel__summary-count">{{ generatedCount }}</strong> 页图片，可导出为以下格式：</p>
      </div>

      <ProductChoiceCardGrid
        class="continuation-export-panel__format-grid"
        aria-label="导出格式"
        :model-value="selectedFormat"
        :items="exportFormatItems"
        @select="handleFormatSelect"
      />

      <UiButton
        variant="primary"
        class="continuation-export-panel__download-action"
        block
        :disabled="isExporting"
        size="lg"
        @click="handleExport"
      >
        <UiIcon v-if="!isExporting" name="download" size="18" />
        <span>{{ isExporting ? '导出中...' : '下载' }}</span>
      </UiButton>

      <ProductActionRow aria-label="导出操作" justify="center">
        <UiButton variant="secondary" @click="clearAndRestart">
          <UiIcon name="trash" size="15" />
          <span>清空并重新开始</span>
        </UiButton>
      </ProductActionRow>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChoiceCardGrid, { type ProductChoiceCardItem } from '@/components/product/ProductChoiceCardGrid.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import { ref } from 'vue'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'
import { confirmProductAction } from '@/composables/useProductConfirm'
import * as continuationApi from '@/api/continuation'
import { triggerBlobDownload } from '@/utils/browserDownload'

const props = defineProps<{
  bookId: string
  generatedCount: number
  state: ContinuationState
}>()

const emit = defineEmits<{
  'clear-and-restart': []
}>()

const state = props.state
type ExportFormat = 'images' | 'pdf'

const exportFormatItems: ProductChoiceCardItem[] = [
  { id: 'images', label: '图片 ZIP', description: '所有页面打包下载', iconName: 'image' },
  { id: 'pdf', label: 'PDF 文档', description: '方便阅读和分享', iconName: 'file-text' },
]

const selectedFormat = ref<ExportFormat>('images')
const isExporting = ref(false)

function handleFormatSelect(formatId: string): void {
  if (formatId !== 'images' && formatId !== 'pdf') return
  selectedFormat.value = formatId
}

async function handleExport() {
  if (!props.bookId || state.pages.value.length === 0) {
    state.showMessage('没有可导出的页面', 'error')
    return
  }

  isExporting.value = true

  try {
    let blob: Blob
    let filename: string

    if (selectedFormat.value === 'images') {
      blob = await continuationApi.exportAsImages(props.bookId)
      filename = `continuation_${Date.now()}.zip`
    } else {
      blob = await continuationApi.exportAsPdf(props.bookId)
      filename = `continuation_${Date.now()}.pdf`
    }

    triggerBlobDownload(blob, filename)

    state.showMessage('导出成功', 'success')
  } catch (error) {
    state.showMessage('导出失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isExporting.value = false
  }
}

async function clearAndRestart() {
  const confirmed = await confirmProductAction({
    title: '清空续写数据',
    message: '确定要清空所有续写数据并重新开始吗？此操作不可恢复。',
    confirmText: '清空',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

  emit('clear-and-restart')
}
</script>

<style scoped>
.continuation-export-panel {
  min-width: 0;
}

.continuation-export-panel__options {
  width: 100%;
}

.continuation-export-panel__summary {
  margin-bottom: 24px;
  text-align: center;
}

.continuation-export-panel__summary-text {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-supporting);
}

.continuation-export-panel__summary-count {
  color: var(--color-text-brand);
  font-size: 20px;
}

.continuation-export-panel__format-grid {
  margin-bottom: 24px;
}

.continuation-export-panel__download-action {
  margin-bottom: 16px;
}

</style>
