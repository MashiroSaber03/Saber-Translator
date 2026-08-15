<template>
  <div class="continuation-export-panel">
    <ProductSectionHeader title="导出成品" icon-name="download" />

    <div class="continuation-export-panel__options">
      <div class="continuation-export-panel__summary">
        <p class="continuation-export-panel__summary-text">
          共生成
          <strong class="continuation-export-panel__summary-count">{{ generatedCount }}</strong>
          页图片，可导出为以下格式：
        </p>
      </div>

      <ProductChoiceCardGrid
        class="continuation-export-panel__format-grid"
        accessibility-label="导出格式"
        :model-value="selectedFormat"
        :items="exportFormatItems"
        @select="handleFormatSelect"
      />

      <UiButton
        variant="primary"
        class="continuation-export-panel__download-action"
        block
        :disabled="isExporting || generatedCount <= 0"
        size="lg"
        @click="handleExport"
      >
        <UiIcon v-if="!isExporting" name="download" size="18" />
        <span>{{ isExporting ? '导出中...' : '下载' }}</span>
      </UiButton>

      <ProductActionRow aria-label="导出操作" justify="center">
        <UiButton variant="secondary" :disabled="isClearing" @click="clearAndRestart">
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
import ProductChoiceCardGrid, {
  type ProductChoiceCardItem,
} from '@/components/product/ProductChoiceCardGrid.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import { onBeforeUnmount, ref, watch } from 'vue'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'
import { confirmProductAction } from '@/composables/useProductConfirm'
import * as continuationApi from '@/api/continuation'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { triggerBlobDownload } from '@/utils/browserDownload'

const props = defineProps<{
  bookId: string
  generatedCount: number
  state: ContinuationState
  isClearing?: boolean
}>()

const emit = defineEmits<{
  'clear-and-restart': [bookId: string]
}>()

const state = props.state
const taskCenterStore = useTaskCenterStore()
type ExportFormat = 'images' | 'pdf'

const exportFormatItems: ProductChoiceCardItem[] = [
  { id: 'images', label: '图片 ZIP', description: '所有页面打包下载', iconName: 'image' },
  { id: 'pdf', label: 'PDF 文档', description: '方便阅读和分享', iconName: 'file-text' },
]

const selectedFormat = ref<ExportFormat>('images')
const isExporting = ref(false)
let exportRequestId = 0
let isMounted = true

function handleFormatSelect(formatId: string): void {
  if (formatId !== 'images' && formatId !== 'pdf') return
  selectedFormat.value = formatId
}

async function handleExport() {
  if (isExporting.value) return
  if (!props.bookId || props.generatedCount <= 0) {
    state.showMessage('没有已生成的图片可导出', 'error')
    return
  }

  const bookId = props.bookId
  const requestId = ++exportRequestId
  const isCurrent = () => isMounted && requestId === exportRequestId && props.bookId === bookId
  isExporting.value = true

  try {
    const format = selectedFormat.value === 'images' ? 'zip' : 'pdf'
    const jobId = await continuationApi.createContinuationExportJob(bookId, format)
    if (!isCurrent()) return
    state.showMessage('续写导出任务已进入任务中心，关闭浏览器也会继续运行', 'info')
    const job = await taskCenterStore.waitForJob(jobId)
    if (!isCurrent()) return
    const assetId = job.artifacts[0]?.assetId
    if (!assetId) throw new Error('导出任务未生成文件')
    const blob = await continuationApi.downloadContinuationExport(assetId, bookId, format)
    if (!isCurrent()) return
    const filename = `continuation_${Date.now()}.${format}`

    triggerBlobDownload(blob, filename)

    state.showMessage('导出成功', 'success')
  } catch (error) {
    if (isCurrent()) {
      state.showMessage('导出失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    }
  } finally {
    if (requestId === exportRequestId) isExporting.value = false
  }
}

async function clearAndRestart() {
  const bookId = props.bookId
  if (!bookId || props.isClearing) return
  const confirmed = await confirmProductAction({
    title: '清空续写数据',
    message: '确定要清空所有续写数据并重新开始吗？此操作不可恢复。',
    confirmText: '清空',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed || props.bookId !== bookId) return

  emit('clear-and-restart', bookId)
}

watch(() => props.bookId, () => {
  exportRequestId += 1
  isExporting.value = false
})

onBeforeUnmount(() => {
  isMounted = false
  exportRequestId += 1
})
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
