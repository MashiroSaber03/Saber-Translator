<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

import { ref, computed, watch } from 'vue'
import { useInsightStore, type AnalysisMode } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { ApiError } from '@/types'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { triggerBlobDownload } from '@/utils/browserDownload'

const analysisModeOptions = [
  { label: '全书', value: 'full' },
  { label: '章节', value: 'chapter' },
  { label: '单页', value: 'page' }
]

const chapterOptions = computed(() => {
  const options = [{ label: '选择章节...', value: '' }]
  insightStore.chapters.forEach(chapter => {
    options.push({
      label: `${chapter.title} (${chapter.startPage}-${chapter.endPage}页)`,
      value: chapter.id
    })
  })
  return options
})

const emit = defineEmits<{
  (e: 'start-polling'): void
  (e: 'stop-polling'): void
}>()

const insightStore = useInsightStore()

const analysisMode = ref<AnalysisMode>('full')
const selectedChapterId = ref('')
const inputPageNum = ref<number | null>(null)
const isStarting = ref(false)
const errorMessage = ref('')

const statusDotClass = computed(() => {
  const status = insightStore.analysisStatus
  return {
    'analysis-progress-panel__status-dot': true,
    'analysis-progress-panel__status-dot--running': status === 'running',
    'analysis-progress-panel__status-dot--paused': status === 'paused',
    'analysis-progress-panel__status-dot--completed': status === 'completed',
    'analysis-progress-panel__status-dot--failed': status === 'failed'
  }
})

const statusLabel = computed(() => {
  switch (insightStore.analysisStatus) {
    case 'running': return '分析中'
    case 'paused': return '已暂停'
    case 'completed': return '已完成'
    case 'failed': return '分析失败'
    default: return '未分析'
  }
})

const progressText = computed(() => {
  const { current, total } = insightStore.progress
  if (total === 0) return ''
  return `${current}/${total}`
})

const showIdleButtons = computed(() => {
  return insightStore.analysisStatus === 'idle'
    || insightStore.analysisStatus === 'completed'
    || insightStore.analysisStatus === 'failed'
})

const showRunningButtons = computed(() => {
  return insightStore.analysisStatus === 'running'
})

const showPausedButtons = computed(() => {
  return insightStore.analysisStatus === 'paused'
})

const startButtonText = computed(() => {
  return (insightStore.analysisStatus === 'completed' || insightStore.analysisStatus === 'failed')
    ? '重新分析'
    : '开始分析'
})

const showChapterSelect = computed(() => analysisMode.value === 'chapter')

const showPageInput = computed(() => analysisMode.value === 'page')

const canStartAnalysis = computed(() => {
  if (isStarting.value) return false
  if (insightStore.isAnalyzing) return false
  if (analysisMode.value === 'chapter' && !selectedChapterId.value) return false
  if (analysisMode.value === 'page' && !inputPageNum.value) return false
  return true
})

const analysisModeDescription = computed(() => {
  switch (analysisMode.value) {
    case 'full':
      return '全量重跑整本书（会清理旧结果）'
    case 'chapter':
      return '仅分析选中章节的页面'
    case 'page':
      return '仅分析指定的单个页面'
    default:
      return ''
  }
})

const estimatedTime = computed(() => {
  const totalPages = insightStore.totalPageCount
  if (totalPages === 0) return ''

  const pagesPerBatch = insightStore.config.batch.pagesPerBatch || 5
  const batches = Math.ceil(totalPages / pagesPerBatch)
  const seconds = batches * 10

  if (seconds < 60) return `约 ${seconds} 秒`
  const minutes = Math.ceil(seconds / 60)
  return `约 ${minutes} 分钟`
})

const progressPercent = computed(() => Math.max(0, Math.min(100, insightStore.progressPercent)))

function onAnalysisModeChange(): void {
  insightStore.setAnalysisMode(analysisMode.value)
}

function updateAnalysisMode(value: string | number): void {
  analysisMode.value = value as AnalysisMode
  onAnalysisModeChange()
}

function updateSelectedChapter(value: string | number): void {
  selectedChapterId.value = String(value)
}

function getStartErrorMessage(error: unknown): string {
  const apiError = error as Partial<ApiError> | undefined
  if (apiError?.status === 409) {
    return apiError.message || '启动被拒绝：当前书籍已有运行中的任务'
  }
  if (apiError?.message) {
    return apiError.message
  }
  return '启动分析失败'
}

async function startAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return

  if (insightStore.isAnalyzing || isStarting.value) {
    errorMessage.value = '分析正在进行中，请稍候'
    return
  }

  if (analysisMode.value === 'chapter' && !selectedChapterId.value) {
    errorMessage.value = '请选择要分析的章节'
    return
  }
  if (analysisMode.value === 'page' && !inputPageNum.value) {
    errorMessage.value = '请输入要分析的页码'
    return
  }

  isStarting.value = true
  errorMessage.value = ''

  try {
    const options: Parameters<typeof insightApi.startAnalysis>[1] = {}

    if (analysisMode.value === 'chapter' && selectedChapterId.value) {
      options.mode = 'chapters'
      options.chapters = [selectedChapterId.value]
    } else if (analysisMode.value === 'page' && inputPageNum.value) {
      options.mode = 'pages'
      options.pages = [inputPageNum.value]
    } else {
      options.mode = insightStore.incrementalAnalysis ? 'incremental' : 'full'
    }

    const response = await insightApi.startAnalysis(insightStore.currentBookId, options)

    if (response.success) {
      if (response.task_id) {
        insightStore.setCurrentTaskId(response.task_id)
      }
      insightStore.setAnalysisStatus('running')
      emit('start-polling')
    } else {
      errorMessage.value = response.error || '启动分析失败'
    }
  } catch (error) {
    errorMessage.value = getStartErrorMessage(error)
  } finally {
    isStarting.value = false
  }
}

async function pauseAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return
  errorMessage.value = ''

  try {
    const response = await insightApi.pauseAnalysis(
      insightStore.currentBookId,
      insightStore.currentTaskId || undefined
    )
    if (response.success) {
      insightStore.setAnalysisStatus('paused')
    } else {
      errorMessage.value = response.error || '暂停分析失败'
    }
  } catch {
    errorMessage.value = '暂停分析失败'
  }
}

async function resumeAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return
  errorMessage.value = ''

  try {
    const response = await insightApi.resumeAnalysis(
      insightStore.currentBookId,
      insightStore.currentTaskId || undefined
    )
    if (response.success) {
      insightStore.setAnalysisStatus('running')
      emit('start-polling')
    } else {
      errorMessage.value = response.error || '继续分析失败'
    }
  } catch {
    errorMessage.value = '继续分析失败'
  }
}

async function cancelAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return
  const confirmed = await confirmProductAction({
    title: '取消分析',
    message: '确定要取消分析吗？',
    confirmText: '取消分析',
    cancelText: '继续分析',
    tone: 'danger',
  })
  if (!confirmed) return
  errorMessage.value = ''

  try {
    const response = await insightApi.cancelAnalysis(
      insightStore.currentBookId,
      insightStore.currentTaskId || undefined
    )
    if (response.success) {
      insightStore.setAnalysisStatus('idle')
      insightStore.setCurrentTaskId(null)
      emit('stop-polling')
    } else {
      errorMessage.value = response.error || '取消分析失败'
    }
  } catch {
    errorMessage.value = '取消分析失败'
  }
}

async function exportAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) {
    errorMessage.value = '请先选择书籍'
    return
  }

  try {
    const response = await insightApi.exportAnalysis(insightStore.currentBookId)

    if (response.success && response.markdown) {
      const blob = new Blob([response.markdown], { type: 'text/markdown' })
      triggerBlobDownload(blob, `${insightStore.currentBookId}_analysis.md`)
    } else {
      errorMessage.value = response.error || '导出失败'
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导出失败'
  }
}

function clearError(): void {
  errorMessage.value = ''
}

watch(analysisMode, () => {
  clearError()
})
</script>

<template>
  <div class="analysis-progress-panel">
    <div class="analysis-progress-panel__status-bar">
      <div class="analysis-progress-panel__status-main">
        <span :class="statusDotClass"></span>
        <span class="analysis-progress-panel__status-label">{{ statusLabel }}</span>
      </div>
      <div class="analysis-progress-panel__status-meta">
        <span class="analysis-progress-panel__status-progress">{{ progressText }}</span>
      </div>
    </div>

    <UiProgressBar
      v-if="showRunningButtons || showPausedButtons"
      class="analysis-progress-panel__progress"
      :value="progressPercent"
      :max="100"
      label="漫画分析进度"
    >
      <span>{{ progressText || statusLabel }}</span>
    </UiProgressBar>

    <div
      v-if="insightStore.progress.message && (showRunningButtons || showPausedButtons)"
      class="analysis-progress-panel__progress-message"
    >
      {{ insightStore.progress.message }}
    </div>

    <ProductStatusBanner
      v-if="errorMessage"
      class="analysis-progress-panel__error"
      tone="danger"
      aria-live="assertive"
    >
      <span>{{ errorMessage }}</span>
      <template #actions>
        <UiButton
          variant="secondary"
          size="xs"
          aria-label="清除分析错误"
          @click="clearError"
        >
          清除
        </UiButton>
      </template>
    </ProductStatusBanner>

    <div class="analysis-progress-panel__actions">
      <ProductActionRow
        v-if="showIdleButtons"
        class="analysis-progress-panel__action-row"
        aria-label="分析启动操作"
        justify="start"
        variant="toolbar"
      >
        <UiSelect
          class="analysis-progress-panel__mode-select"
          :model-value="analysisMode"
          :options="analysisModeOptions"
          aria-label="选择分析范围"
          size="sm"
          @change="updateAnalysisMode"
        />
        <UiButton
          variant="primary"
          size="sm"
          class="analysis-progress-panel__start-action"
          :disabled="!canStartAnalysis"
          :loading="isStarting"
          :aria-label="startButtonText"
          @click="startAnalysis"
        >
          <UiSpinner v-if="isStarting" />
          <UiIcon v-else name="play" size="16" />
          <span>{{ isStarting ? '启动中...' : startButtonText }}</span>
        </UiButton>
      </ProductActionRow>

      <ProductActionRow
        v-if="showRunningButtons"
        class="analysis-progress-panel__action-row"
        aria-label="运行中的分析操作"
        justify="start"
        variant="toolbar"
      >
        <UiButton
          variant="secondary"
          tone="warning"
          size="sm"
          class="analysis-progress-panel__action-button"
          title="暂停分析"
          aria-label="暂停分析"
          @click="pauseAnalysis"
        >
          <UiIcon name="pause" size="18" />
          <span>暂停</span>
        </UiButton>
        <UiButton
          variant="danger"
          size="sm"
          class="analysis-progress-panel__action-button"
          title="取消分析"
          aria-label="取消分析"
          @click="cancelAnalysis"
        >
          <UiIcon name="square" size="18" />
          <span>取消</span>
        </UiButton>
      </ProductActionRow>

      <ProductActionRow
        v-if="showPausedButtons"
        class="analysis-progress-panel__action-row"
        aria-label="已暂停分析操作"
        justify="start"
        variant="toolbar"
      >
        <UiButton
          variant="primary"
          size="sm"
          class="analysis-progress-panel__action-button"
          title="继续分析"
          aria-label="继续分析"
          @click="resumeAnalysis"
        >
          <UiIcon name="play" size="18" />
          <span>继续</span>
        </UiButton>
        <UiButton
          variant="danger"
          size="sm"
          class="analysis-progress-panel__action-button"
          title="取消分析"
          aria-label="取消分析"
          @click="cancelAnalysis"
        >
          <UiIcon name="square" size="18" />
          <span>取消</span>
        </UiButton>
      </ProductActionRow>
    </div>

    <UiSelect
      v-if="showChapterSelect"
      :model-value="selectedChapterId"
      :options="chapterOptions"
      aria-label="选择分析章节"
      size="sm"
      @change="updateSelectedChapter"
    />

    <div v-if="showPageInput" class="analysis-progress-panel__page-input">
      <UiNumberField
        v-model="inputPageNum"
        class="analysis-progress-panel__page-number"
        nullable
        aria-label="输入页码"
        :min="1"
        :max="insightStore.totalPageCount || undefined"
      />
      <span class="analysis-progress-panel__page-hint">/ {{ insightStore.totalPageCount || '?' }}</span>
    </div>

    <div v-if="showIdleButtons && analysisModeDescription" class="analysis-progress-panel__mode-description">
      {{ analysisModeDescription }}
    </div>

    <div v-if="showIdleButtons && analysisMode === 'full' && estimatedTime" class="analysis-progress-panel__estimated-time">
      <UiIcon name="clock" />
      <span>{{ estimatedTime }}</span>
    </div>

    <ProductActionRow
      class="analysis-progress-panel__options-row"
      aria-label="分析附加操作"
      justify="between"
      variant="toolbar"
    >
      <UiCheckbox
        class="analysis-progress-panel__incremental-checkbox"
        :model-value="insightStore.incrementalAnalysis"
        label="增量模式"
        title="仅分析未分析的页面，跳过已分析的页面"
        @change="insightStore.setIncrementalAnalysis"
      />
      <UiIconButton
        size="xs"
        variant="soft"
        label="导出分析报告"
        title="导出分析报告"
        :disabled="insightStore.analyzedPageCount === 0"
        @click="exportAnalysis"
      >
        <UiIcon name="download" size="14" />
      </UiIconButton>
    </ProductActionRow>
  </div>
</template>

<style scoped>
.analysis-progress-panel {
  --analysis-progress-running-shadow: var(--shadow-action-brand);

  padding: 12px 16px;
}

.analysis-progress-panel__progress-message {
  margin-top: 4px;
  color: var(--insight-text-secondary);
  font-size: 12px;
  text-align: center;
}

.analysis-progress-panel__error {
  margin-top: 8px;
}

.analysis-progress-panel__page-input {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}

.analysis-progress-panel__page-hint {
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.analysis-progress-panel__mode-description {
  margin-top: 6px;
  color: var(--insight-text-secondary);
  font-size: 11px;
  font-style: italic;
}

.analysis-progress-panel__estimated-time {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-top: 4px;
  color: var(--insight-text-secondary);
  font-size: 11px;
}

.analysis-progress-panel__status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.analysis-progress-panel__status-main {
  display: flex;
  align-items: center;
  gap: 8px;
}

.analysis-progress-panel__status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--insight-text-muted);
  transition: background-color 0.3s ease, box-shadow 0.3s ease;
}

.analysis-progress-panel__status-dot--running {
  background: var(--insight-action-primary);
  box-shadow: 0 0 8px var(--analysis-progress-running-shadow);
  animation: pulse-glow 1.5s infinite;
}

.analysis-progress-panel__status-dot--paused {
  background: var(--insight-status-warning);
}

.analysis-progress-panel__status-dot--completed {
  background: var(--insight-status-success);
}

.analysis-progress-panel__status-dot--failed {
  background: var(--insight-status-error);
}

.analysis-progress-panel__status-label {
  color: var(--insight-text-primary);
  font-size: 13px;
  font-weight: 500;
}

.analysis-progress-panel__status-progress {
  color: var(--insight-text-secondary);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.analysis-progress-panel__progress {
  margin-bottom: 10px;
}

.analysis-progress-panel__actions {
  margin-bottom: 10px;
}

.analysis-progress-panel__action-row {
  flex-wrap: nowrap;
  width: 100%;
  margin-bottom: 10px;
}

.analysis-progress-panel__mode-select {
  flex: 0 0 104px;
}

.analysis-progress-panel__start-action {
  flex: 1 1 120px;
}

.analysis-progress-panel__action-button {
  flex: 1 1 0;
}

.analysis-progress-panel__action-row:last-child {
  margin-bottom: 0;
}

.analysis-progress-panel__page-number {
  --ui-number-field-input-width: 100%;

  flex: 1;
  margin-top: 8px;
}

.analysis-progress-panel__options-row {
  width: 100%;
  padding-top: 8px;
  border-top: 1px solid var(--color-border-muted);
}

.analysis-progress-panel__incremental-checkbox {
  color: var(--insight-text-secondary);
  font-size: 12px;
}
</style>
