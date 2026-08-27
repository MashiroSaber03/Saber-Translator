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
import { useInsightStore } from '@/stores/insightStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import * as insightApi from '@/api/insight'
import { NONTERMINAL_JOB_STATUSES } from '@/api/v2/jobs'
import type { ApiError } from '@/types'
import { confirmProductAction } from '@/composables/useProductConfirm'

type AnalysisScope = 'full' | 'chapter' | 'page'

const analysisModeOptions: Array<{ label: string; value: AnalysisScope }> = [
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

const insightStore = useInsightStore()
const taskCenterStore = useTaskCenterStore()

const analysisMode = ref<AnalysisScope>('full')
const incrementalAnalysis = ref(true)
const selectedChapterId = ref('')
const inputPageNum = ref<number | null>(null)
const isStarting = ref(false)
const isControlling = ref(false)
const isExporting = ref(false)
const feedbackMessage = ref('')
const feedbackTone = ref<'danger' | 'info'>('danger')

const statusDotClass = computed(() => {
  const status = insightStore.analysisStatus
  return {
    'analysis-progress-panel__status-dot': true,
    'analysis-progress-panel__status-dot--queued': status === 'queued',
    'analysis-progress-panel__status-dot--running': status === 'running',
    'analysis-progress-panel__status-dot--paused': status === 'paused',
    'analysis-progress-panel__status-dot--interrupted': status === 'interrupted',
    'analysis-progress-panel__status-dot--completed': status === 'completed',
    'analysis-progress-panel__status-dot--completed-with-errors': status === 'completed_with_errors',
    'analysis-progress-panel__status-dot--failed': status === 'failed'
  }
})

const statusLabel = computed(() => {
  switch (insightStore.analysisStatus) {
    case 'queued': return '已排队'
    case 'running': return '分析中'
    case 'paused': return '已暂停'
    case 'interrupted': return '已中断'
    case 'completed': return '已完成'
    case 'completed_with_errors': return '部分完成'
    case 'cancelled': return '已取消'
    case 'failed': return '分析失败'
    default: return insightStore.analyzedPageCount > 0 ? '部分分析' : '未分析'
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
    || insightStore.analysisStatus === 'completed_with_errors'
    || insightStore.analysisStatus === 'cancelled'
    || insightStore.analysisStatus === 'failed'
})

const showRunningButtons = computed(() => {
  return insightStore.analysisStatus === 'running'
})

const showPausedButtons = computed(() => {
  return insightStore.analysisStatus === 'paused'
})

const showQueuedButtons = computed(() => insightStore.analysisStatus === 'queued')
const showInterruptedButtons = computed(() => insightStore.analysisStatus === 'interrupted')
const showProgress = computed(() => (
  insightStore.analysisStatus !== 'idle'
  && NONTERMINAL_JOB_STATUSES.has(insightStore.analysisStatus)
))

const startButtonText = computed(() => {
  return insightStore.analysisStatus === 'idle' || insightStore.analysisStatus === 'cancelled'
    ? '开始分析'
    : '重新分析'
})

const showChapterSelect = computed(() => analysisMode.value === 'chapter')

const showPageInput = computed(() => analysisMode.value === 'page')

const canStartAnalysis = computed(() => {
  if (isStarting.value || isControlling.value) return false
  if (insightStore.isAnalyzing) return false
  if (analysisMode.value === 'chapter' && !selectedChapterId.value) return false
  if (analysisMode.value === 'page' && !inputPageNum.value) return false
  return true
})

const analysisModeDescription = computed(() => {
  switch (analysisMode.value) {
    case 'full':
      return incrementalAnalysis.value
        ? '仅分析尚未完成的页面'
        : '全量重跑整本书（旧结果持续可读）'
    case 'chapter':
      return '仅分析选中章节的页面'
    case 'page':
      return '仅分析指定的单个页面'
    default:
      return ''
  }
})

const progressPercent = computed(() => Math.max(0, Math.min(100, insightStore.progressPercent)))

function updateAnalysisMode(value: string | number): void {
  const nextMode = String(value)
  if (nextMode !== 'full' && nextMode !== 'chapter' && nextMode !== 'page') return
  analysisMode.value = nextMode
}

function updateIncrementalAnalysis(value: boolean): void {
  incrementalAnalysis.value = value
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

function showFeedback(message: string, tone: 'danger' | 'info' = 'danger'): void {
  feedbackMessage.value = message
  feedbackTone.value = tone
}

function commandErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback
}

function isCurrentBook(bookId: string): boolean {
  return insightStore.currentBookId === bookId
}

function isCurrentTask(bookId: string, taskId: string): boolean {
  return isCurrentBook(bookId) && insightStore.currentTaskId === taskId
}

async function startAnalysis(): Promise<void> {
  const bookId = insightStore.currentBookId
  if (!bookId) return

  if (insightStore.isAnalyzing || isStarting.value) {
    showFeedback('分析正在进行中，请稍候')
    return
  }

  if (analysisMode.value === 'chapter' && !selectedChapterId.value) {
    showFeedback('请选择要分析的章节')
    return
  }
  if (analysisMode.value === 'page' && !inputPageNum.value) {
    showFeedback('请输入要分析的页码')
    return
  }

  isStarting.value = true
  feedbackMessage.value = ''

  try {
    const options: Parameters<typeof insightApi.startAnalysis>[1] = {}

    if (analysisMode.value === 'chapter' && selectedChapterId.value) {
      options.mode = 'chapters'
      options.chapters = [selectedChapterId.value]
    } else if (analysisMode.value === 'page' && inputPageNum.value) {
      options.mode = 'pages'
      options.pages = [inputPageNum.value]
    } else {
      options.mode = incrementalAnalysis.value ? 'incremental' : 'full'
    }

    const submission = await insightApi.startAnalysis(bookId, options)
    if (isCurrentBook(bookId)) {
      taskCenterStore.trackJob(submission.jobId)
    }
  } catch (error) {
    if (isCurrentBook(bookId)) showFeedback(getStartErrorMessage(error))
  } finally {
    isStarting.value = false
  }
}

async function pauseAnalysis(): Promise<void> {
  const bookId = insightStore.currentBookId
  const taskId = insightStore.currentTaskId
  if (!bookId || !taskId || isControlling.value) return
  feedbackMessage.value = ''
  isControlling.value = true

  try {
    await taskCenterStore.pause(taskId)
  } catch (error) {
    if (isCurrentBook(bookId)) showFeedback(commandErrorMessage(error, '暂停分析失败'))
  } finally {
    isControlling.value = false
  }
}

async function resumeAnalysis(): Promise<void> {
  const bookId = insightStore.currentBookId
  const taskId = insightStore.currentTaskId
  if (!bookId || !taskId || isControlling.value) return
  feedbackMessage.value = ''
  isControlling.value = true

  try {
    await taskCenterStore.resume(taskId)
  } catch (error) {
    if (isCurrentBook(bookId)) showFeedback(commandErrorMessage(error, '继续分析失败'))
  } finally {
    isControlling.value = false
  }
}

async function continueAnalysis(): Promise<void> {
  const bookId = insightStore.currentBookId
  const taskId = insightStore.currentTaskId
  if (!bookId || !taskId || isControlling.value) return
  feedbackMessage.value = ''
  isControlling.value = true

  try {
    await taskCenterStore.continueJob(taskId)
  } catch (error) {
    if (isCurrentBook(bookId)) showFeedback(commandErrorMessage(error, '继续中断任务失败'))
  } finally {
    isControlling.value = false
  }
}

async function cancelAnalysis(): Promise<void> {
  const bookId = insightStore.currentBookId
  const taskId = insightStore.currentTaskId
  if (!bookId || !taskId || isControlling.value) return
  feedbackMessage.value = ''
  isControlling.value = true

  try {
    const confirmed = await confirmProductAction({
      title: '取消分析',
      message: '确定要取消分析吗？',
      confirmText: '取消分析',
      cancelText: '继续分析',
      tone: 'danger',
    })
    if (!confirmed) return
    if (!isCurrentTask(bookId, taskId)) return
    await taskCenterStore.cancel(taskId)
  } catch (error) {
    if (isCurrentBook(bookId)) showFeedback(commandErrorMessage(error, '取消分析失败'))
  } finally {
    isControlling.value = false
  }
}

async function exportAnalysis(): Promise<void> {
  if (isExporting.value) return
  const bookId = insightStore.currentBookId
  if (!bookId) {
    showFeedback('请先选择书籍')
    return
  }

  feedbackMessage.value = ''
  isExporting.value = true
  try {
    await insightApi.exportAnalysis(bookId)
    if (isCurrentBook(bookId)) {
      showFeedback('导出任务已进入任务中心，完成后可在那里下载', 'info')
    }
  } catch (error) {
    if (isCurrentBook(bookId)) showFeedback(commandErrorMessage(error, '导出失败'))
  } finally {
    isExporting.value = false
  }
}

function clearFeedback(): void {
  feedbackMessage.value = ''
}

watch(analysisMode, () => {
  clearFeedback()
})

watch(() => insightStore.currentBookId, () => {
  selectedChapterId.value = ''
  inputPageNum.value = null
  clearFeedback()
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
      v-if="showProgress"
      class="analysis-progress-panel__progress"
      :value="progressPercent"
      :max="100"
      label="漫画分析进度"
    >
      <span>{{ progressText || statusLabel }}</span>
    </UiProgressBar>

    <div
      v-if="insightStore.progress.message && showProgress"
      class="analysis-progress-panel__progress-message"
    >
      {{ insightStore.progress.message }}
    </div>

    <ProductStatusBanner
      v-if="feedbackMessage"
      class="analysis-progress-panel__error"
      :tone="feedbackTone"
      :aria-live="feedbackTone === 'danger' ? 'assertive' : 'polite'"
    >
      <span>{{ feedbackMessage }}</span>
      <template #actions>
        <UiButton
          variant="secondary"
          size="xs"
          aria-label="清除分析提示"
          @click="clearFeedback"
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
          :disabled="isStarting || isControlling"
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
          :disabled="isControlling"
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
          :disabled="isControlling"
          @click="cancelAnalysis"
        >
          <UiIcon name="square" size="18" />
          <span>取消</span>
        </UiButton>
      </ProductActionRow>

      <ProductActionRow
        v-if="showQueuedButtons"
        class="analysis-progress-panel__action-row"
        aria-label="已排队分析操作"
        justify="start"
        variant="toolbar"
      >
        <UiButton
          variant="danger"
          size="sm"
          class="analysis-progress-panel__action-button"
          title="取消分析"
          aria-label="取消分析"
          :disabled="isControlling"
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
          :disabled="isControlling"
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
          :disabled="isControlling"
          @click="cancelAnalysis"
        >
          <UiIcon name="square" size="18" />
          <span>取消</span>
        </UiButton>
      </ProductActionRow>

      <ProductActionRow
        v-if="showInterruptedButtons"
        class="analysis-progress-panel__action-row"
        aria-label="已中断分析操作"
        justify="start"
        variant="toolbar"
      >
        <UiButton
          variant="primary"
          size="sm"
          class="analysis-progress-panel__action-button"
          title="从安全点继续中断任务"
          aria-label="继续中断任务"
          :disabled="isControlling"
          @click="continueAnalysis"
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
          :disabled="isControlling"
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
      :disabled="isStarting || isControlling"
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
        :disabled="isStarting || isControlling"
      />
      <span class="analysis-progress-panel__page-hint">/ {{ insightStore.totalPageCount || '?' }}</span>
    </div>

    <div v-if="showIdleButtons && analysisModeDescription" class="analysis-progress-panel__mode-description">
      {{ analysisModeDescription }}
    </div>

    <ProductActionRow
      class="analysis-progress-panel__options-row"
      aria-label="分析附加操作"
      justify="between"
      variant="toolbar"
    >
      <UiCheckbox
        v-if="analysisMode === 'full'"
        class="analysis-progress-panel__incremental-checkbox"
        :model-value="incrementalAnalysis"
        label="增量模式"
        aria-label="增量模式"
        :disabled="isStarting || isControlling"
        @change="updateIncrementalAnalysis"
      />
      <UiIconButton
        size="xs"
        variant="soft"
        label="导出分析报告"
        title="导出分析报告"
        :disabled="insightStore.analyzedPageCount === 0 || isExporting"
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

.analysis-progress-panel__status-dot--queued {
  background: var(--insight-action-primary);
  animation: pulse-glow 1.5s infinite;
}

.analysis-progress-panel__status-dot--paused {
  background: var(--insight-status-warning);
}

.analysis-progress-panel__status-dot--interrupted,
.analysis-progress-panel__status-dot--completed-with-errors {
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
}

.analysis-progress-panel__options-row {
  width: 100%;
  padding-top: 8px;
  border-top: 1px solid var(--color-border-muted);
}

.analysis-progress-panel__incremental-checkbox {
  --ui-checkbox-align-items: center;
  --ui-checkbox-gap: 6px;
  --ui-checkbox-color: var(--insight-text-secondary);
  --ui-checkbox-input-width: 14px;
  --ui-checkbox-input-height: 14px;
  --ui-checkbox-input-margin: 0;
  --ui-checkbox-input-accent-color: var(--insight-action-primary);
  --ui-checkbox-label-font-weight: 400;

  font-size: 12px;
}

</style>
