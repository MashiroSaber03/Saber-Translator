<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

import { ref, computed, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import type { AnalysisMode } from '@/types/insight'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import * as insightApi from '@/api/insight'
import type { ApiError } from '@/types'
import { confirmProductAction } from '@/composables/useProductConfirm'

const analysisModeOptions = [
  { label: '全书', value: 'full' },
  { label: '增量', value: 'incremental' },
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

const analysisMode = ref<AnalysisMode>('incremental')
const selectedChapterId = ref('')
const inputPageNum = ref<number | null>(null)
const isStarting = ref(false)
const errorMessage = ref('')

const statusDotClass = computed(() => {
  const status = insightStore.analysisStatus
  return {
    'analysis-progress-panel__status-dot': true,
    'analysis-progress-panel__status-dot--queued': status === 'queued',
    'analysis-progress-panel__status-dot--running': status === 'running',
    'analysis-progress-panel__status-dot--pausing': status === 'pausing',
    'analysis-progress-panel__status-dot--paused': status === 'paused',
    'analysis-progress-panel__status-dot--cancelling': status === 'cancelling',
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
    case 'pausing': return '暂停中'
    case 'paused': return '已暂停'
    case 'cancelling': return '取消中'
    case 'interrupted': return '已中断'
    case 'completed': return '已完成'
    case 'completed_with_errors': return '部分完成'
    case 'cancelled': return '已取消'
    case 'failed': return '分析失败'
    case 'error': return '状态异常'
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
    || insightStore.analysisStatus === 'completed_with_errors'
    || insightStore.analysisStatus === 'cancelled'
    || insightStore.analysisStatus === 'failed'
    || insightStore.analysisStatus === 'error'
})

const showRunningButtons = computed(() => {
  return insightStore.analysisStatus === 'running'
})

const showPausedButtons = computed(() => {
  return insightStore.analysisStatus === 'paused'
})

const showQueuedButtons = computed(() => insightStore.analysisStatus === 'queued')
const showPausingButtons = computed(() => insightStore.analysisStatus === 'pausing')
const showInterruptedButtons = computed(() => insightStore.analysisStatus === 'interrupted')
const showProgress = computed(() => [
  'queued',
  'running',
  'pausing',
  'paused',
  'cancelling',
  'interrupted',
].includes(insightStore.analysisStatus))

const startButtonText = computed(() => {
  return insightStore.analysisStatus === 'idle' || insightStore.analysisStatus === 'cancelled'
    ? '开始分析'
    : '重新分析'
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
      return '使用新 run 全量重跑；完成发布前旧结果持续可读'
    case 'incremental':
      return '仅分析新增或原图内容已变化的页面'
    case 'chapter':
      return '仅分析选中章节的页面'
    case 'page':
      return '仅分析指定的单个页面'
    default:
      return ''
  }
})

const estimatedTime = computed(() => {
  let pageCount = insightStore.totalPageCount
  if (analysisMode.value === 'page') pageCount = inputPageNum.value ? 1 : 0
  if (analysisMode.value === 'chapter') {
    const chapter = insightStore.chapters.find(value => value.id === selectedChapterId.value)
    pageCount = chapter ? chapter.endPage - chapter.startPage + 1 : 0
  }
  if (pageCount === 0) return ''

  const pagesPerBatch = insightStore.config.batch.pagesPerBatch || 5
  const batches = Math.ceil(pageCount / pagesPerBatch)
  const seconds = batches * 10

  if (seconds < 60) return `约 ${seconds} 秒`
  const minutes = Math.ceil(seconds / 60)
  return `约 ${minutes} 分钟`
})

const progressPercent = computed(() => Math.max(0, Math.min(100, insightStore.progressPercent)))

function updateAnalysisMode(value: string | number): void {
  analysisMode.value = value as AnalysisMode
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
      options.mode = analysisMode.value === 'incremental' ? 'incremental' : 'full'
    }

    const submission = await insightApi.startAnalysis(insightStore.currentBookId, options)
    insightStore.setCurrentTaskId(submission.jobId)
    insightStore.setAnalysisStatus('queued')
    await taskCenterStore.refresh()
  } catch (error) {
    errorMessage.value = getStartErrorMessage(error)
  } finally {
    isStarting.value = false
  }
}

async function pauseAnalysis(): Promise<void> {
  if (!insightStore.currentBookId || !insightStore.currentTaskId) return
  errorMessage.value = ''

  try {
    await insightApi.pauseAnalysis(insightStore.currentTaskId)
    insightStore.setAnalysisStatus('pausing')
    await taskCenterStore.refresh()
  } catch {
    errorMessage.value = '暂停分析失败'
  }
}

async function resumeAnalysis(): Promise<void> {
  if (!insightStore.currentBookId || !insightStore.currentTaskId) return
  errorMessage.value = ''

  try {
    await insightApi.resumeAnalysis(insightStore.currentTaskId)
    insightStore.setAnalysisStatus('queued')
    await taskCenterStore.refresh()
  } catch {
    errorMessage.value = '继续分析失败'
  }
}

async function continueAnalysis(): Promise<void> {
  if (!insightStore.currentBookId || !insightStore.currentTaskId) return
  errorMessage.value = ''

  try {
    await insightApi.continueAnalysis(insightStore.currentTaskId)
    insightStore.setAnalysisStatus('queued')
    await taskCenterStore.refresh()
  } catch {
    errorMessage.value = '继续中断任务失败'
  }
}

async function cancelAnalysis(): Promise<void> {
  if (!insightStore.currentBookId || !insightStore.currentTaskId) return
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
    await insightApi.cancelAnalysis(insightStore.currentTaskId)
    insightStore.setAnalysisStatus('cancelling')
    await taskCenterStore.refresh()
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
    await insightApi.exportAnalysis(insightStore.currentBookId)
    errorMessage.value = '导出任务已进入任务中心，完成后可在那里下载'
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
        v-if="showQueuedButtons || showPausingButtons"
        class="analysis-progress-panel__action-row"
        :aria-label="showQueuedButtons ? '已排队分析操作' : '暂停中的分析操作'"
        justify="start"
        variant="toolbar"
      >
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

    <div v-if="showIdleButtons && estimatedTime" class="analysis-progress-panel__estimated-time">
      <UiIcon name="clock" />
      <span>{{ estimatedTime }}</span>
    </div>

    <ProductActionRow
      class="analysis-progress-panel__options-row"
      aria-label="分析附加操作"
      justify="end"
      variant="toolbar"
    >
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

.analysis-progress-panel__status-dot--queued,
.analysis-progress-panel__status-dot--pausing,
.analysis-progress-panel__status-dot--cancelling {
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
  margin-top: 8px;
}

.analysis-progress-panel__options-row {
  width: 100%;
  padding-top: 8px;
  border-top: 1px solid var(--color-border-muted);
}

</style>
