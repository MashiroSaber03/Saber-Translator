<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'

import { ref, computed, watch } from 'vue'
import { useInsightStore, type AnalysisMode } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import CustomSelect from '@/components/common/CustomSelect.vue'
import type { ApiError } from '@/types'

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
    'status-dot': true,
    'running': status === 'running',
    'paused': status === 'paused',
    'completed': status === 'completed',
    'failed': status === 'failed'
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
  if (!confirm('确定要取消分析吗？')) return
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
      const url = URL.createObjectURL(blob)
      try {
        const a = document.createElement('a')
        a.href = url
        a.download = `${insightStore.currentBookId}_analysis.md`
        a.click()
      } finally {
        URL.revokeObjectURL(url)
      }
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
  <div class="sidebar-section analysis-control-compact">
    <div class="analysis-status-bar">
      <div class="status-left">
        <span :class="statusDotClass"></span>
        <span class="status-label">{{ statusLabel }}</span>
      </div>
      <div class="status-right">
        <span class="status-progress">{{ progressText }}</span>
      </div>
    </div>

    <div
      v-if="showRunningButtons || showPausedButtons"
      class="progress-bar-slim"
      :class="{ paused: showPausedButtons }"
    >
      <div
        class="progress-fill-slim"
        :style="{ width: progressPercent + '%' }"
      ></div>
    </div>

    <div
      v-if="insightStore.progress.message && (showRunningButtons || showPausedButtons)"
      class="progress-message"
    >
      {{ insightStore.progress.message }}
    </div>

    <UiButton
      v-if="errorMessage"
      variant="toolbar"
      class="error-message"
      aria-label="清除分析错误"
      @click="clearError"
    >
      ⚠️ {{ errorMessage }}
    </UiButton>

    <div class="analysis-btn-group">
      <div v-if="showIdleButtons" class="btn-group-idle">
        <CustomSelect
          v-model="analysisMode"
          :options="analysisModeOptions"
          variant="compact"
          @change="onAnalysisModeChange"
        />
        <UiButton
          variant="toolbar"
          class="btn-analysis-start"
          :disabled="!canStartAnalysis"
          :class="{ loading: isStarting }"
          @click="startAnalysis"
        >
          <svg v-if="!isStarting" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          <span v-if="isStarting" class="loading-spinner"></span>
          <span>{{ isStarting ? '启动中...' : startButtonText }}</span>
        </UiButton>
      </div>

      <div v-if="showRunningButtons" class="btn-group-running">
        <UiButton
          variant="toolbar"
          class="btn-control btn-pause"
          title="暂停分析"
          @click="pauseAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
          </svg>
          <span class="btn-label">暂停</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="btn-control btn-cancel"
          title="取消分析"
          @click="cancelAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 6h12v12H6z" />
          </svg>
          <span class="btn-label">取消</span>
        </UiButton>
      </div>

      <div v-if="showPausedButtons" class="btn-group-paused">
        <UiButton
          variant="toolbar"
          class="btn-control btn-resume"
          title="继续分析"
          @click="resumeAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          <span class="btn-label">继续</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="btn-control btn-cancel"
          title="取消分析"
          @click="cancelAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 6h12v12H6z" />
          </svg>
          <span class="btn-label">取消</span>
        </UiButton>
      </div>
    </div>

    <CustomSelect
      v-if="showChapterSelect"
      v-model="selectedChapterId"
      :options="chapterOptions"
    />

    <div v-if="showPageInput" class="page-input-wrapper">
      <UiInput
        v-model.number="inputPageNum"
        type="number"
        class="form-input-compact"
        placeholder="输入页码"
        min="1"
        :max="insightStore.totalPageCount || undefined"
      />
      <span class="page-hint">/ {{ insightStore.totalPageCount || '?' }}</span>
    </div>

    <div v-if="showIdleButtons && analysisModeDescription" class="mode-description">
      {{ analysisModeDescription }}
    </div>

    <div v-if="showIdleButtons && analysisMode === 'full' && estimatedTime" class="estimated-time">
      ⏱️ {{ estimatedTime }}
    </div>

    <div class="analysis-options-row">
      <label class="checkbox-compact" title="仅分析未分析的页面，跳过已分析的页面">
        <UiInput
          type="checkbox"
          class="analysis-progress__incremental-checkbox"
          :checked="insightStore.incrementalAnalysis"
          @change="insightStore.setIncrementalAnalysis(($event.target as HTMLInputElement).checked)"
        />
        <span>增量模式</span>
      </label>
      <UiButton
        variant="toolbar"
        class="button-icon-sm"
        title="导出分析报告"
        :disabled="insightStore.analyzedPageCount === 0"
        @click="exportAnalysis"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </UiButton>
    </div>
  </div>
</template>

<style scoped>
.analysis-control-compact {
  --analysis-progress-shadow-default: rgba(99, 102, 241, .3);
  --analysis-progress-surface-base: rgba(239, 68, 68, .1);
  --analysis-progress-surface-raised: #f59e0b;
  --analysis-progress-surface-muted: #16a34a;
  --analysis-progress-text-primary: #ef4444;
}

.analysis-control-compact .progress-message {
  font-size: 12px;
  color: var(--insight-text-secondary);
  margin-top: 4px;
  text-align: center;
}

.analysis-control-compact .error-message {
  display: block;
  width: 100%;
  font-size: 12px;
  color: var(--analysis-progress-text-primary);
  background: var(--analysis-progress-surface-base);
  border: 0;
  padding: 6px 10px;
  border-radius: 4px;
  margin-top: 8px;
  cursor: pointer;
  text-align: left;
}

.analysis-control-compact .btn-label {
  font-size: 12px;
  margin-left: 4px;
}

.analysis-control-compact .page-input-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}

.analysis-control-compact .page-hint {
  font-size: 12px;
  color: var(--insight-text-secondary);
}

.analysis-control-compact .mode-description {
  font-size: 11px;
  color: var(--insight-text-secondary);
  margin-top: 6px;
  font-style: italic;
}

.analysis-control-compact .estimated-time {
  font-size: 11px;
  color: var(--insight-text-secondary);
  margin-top: 4px;
}

.analysis-control-compact .btn-analysis-start.loading {
  opacity: 0.7;
  cursor: wait;
}

.analysis-control-compact .loading-spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid currentcolor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.sidebar-section.analysis-control-compact {
    padding: 12px 16px;
}

.analysis-control-compact .analysis-status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.analysis-control-compact .status-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.analysis-control-compact .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--insight-text-muted);
    transition: all 0.3s;
}

.analysis-control-compact .status-dot.running {
    background: var(--insight-action-primary);
    box-shadow: 0 0 8px var(--insight-action-primary);
    animation: pulse-glow 1.5s infinite;
}

.analysis-control-compact .status-dot.paused {
    background: var(--insight-status-warning);
}

.analysis-control-compact .status-dot.completed {
    background: var(--insight-status-success);
}

.analysis-control-compact .status-dot.failed {
    background: var(--insight-status-error);
}

.analysis-control-compact .status-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--insight-text-primary);
}

.analysis-control-compact .status-progress {
    font-size: 12px;
    color: var(--insight-text-secondary);
    font-variant-numeric: tabular-nums;
}

.analysis-control-compact .progress-bar-slim {
    height: 3px;
    background: var(--insight-surface-tertiary);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 10px;
}

.analysis-control-compact .progress-fill-slim {
    height: 100%;
    background: linear-gradient(90deg, var(--insight-action-primary), var(--insight-action-primary-soft));
    transition: width 0.3s ease;
    width: 0%;
}

.analysis-control-compact .progress-bar-slim.paused .progress-fill-slim {
    background: var(--analysis-progress-surface-raised);
    animation: none;
}

.analysis-control-compact .analysis-btn-group {
    margin-bottom: 10px;
}

.analysis-control-compact .btn-group-idle {
    display: flex;
    gap: 8px;
    flex-wrap: nowrap;
}

.analysis-control-compact .btn-analysis-start {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    background: linear-gradient(135deg, var(--insight-action-primary), var(--insight-action-primary-strong));
    color: var(--color-text-inverse);
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.analysis-control-compact .btn-analysis-start:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--analysis-progress-shadow-default);
}

.analysis-control-compact .btn-analysis-start:active {
    transform: translateY(0);
}

.analysis-control-compact .btn-analysis-start svg {
    flex-shrink: 0;
}

.analysis-control-compact .btn-group-running,
.analysis-control-compact .btn-group-paused {
    display: flex;
    gap: 8px;
}

.analysis-control-compact .btn-control {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.analysis-control-compact .btn-pause {
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-primary);
}

.analysis-control-compact .btn-pause:hover {
    background: var(--insight-status-warning);
    color: var(--color-text-inverse);
}

.analysis-control-compact .btn-resume {
    background: var(--insight-status-success);
    color: var(--color-text-inverse);
}

.analysis-control-compact .btn-resume:hover {
    background: var(--analysis-progress-surface-muted);
}

.analysis-control-compact .btn-cancel {
    background: var(--insight-surface-tertiary);
    color: var(--insight-text-secondary);
}

.analysis-control-compact .btn-cancel:hover {
    background: var(--insight-status-error);
    color: var(--color-text-inverse);
}

.analysis-control-compact .form-input-compact {
    width: 100%;
    padding: 8px 12px;
    font-size: 13px;
    border: 1px solid var(--color-border-muted);
    border-radius: 8px;
    background: var(--insight-surface-page);
    color: var(--insight-text-primary);
    margin-top: 8px;
}

.analysis-control-compact .form-input-compact:focus {
    outline: none;
    border-color: var(--insight-action-primary);
}

.analysis-control-compact .analysis-options-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 8px;
    border-top: 1px solid var(--color-border-muted);
}

.analysis-control-compact .checkbox-compact {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--insight-text-secondary);
    cursor: pointer;
}

.analysis-progress__incremental-checkbox {
    width: 14px;
    height: 14px;
    cursor: pointer;
    accent-color: var(--insight-action-primary);
}

.analysis-control-compact .button-icon-sm {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid var(--color-border-muted);
    border-radius: 6px;
    color: var(--insight-text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.button-icon-sm:hover {
    background: var(--insight-surface-tertiary);
    color: var(--insight-action-primary);
    border-color: var(--insight-action-primary);
}

.button-icon-sm:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
