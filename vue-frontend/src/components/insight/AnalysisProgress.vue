<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
/**
 * 分析进度组件
 * 显示分析进度、状态指示和控制按钮
 * 支持全书分析、单章节分析、单页分析三种模式
 * 支持增量分析（仅分析未分析的页面）
 */

import { ref, computed, watch } from 'vue'
import { useInsightStore, type AnalysisMode } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import CustomSelect from '@/components/common/CustomSelect.vue'
import type { ApiError } from '@/types'

/** 分析模式选项 */
const analysisModeOptions = [
  { label: '全书', value: 'full' },
  { label: '章节', value: 'chapter' },
  { label: '单页', value: 'page' }
]

/** 章节选项（动态计算） */
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

// ============================================================
// 事件定义
// ============================================================

const emit = defineEmits<{
  /** 启动轮询事件 */
  (e: 'start-polling'): void
  /** 停止轮询事件 */
  (e: 'stop-polling'): void
}>()

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 分析模式 */
const analysisMode = ref<AnalysisMode>('full')

/** 选中的章节ID */
const selectedChapterId = ref('')

/** 输入的页码 */
const inputPageNum = ref<number | null>(null)

/** 是否正在启动分析 */
const isStarting = ref(false)

/** 错误消息 */
const errorMessage = ref('')

// ============================================================
// 计算属性
// ============================================================

/** 状态点样式类 */
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

/** 状态标签文本 */
const statusLabel = computed(() => {
  switch (insightStore.analysisStatus) {
    case 'running': return '分析中'
    case 'paused': return '已暂停'
    case 'completed': return '已完成'
    case 'failed': return '分析失败'
    default: return '未分析'
  }
})

/** 进度文本 */
const progressText = computed(() => {
  const { current, total } = insightStore.progress
  if (total === 0) return ''
  return `${current}/${total}`
})

/** 是否显示空闲状态按钮组 */
const showIdleButtons = computed(() => {
  return insightStore.analysisStatus === 'idle'
    || insightStore.analysisStatus === 'completed'
    || insightStore.analysisStatus === 'failed'
})

/** 是否显示运行中按钮组 */
const showRunningButtons = computed(() => {
  return insightStore.analysisStatus === 'running'
})

/** 是否显示暂停状态按钮组 */
const showPausedButtons = computed(() => {
  return insightStore.analysisStatus === 'paused'
})

/** 开始按钮文本 */
const startButtonText = computed(() => {
  return (insightStore.analysisStatus === 'completed' || insightStore.analysisStatus === 'failed')
    ? '重新分析'
    : '开始分析'
})

/** 是否显示章节选择 */
const showChapterSelect = computed(() => analysisMode.value === 'chapter')

/** 是否显示页码输入 */
const showPageInput = computed(() => analysisMode.value === 'page')

/** 是否可以开始分析 */
const canStartAnalysis = computed(() => {
  // 正在启动中不能再次启动
  if (isStarting.value) return false
  // 正在分析中不能启动
  if (insightStore.isAnalyzing) return false
  // 章节模式需要选择章节
  if (analysisMode.value === 'chapter' && !selectedChapterId.value) return false
  // 单页模式需要输入页码
  if (analysisMode.value === 'page' && !inputPageNum.value) return false
  return true
})

/** 分析模式描述 */
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

/** 预估分析时间（基于页数和每批页数） */
const estimatedTime = computed(() => {
  const totalPages = insightStore.totalPageCount
  if (totalPages === 0) return ''
  
  const pagesPerBatch = insightStore.config.batch.pagesPerBatch || 5
  const batches = Math.ceil(totalPages / pagesPerBatch)
  // 假设每批约需要10秒
  const seconds = batches * 10
  
  if (seconds < 60) return `约 ${seconds} 秒`
  const minutes = Math.ceil(seconds / 60)
  return `约 ${minutes} 分钟`
})

// ============================================================
// 方法
// ============================================================

/**
 * 分析模式变更处理
 */
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

/**
 * 开始分析
 */
async function startAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return

  // 防止重复启动
  if (insightStore.isAnalyzing || isStarting.value) {
    console.warn('分析正在进行中或正在启动')
    return
  }

  // 验证输入
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
    // 构建符合后端 API 协议的参数
    // 后端期望: mode = 'full' | 'incremental' | 'chapters' | 'pages'
    const options: Parameters<typeof insightApi.startAnalysis>[1] = {}

    if (analysisMode.value === 'chapter' && selectedChapterId.value) {
      // 章节模式：mode='chapters', chapters=[id]
      options.mode = 'chapters'
      options.chapters = [selectedChapterId.value]
    } else if (analysisMode.value === 'page' && inputPageNum.value) {
      // 单页模式：mode='pages', pages=[num]
      options.mode = 'pages'
      options.pages = [inputPageNum.value]
    } else {
      // 全书模式：根据增量开关决定是 'incremental' 还是 'full'
      options.mode = insightStore.incrementalAnalysis ? 'incremental' : 'full'
    }

    const response = await insightApi.startAnalysis(insightStore.currentBookId, options) as any
    
    if (response.success) {
      // 保存任务ID（用于后续暂停/恢复/取消操作）
      if (response.task_id) {
        insightStore.setCurrentTaskId(response.task_id)
      }
      insightStore.setAnalysisStatus('running')
      emit('start-polling')
    } else {
      errorMessage.value = response.error || '启动分析失败'
      console.error('启动分析失败:', response.error)
    }
  } catch (error) {
    errorMessage.value = getStartErrorMessage(error)
    console.error('启动分析失败:', error)
  } finally {
    isStarting.value = false
  }
}

/**
 * 暂停分析
 * 与接口流程 一致：传递 task_id 参数
 */
async function pauseAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    const response = await insightApi.pauseAnalysis(
      insightStore.currentBookId,
      insightStore.currentTaskId || undefined
    )
    if (response.success) {
      insightStore.setAnalysisStatus('paused')
    }
  } catch (error) {
    console.error('暂停分析失败:', error)
  }
}

/**
 * 继续分析
 * 与接口流程 一致：传递 task_id 参数
 */
async function resumeAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    const response = await insightApi.resumeAnalysis(
      insightStore.currentBookId,
      insightStore.currentTaskId || undefined
    )
    if (response.success) {
      insightStore.setAnalysisStatus('running')
      emit('start-polling')
    }
  } catch (error) {
    console.error('继续分析失败:', error)
  }
}

/**
 * 取消分析
 * 与接口流程 一致：传递 task_id 参数
 */
async function cancelAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return
  if (!confirm('确定要取消分析吗？')) return

  try {
    const response = await insightApi.cancelAnalysis(
      insightStore.currentBookId,
      insightStore.currentTaskId || undefined
    )
    if (response.success) {
      insightStore.setAnalysisStatus('idle')
      insightStore.setCurrentTaskId(null)
      emit('stop-polling')
    }
  } catch (error) {
    console.error('取消分析失败:', error)
  }
}

/**
 * 导出分析报告
 */
async function exportAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) {
    errorMessage.value = '请先选择书籍'
    return
  }

  try {
    const response = await insightApi.exportAnalysis(insightStore.currentBookId) as any
    
    if (response.success && response.markdown) {
      // 下载 Markdown 文件
      const blob = new Blob([response.markdown], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${insightStore.currentBookId}_analysis.md`
      a.click()
      URL.revokeObjectURL(url)
    } else {
      errorMessage.value = response.error || '导出失败'
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导出失败'
    console.error('导出失败:', error)
  }
}

/**
 * 清除错误消息
 */
function clearError(): void {
  errorMessage.value = ''
}

// ============================================================
// 监听
// ============================================================

// 监听分析模式变化，清除错误消息
watch(analysisMode, () => {
  clearError()
})
</script>

<template>
  <div class="sidebar-section analysis-control-compact">
    <!-- 状态栏 -->
    <div class="analysis-status-bar">
      <div class="status-left">
        <span :class="statusDotClass"></span>
        <span class="status-label">{{ statusLabel }}</span>
      </div>
      <div class="status-right">
        <span class="status-progress">{{ progressText }}</span>
      </div>
    </div>
    
    <!-- 进度条（分析中或暂停时显示） -->
    <div 
      v-if="showRunningButtons || showPausedButtons" 
      class="progress-bar-slim"
      :class="{ paused: showPausedButtons }"
    >
      <div 
        class="progress-fill-slim" 
        :style="{ width: insightStore.progressPercent + '%' }"
      ></div>
    </div>
    
    <!-- 进度消息 -->
    <div 
      v-if="insightStore.progress.message && (showRunningButtons || showPausedButtons)" 
      class="progress-message"
    >
      {{ insightStore.progress.message }}
    </div>
    
    <!-- 错误消息 -->
    <div v-if="errorMessage" class="error-message" @click="clearError">
      ⚠️ {{ errorMessage }}
    </div>
    
    <!-- 控制按钮组 -->
    <div class="analysis-btn-group">
      <!-- 初始/完成状态 -->
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
      
      <!-- 运行中状态 -->
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
      
      <!-- 暂停状态 -->
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
    
    <!-- 章节选择（单章节模式时显示） -->
    <CustomSelect
      v-if="showChapterSelect"
      v-model="selectedChapterId"
      :options="chapterOptions"
    />
    
    <!-- 页码输入（单页模式时显示） -->
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
    
    <!-- 分析模式描述 -->
    <div v-if="showIdleButtons && analysisModeDescription" class="mode-description">
      {{ analysisModeDescription }}
    </div>
    
    <!-- 预估时间（全书模式时显示） -->
    <div v-if="showIdleButtons && analysisMode === 'full' && estimatedTime" class="estimated-time">
      ⏱️ {{ estimatedTime }}
    </div>
    
    <!-- 选项行 -->
    <div class="analysis-options-row">
      <label class="checkbox-compact" title="仅分析未分析的页面，跳过已分析的页面">
        <UiInput 
          type="checkbox" 
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

<style scoped>/* ==================== AnalysisProgress样式 ==================== */

/* ==================== 组件特定样式 ==================== */

/* 进度消息 */
.analysis-control-compact .progress-message {
  font-size: 12px;
  color: var(--insight-text-secondary);
  margin-top: 4px;
  text-align: center;
}

/* 错误消息 */
.analysis-control-compact .error-message {
  font-size: 12px;
  color: var(--analysis-progress-text-primary);
  background: var(--analysis-progress-surface-base);
  padding: 6px 10px;
  border-radius: 4px;
  margin-top: 8px;
  cursor: pointer;
}

/* 按钮标签 */
.analysis-control-compact .btn-label {
  font-size: 12px;
  margin-left: 4px;
}

/* 页码输入包装器 */
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

/* 模式描述 */
.analysis-control-compact .mode-description {
  font-size: 11px;
  color: var(--insight-text-secondary);
  margin-top: 6px;
  font-style: italic;
}

/* 预估时间 */
.analysis-control-compact .estimated-time {
  font-size: 11px;
  color: var(--insight-text-secondary);
  margin-top: 4px;
}

/* 加载中按钮 */
.analysis-control-compact .btn-analysis-start.loading {
  opacity: 0.7;
  cursor: wait;
}

/* 加载动画 */
.analysis-control-compact .loading-spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid currentcolor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* ==================== 分析控制样式 ==================== */

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
    background: var(--insight-color-primary);
    box-shadow: 0 0 8px var(--insight-color-primary);
    animation: pulse-glow 1.5s infinite;
}

.analysis-control-compact .status-dot.paused {
    background: var(--insight-warning-color);
}

.analysis-control-compact .status-dot.completed {
    background: var(--insight-success-color);
}

.analysis-control-compact .status-dot.failed {
    background: var(--insight-error-color);
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
    background: var(--insight-bg-tertiary);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 10px;
}

.analysis-control-compact .progress-fill-slim {
    height: 100%;
    background: linear-gradient(90deg, var(--insight-color-primary), var(--insight-primary-light));
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

.analysis-control-compact .analysis-mode-select {
    flex: 0 0 auto;
    padding: 8px 12px;
    font-size: 13px;
    border: 1px solid var(--color-border-muted);
    border-radius: 8px;
    background: var(--insight-bg-primary);
    color: var(--insight-text-primary);
    cursor: pointer;
    min-width: 70px;
}

.analysis-control-compact .analysis-mode-select:focus {
    outline: none;
    border-color: var(--insight-color-primary);
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
    background: linear-gradient(135deg, var(--insight-color-primary), var(--insight-primary-dark));
    color: white;
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
    background: var(--insight-bg-tertiary);
    color: var(--insight-text-primary);
}

.analysis-control-compact .btn-pause:hover {
    background: var(--insight-warning-color);
    color: white;
}

.analysis-control-compact .btn-resume {
    background: var(--insight-success-color);
    color: white;
}

.analysis-control-compact .btn-resume:hover {
    background: var(--analysis-progress-surface-muted);
}

.analysis-control-compact .btn-cancel {
    background: var(--insight-bg-tertiary);
    color: var(--insight-text-secondary);
}

.analysis-control-compact .btn-cancel:hover {
    background: var(--insight-error-color);
    color: white;
}

.analysis-control-compact .form-select-compact,
.analysis-control-compact .form-input-compact {
    width: 100%;
    padding: 8px 12px;
    font-size: 13px;
    border: 1px solid var(--color-border-muted);
    border-radius: 8px;
    background: var(--insight-bg-primary);
    color: var(--insight-text-primary);
    margin-top: 8px;
}

.analysis-control-compact .form-select-compact:focus,
.analysis-control-compact .form-input-compact:focus {
    outline: none;
    border-color: var(--insight-color-primary);
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

.analysis-control-compact .checkbox-compact input[type="checkbox"] {
    width: 14px;
    height: 14px;
    cursor: pointer;
    accent-color: var(--insight-color-primary);
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
    background: var(--insight-bg-tertiary);
    color: var(--insight-color-primary);
    border-color: var(--insight-color-primary);
}

.button-icon-sm:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
