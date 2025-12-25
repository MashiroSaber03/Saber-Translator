<script setup lang="ts">
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
  return insightStore.analysisStatus === 'idle' || insightStore.analysisStatus === 'completed'
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
  return insightStore.analysisStatus === 'completed' ? '重新分析' : '开始分析'
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
      return '分析整本书的所有页面'
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
    const options: Parameters<typeof insightApi.startAnalysis>[1] = {
      incremental: insightStore.incrementalAnalysis
    }

    if (analysisMode.value === 'chapter' && selectedChapterId.value) {
      options.mode = 'chapter'
      options.chapter_id = selectedChapterId.value
    } else if (analysisMode.value === 'page' && inputPageNum.value) {
      options.mode = 'page'
      options.page_num = inputPageNum.value
    } else {
      options.mode = 'full'
    }

    const response = await insightApi.startAnalysis(insightStore.currentBookId, options)
    
    if (response.success) {
      insightStore.setAnalysisStatus('running')
      emit('start-polling')
    } else {
      errorMessage.value = response.error || '启动分析失败'
      console.error('启动分析失败:', response.error)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '启动分析失败'
    console.error('启动分析失败:', error)
  } finally {
    isStarting.value = false
  }
}

/**
 * 暂停分析
 */
async function pauseAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    const response = await insightApi.pauseAnalysis(insightStore.currentBookId)
    if (response.success) {
      insightStore.setAnalysisStatus('paused')
    }
  } catch (error) {
    console.error('暂停分析失败:', error)
  }
}

/**
 * 继续分析
 */
async function resumeAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return

  try {
    const response = await insightApi.resumeAnalysis(insightStore.currentBookId)
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
 */
async function cancelAnalysis(): Promise<void> {
  if (!insightStore.currentBookId) return
  if (!confirm('确定要取消分析吗？')) return

  try {
    const response = await insightApi.cancelAnalysis(insightStore.currentBookId)
    if (response.success) {
      insightStore.setAnalysisStatus('idle')
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
          @change="onAnalysisModeChange"
        />
        <button 
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
        </button>
      </div>
      
      <!-- 运行中状态 -->
      <div v-if="showRunningButtons" class="btn-group-running">
        <button 
          class="btn-control btn-pause" 
          title="暂停分析"
          @click="pauseAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
          </svg>
          <span class="btn-label">暂停</span>
        </button>
        <button 
          class="btn-control btn-cancel" 
          title="取消分析"
          @click="cancelAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 6h12v12H6z" />
          </svg>
          <span class="btn-label">取消</span>
        </button>
      </div>
      
      <!-- 暂停状态 -->
      <div v-if="showPausedButtons" class="btn-group-paused">
        <button 
          class="btn-control btn-resume" 
          title="继续分析"
          @click="resumeAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          <span class="btn-label">继续</span>
        </button>
        <button 
          class="btn-control btn-cancel" 
          title="取消分析"
          @click="cancelAnalysis"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 6h12v12H6z" />
          </svg>
          <span class="btn-label">取消</span>
        </button>
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
      <input 
        v-model.number="inputPageNum"
        type="number" 
        class="form-input-compact" 
        placeholder="输入页码" 
        min="1"
        :max="insightStore.totalPageCount || undefined"
      >
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
        <input 
          type="checkbox" 
          :checked="insightStore.incrementalAnalysis"
          @change="insightStore.setIncrementalAnalysis(($event.target as HTMLInputElement).checked)"
        >
        <span>增量模式</span>
      </label>
      <button 
        class="btn-icon-sm" 
        title="导出分析报告"
        :disabled="insightStore.analyzedPageCount === 0"
        @click="exportAnalysis"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
/* ==================== AnalysisProgress 完整样式 ==================== */

/* ==================== CSS变量 ==================== */
.analysis-control-compact {
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f1f5f9;
  --bg-hover: rgba(99, 102, 241, 0.1);
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --primary-color: #6366f1;
  --primary-light: #818cf8;
  --primary-dark: #4f46e5;
  --success-color: #22c55e;
  --warning-color: #f59e0b;
  --warning: #f59e0b;
  --error-color: #ef4444;
  --danger: #ef4444;
}

:global(body.dark-theme) .analysis-control-compact,
.analysis-control-compact.dark-theme {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --bg-hover: rgba(99, 102, 241, 0.2);
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border-color: #334155;
}

/* ==================== 组件特定样式 ==================== */

/* 进度条暂停状态 */
.progress-bar-slim.paused .progress-fill-slim {
  background: var(--warning, #f59e0b);
  animation: none;
}

/* 进度消息 */
.progress-message {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
  text-align: center;
}

/* 错误消息 */
.error-message {
  font-size: 12px;
  color: var(--danger, #ef4444);
  background: rgba(239, 68, 68, 0.1);
  padding: 6px 10px;
  border-radius: 4px;
  margin-top: 8px;
  cursor: pointer;
}

/* 按钮标签 */
.btn-label {
  font-size: 12px;
  margin-left: 4px;
}

/* 页码输入包装器 */
.page-input-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}

.page-hint {
  font-size: 12px;
  color: var(--text-secondary);
}

/* 模式描述 */
.mode-description {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 6px;
  font-style: italic;
}

/* 预估时间 */
.estimated-time {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* 加载中按钮 */
.btn-analysis-start.loading {
  opacity: 0.7;
  cursor: wait;
}

/* 加载动画 */
.loading-spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 禁用状态 */
.btn-analysis-start:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon-sm:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ==================== 分析控制完整样式 - 从 manga-insight.css 迁移 ==================== */

.analysis-control-compact {
    padding: 12px 16px;
}

.analysis-status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.status-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.3s;
}

.status-dot.running {
    background: var(--primary-color);
    box-shadow: 0 0 8px var(--primary-color);
    animation: pulse-glow 1.5s infinite;
}

.status-dot.paused {
    background: var(--warning-color);
}

.status-dot.completed {
    background: var(--success-color);
}

.status-dot.failed {
    background: var(--error-color);
}

@keyframes pulse-glow {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--primary-color); }
    50% { opacity: 0.6; box-shadow: 0 0 4px var(--primary-color); }
}

.status-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
}

.status-progress {
    font-size: 12px;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
}

.progress-bar-slim {
    height: 3px;
    background: var(--bg-tertiary);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 10px;
}

.progress-fill-slim {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
    transition: width 0.3s ease;
    width: 0%;
}

.analysis-btn-group {
    margin-bottom: 10px;
}

.btn-group-idle {
    display: flex;
    gap: 8px;
    flex-wrap: nowrap;
}

/* 覆盖CustomSelect在按钮组中的宽度 */
.btn-group-idle :deep(.custom-select) {
    min-width: 70px !important;
    flex: 0 0 auto !important;
}

.btn-group-idle :deep(.custom-select-trigger) {
    height: 38px !important;
    padding: 0 10px !important;
}

.analysis-mode-select {
    flex: 0 0 auto;
    padding: 8px 12px;
    font-size: 13px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-primary);
    color: var(--text-primary);
    cursor: pointer;
    min-width: 70px;
}

.analysis-mode-select:focus {
    outline: none;
    border-color: var(--primary-color);
}

.btn-analysis-start {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-analysis-start:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.btn-analysis-start:active {
    transform: translateY(0);
}

.btn-analysis-start svg {
    flex-shrink: 0;
}

.btn-group-running,
.btn-group-paused {
    display: flex;
    gap: 8px;
}

.btn-control {
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

.btn-pause {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.btn-pause:hover {
    background: var(--warning-color);
    color: white;
}

.btn-resume {
    background: var(--success-color);
    color: white;
}

.btn-resume:hover {
    background: #16a34a;
}

.btn-cancel {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
}

.btn-cancel:hover {
    background: var(--error-color);
    color: white;
}

.form-select-compact,
.form-input-compact {
    width: 100%;
    padding: 8px 12px;
    font-size: 13px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-primary);
    color: var(--text-primary);
    margin-top: 8px;
}

.form-select-compact:focus,
.form-input-compact:focus {
    outline: none;
    border-color: var(--primary-color);
}

.analysis-options-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-top: 8px;
    border-top: 1px solid var(--border-color);
}

.checkbox-compact {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
    cursor: pointer;
}

.checkbox-compact input[type="checkbox"] {
    width: 14px;
    height: 14px;
    cursor: pointer;
    accent-color: var(--primary-color);
}

.btn-icon-sm {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.btn-icon-sm:hover {
    background: var(--bg-tertiary);
    color: var(--primary-color);
    border-color: var(--primary-color);
}
</style>
