<template>
  <!-- 翻译进度组件 -->
  <div v-if="showProgress" id="translationProgressBar" class="translation-progress-bar">
    <!-- 并行模式：新版多进度条 -->
    <template v-if="isParallelMode && parallelProgress">
      <!-- 标题行 -->
      <div class="parallel-header">
        <span class="header-title">🚀 并行翻译中：{{ parallelProgress.totalCompleted }}/{{ parallelProgress.totalPages }}</span>
      </div>
      
      <!-- 预保存进度（仅在预保存进行中显示） -->
      <div v-if="parallelProgress.preSave?.isRunning" class="presave-section">
        <div class="presave-label">
          📥 预保存原始图片：{{ parallelProgress.preSave.current }}/{{ parallelProgress.preSave.total }}
        </div>
        <div class="presave-progress-bar">
          <div 
            class="presave-progress-fill" 
            :style="{ width: getPreSavePercent() + '%' }"
          ></div>
        </div>
      </div>
      
      <!-- 各池子进度条 -->
      <div class="pools-list">
        <div 
          v-for="pool in parallelProgress.pools" 
          :key="pool.name"
          class="pool-row"
          :class="{ 
            'pool-processing': pool.processing,
            'pool-waiting-lock': pool.isWaitingLock 
          }"
        >
          <!-- 图标 + 名称 -->
          <div class="pool-label">
            <span class="pool-icon">{{ pool.icon }}</span>
            <span class="pool-name">{{ pool.name }}</span>
          </div>
          
          <!-- 进度条 -->
          <div class="pool-progress-bar">
            <!-- 完成部分（绿色） -->
            <div 
              class="progress-completed" 
              :style="{ width: getPoolCompletedPercent(pool) + '%' }"
            ></div>
            <!-- 处理中部分（蓝色，仅当正在处理时显示） -->
            <div 
              v-if="pool.processing"
              class="progress-processing" 
              :style="{ 
                left: getPoolCompletedPercent(pool) + '%',
                width: getPoolProcessingWidth() + '%'
              }"
            ></div>
          </div>
          
          <!-- 完成数/总数 -->
          <div class="pool-stats">
            <span class="completed-count">{{ pool.completed }}</span>
            <span class="total-count">/ {{ parallelProgress.totalPages }}</span>
            <!-- 等待徽章 -->
            <span v-if="pool.waiting > 0" class="waiting-badge">+{{ pool.waiting }}</span>
            <!-- 锁止指示器 -->
            <span v-if="pool.isWaitingLock" class="lock-indicator" title="等待深度学习锁">🔒</span>
          </div>
        </div>
        
        <!-- 保存进度行（仅在启用自动保存时显示） -->
        <div v-if="parallelProgress.save" class="pool-row save-row">
          <div class="pool-label">
            <span class="pool-icon">💾</span>
            <span class="pool-name">保存</span>
          </div>
          <div class="pool-progress-bar">
            <div 
              class="progress-completed save-progress" 
              :style="{ width: getSavePercent() + '%' }"
            ></div>
          </div>
          <div class="pool-stats">
            <span class="completed-count">{{ parallelProgress.save.completed }}</span>
            <span class="total-count">/ {{ parallelProgress.save.total }}</span>
          </div>
        </div>
      </div>
      
      <!-- 分隔线 -->
      <div class="divider"></div>
      
      <!-- 总体进度 -->
      <div class="overall-section">
        <div class="overall-label">
          总进度：{{ parallelOverallPercent }}%
          <span v-if="parallelProgress.totalFailed > 0" class="failed-text">
            （{{ parallelProgress.totalFailed }} 失败）
          </span>
        </div>
        <div class="overall-progress-bar">
          <div class="overall-progress-fill progress-stripe" :style="{ width: parallelOverallPercent + '%' }"></div>
        </div>
      </div>
    </template>
    
    <!-- 普通模式：单进度条 -->
    <template v-else>
      <div class="progress-bar-label">
        {{ progressLabel }}
        <template v-if="failedCount > 0">
          <span class="failed-count">（{{ failedCount }} 张失败）</span>
        </template>
      </div>
      <div class="progress-bar">
        <div class="progress progress-stripe" :style="{ width: `${progressPercent}%` }"></div>
      </div>
    </template>
  </div>
</template>


<script setup lang="ts">
/**
 * 翻译进度组件
 * 显示翻译进度条、当前处理图片序号
 * 支持并行模式的多进度条显示
 */

import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useTranslation, type TranslationProgress } from '@/composables/useTranslationPipeline'
import { useParallelTranslation } from '@/composables/translation/parallel'
import type { PoolStatus } from '@/composables/translation/parallel/types'

// ============================================================
// Props 定义
// ============================================================

interface Props {
  /** 进度数据（可选，默认从 useTranslation 获取） */
  progress?: TranslationProgress
}

const props = defineProps<Props>()

// ============================================================
// Store 和 Composables
// ============================================================

const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const translation = useTranslation()
const parallelTranslation = useParallelTranslation()

// ============================================================
// 计算属性
// ============================================================

/** 是否并行模式（包括预保存阶段） */
const isParallelMode = computed(() => {
  // 如果并行模式启用且（正在运行 或 正在预保存）
  const isPreSaving = parallelTranslation.progress.value?.preSave?.isRunning
  return settingsStore.settings.parallel?.enabled && (parallelTranslation.isRunning.value || isPreSaving)
})

/** 并行进度数据 */
const parallelProgress = computed(() => parallelTranslation.progress.value)

/** 并行总体进度百分比 */
const parallelOverallPercent = computed(() => {
  const progress = parallelProgress.value
  if (!progress || progress.totalPages === 0) return 0
  return Math.round((progress.totalCompleted / progress.totalPages) * 100)
})

/** 获取池子完成百分比 */
function getPoolCompletedPercent(pool: PoolStatus): number {
  const total = parallelProgress.value?.totalPages || 0
  if (total === 0) return 0
  return Math.round((pool.completed / total) * 100)
}

/** 获取池子处理中部分宽度（固定一小段） */
function getPoolProcessingWidth(): number {
  const total = parallelProgress.value?.totalPages || 0
  if (total === 0) return 0
  // 处理中显示一个任务的宽度
  return Math.max(3, Math.round((1 / total) * 100))
}

/** 获取预保存百分比 */
function getPreSavePercent(): number {
  const preSave = parallelProgress.value?.preSave
  if (!preSave || preSave.total === 0) return 0
  return Math.round((preSave.current / preSave.total) * 100)
}

/** 获取保存百分比 */
function getSavePercent(): number {
  const save = parallelProgress.value?.save
  if (!save || save.total === 0) return 0
  return Math.round((save.completed / save.total) * 100)
}

/** 当前进度数据 */
const currentProgress = computed(() => {
  return props.progress || translation.progress.value
})

/** 是否显示进度条 */
const showProgress = computed(() => {
  return currentProgress.value.isInProgress || imageStore.isBatchTranslationInProgress || isParallelMode.value
})

/** 当前处理的图片索引 */
const currentIndex = computed(() => currentProgress.value.current)

/** 总图片数 */
const totalCount = computed(() => currentProgress.value.total)

/** 失败数量 */
const failedCount = computed(() => currentProgress.value.failed)

/** 进度百分比（优先使用自定义百分比，否则根据 current/total 计算） */
const progressPercent = computed(() => {
  // 优先使用自定义百分比
  if (currentProgress.value.percentage !== undefined) {
    return currentProgress.value.percentage
  }
  if (totalCount.value === 0) return 0
  return Math.round((currentIndex.value / totalCount.value) * 100)
})

/** 进度标签文本（优先使用自定义标签，复刻原版） */
const progressLabel = computed(() => {
  // 优先使用自定义标签
  if (currentProgress.value.label) {
    return currentProgress.value.label
  }
  return `翻译中：${currentIndex.value} / ${totalCount.value}`
})
</script>


<style scoped>
/* ===================================
   进度条样式 - 新版并行进度条设计
   =================================== */

.translation-progress-bar {
  margin: 20px auto;
  padding: 20px 24px;
  border: none;
  border-radius: 12px;
  background-color: #f8fafc;
  width: 85%;
  box-shadow: 0 2px 12px rgb(0,0,0,0.06);
}

/* ===================================
   并行模式 - 新版样式
   =================================== */

.parallel-header {
  text-align: center;
  margin-bottom: 20px;
}

.header-title {
  font-size: 1.2em;
  font-weight: 600;
  color: #2c3e50;
}

/* 池子列表 */
.pools-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.pool-row {
  display: grid;
  grid-template-columns: 80px 1fr 150px; /* 增加第三列宽度以容纳所有统计元素 */
  align-items: center;
  gap: 12px;
  padding: 4px 0;
}

/* 池子标签 */
.pool-label {
  display: flex;
  align-items: center;
  gap: 6px;
}

.pool-icon {
  font-size: 16px;
}

.pool-name {
  font-size: 14px;
  font-weight: 500;
  color: #4a5568;
}

/* 池子进度条 */
.pool-progress-bar {
  position: relative;
  height: 12px;
  background: #e2e8f0;
  border-radius: 6px;
  overflow: hidden;
}

.progress-completed {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: linear-gradient(90deg, #48bb78, #38a169);
  border-radius: 6px;
  transition: width 0.3s ease;
}

.progress-processing {
  position: absolute;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #4299e1, #3182ce);
  border-radius: 6px;
  transition: left 0.3s ease, width 0.3s ease;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

/* 池子统计 */
.pool-stats {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  font-size: 14px;
  font-variant-numeric: tabular-nums; /* 使用等宽数字 */
  min-width: 150px; /* 增加宽度以容纳数字+徽章+锁 */
  flex-wrap: nowrap; /* 禁止换行，保持单行显示 */
}

.completed-count {
  font-weight: 600;
  color: #2d3748;
  text-align: right;
  min-width: 28px; /* 为最多两位数预留空间 */
}

.total-count {
  color: #a0aec0;
  min-width: 38px; /* 为"/ XX"格式预留空间 */
}

.waiting-badge {
  margin-left: 4px;
  padding: 1px 6px;
  background: #ffc107;
  color: #fff;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.4;
  white-space: nowrap;
}

.lock-indicator {
  margin-left: 2px;
  font-size: 13px;
  animation: lockPulse 1s ease-in-out infinite;
  line-height: 1;
}

@keyframes lockPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 处理中状态 */
.pool-processing .pool-name {
  color: #3182ce;
}

/* 等待锁状态 */
.pool-waiting-lock .pool-name {
  color: #d69e2e;
}

/* 预保存进度 */
.presave-section {
  margin-bottom: 16px;
  padding: 12px;
  background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
  border-radius: 8px;
  border: 1px solid #90caf9;
}

.presave-label {
  font-size: 14px;
  font-weight: 500;
  color: #1565c0;
  margin-bottom: 8px;
}

.presave-progress-bar {
  height: 10px;
  background: #e3f2fd;
  border-radius: 5px;
  overflow: hidden;
}

.presave-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #42a5f5, #1976d2);
  border-radius: 5px;
  transition: width 0.3s ease;
  animation: presavePulse 1.5s ease-in-out infinite;
}

@keyframes presavePulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}

/* 保存进度行 */
.save-row .pool-name {
  color: #7c3aed;
}

.save-progress {
  background: linear-gradient(90deg, #8b5cf6, #7c3aed);
}

/* 分隔线 */
.divider {
  height: 1px;
  background: #e2e8f0;
  margin: 20px 0;
}

/* 总体进度 */
.overall-section {
  margin-top: 8px;
}

.overall-label {
  font-size: 14px;
  color: #4a5568;
  margin-bottom: 8px;
}

.failed-text {
  color: #e53e3e;
  font-weight: 500;
}

.overall-progress-bar {
  height: 20px;
  background: #e2e8f0;
  border-radius: 10px;
  overflow: hidden;
}

.overall-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #48bb78 0%, #68d391 100%);
  border-radius: 10px;
  transition: width 0.3s ease;
  position: relative;
}

/* 公共条纹动画样式 */
.progress-stripe::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: linear-gradient(
    -45deg,
    rgb(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgb(255, 255, 255, 0.2) 50%,
    rgb(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 30px 30px;
  animation: stripeMove 1.5s linear infinite;
  border-radius: inherit;
  overflow: hidden;
}

@keyframes stripeMove {
  0% { background-position: 0 0; }
  100% { background-position: 30px 30px; }
}

/* ===================================
   普通模式样式 - 保持原有
   =================================== */

.progress-bar-label {
  margin-bottom: 15px;
  font-weight: bold;
  font-size: 1.1em;
  color: #2c3e50;
  text-align: center;
}

.progress-bar {
  width: 100%;
  height: 25px;
  background-color: #edf2f7;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgb(0,0,0,0.1);
}

.progress-bar .progress {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #4cae4c 0%, #5cb85c 100%);
  transition: width 0.3s ease;
  border-radius: 20px;
  position: relative;
}

/* 失败数量 */
.failed-count {
  color: #e74c3c;
  font-weight: 500;
}

/* ===================================
   响应式设计 - 小屏幕适配
   =================================== */

/* 平板和小屏幕（宽度 < 768px） */
@media (width <= 768px) {
  .translation-progress-bar {
    width: 95%;
    padding: 16px 20px;
  }

  .pool-row {
    grid-template-columns: 70px 1fr 130px; /* 平板：适当增加宽度 */
    gap: 10px;
  }

  .pool-label {
    gap: 4px;
  }

  .pool-icon {
    font-size: 14px;
  }

  .pool-name {
    font-size: 13px;
  }

  .pool-stats {
    font-size: 13px;
    min-width: 130px; /* 平板：适当增加宽度 */
  }

  .completed-count {
    min-width: 24px;
  }

  .total-count {
    min-width: 34px;
  }

  .waiting-badge {
    font-size: 10px;
    padding: 1px 5px;
  }

  .lock-indicator {
    font-size: 12px;
  }
}

/* 手机屏幕（宽度 < 480px） */
@media (width <= 480px) {
  .translation-progress-bar {
    width: 100%;
    padding: 12px 16px;
  }

  .header-title {
    font-size: 1em;
  }

  .pool-row {
    grid-template-columns: 60px 1fr 120px; /* 手机：适当增加宽度 */
    gap: 8px;
  }

  .pool-label {
    gap: 3px;
  }

  .pool-name {
    font-size: 12px;
  }

  .pool-stats {
    font-size: 12px;
    min-width: 120px; /* 手机：适当增加宽度 */
    gap: 2px;
  }

  .completed-count {
    min-width: 22px;
  }

  .total-count {
    min-width: 32px;
  }

  .waiting-badge {
    font-size: 9px;
    padding: 1px 4px;
    margin-left: 2px;
  }

  .lock-indicator {
    font-size: 11px;
    margin-left: 1px;
  }
}

</style>
