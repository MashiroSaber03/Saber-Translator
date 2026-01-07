<template>
  <!-- 翻译进度组件 -->
  <div v-if="showProgress" id="translationProgressBar" class="translation-progress-bar">
    <!-- 并行模式：多进度条 -->
    <template v-if="isParallelMode && parallelProgress">
      <div class="parallel-progress-header">
        <span class="progress-title">🚀 并行翻译进度</span>
        <span class="progress-overall">{{ parallelOverallPercent }}%</span>
      </div>
      
      <!-- 总体进度条 -->
      <div class="progress-bar overall-bar">
        <div class="progress" :style="{ width: `${parallelOverallPercent}%` }"></div>
      </div>
      
      <!-- 统计信息 -->
      <div class="parallel-stats">
        <span>✅ {{ parallelProgress.totalCompleted }}/{{ parallelProgress.totalPages }}</span>
        <span v-if="parallelProgress.totalFailed > 0" class="failed-count">❌ {{ parallelProgress.totalFailed }} 失败</span>
      </div>
      
      <!-- 各池子进度 -->
      <div class="pools-grid">
        <div 
          v-for="pool in parallelProgress.pools" 
          :key="pool.name"
          class="pool-item"
          :class="{ 
            'pool-processing': pool.processing,
            'pool-waiting-lock': pool.isWaitingLock 
          }"
        >
          <span class="pool-icon">{{ pool.icon }}</span>
          <span class="pool-name">{{ pool.name }}</span>
          <span class="pool-status">
            {{ pool.isWaitingLock ? '等锁' : pool.processing ? `#${pool.currentPage}` : pool.completed > 0 ? `✓${pool.completed}` : '-' }}
          </span>
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
        <div class="progress" :style="{ width: `${progressPercent}%` }"></div>
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

/** 是否并行模式 */
const isParallelMode = computed(() => {
  return settingsStore.settings.parallel?.enabled && parallelTranslation.isRunning.value
})

/** 并行进度数据 */
const parallelProgress = computed(() => parallelTranslation.progress.value)

/** 并行总体进度百分比 */
const parallelOverallPercent = computed(() => {
  const progress = parallelProgress.value
  if (!progress || progress.totalPages === 0) return 0
  return Math.round((progress.totalCompleted / progress.totalPages) * 100)
})

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
   进度条样式 - 完整复刻原版 components.css
   =================================== */

.translation-progress-bar {
  margin-top: 20px;
  margin-bottom: 20px;
  padding: 20px;
  border: none;
  border-radius: 8px;
  background-color: #f8fafc;
  text-align: center;
  width: 85%;
  margin-left: auto;
  margin-right: auto;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.progress-bar-label {
  margin-bottom: 15px;
  font-weight: bold;
  font-size: 1.1em;
  color: #2c3e50;
}

.progress-bar {
  width: 100%;
  height: 25px;
  background-color: #edf2f7;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

.progress-bar .progress {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #4cae4c 0%, #5cb85c 100%);
  transition: width 0.3s ease;
  border-radius: 20px;
  position: relative;
}

.progress-bar .progress:after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  background-image: linear-gradient(
      -45deg,
      rgba(255, 255, 255, .2) 25%,
      transparent 25%,
      transparent 50%,
      rgba(255, 255, 255, .2) 50%,
      rgba(255, 255, 255, .2) 75%,
      transparent 75%,
      transparent
  );
  background-size: 30px 30px;
  animation: move 2s linear infinite;
  border-radius: 20px;
  overflow: hidden;
}

@keyframes move {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: 30px 30px;
  }
}

/* 失败数量 */
.failed-count {
  color: #e74c3c;
  font-weight: 500;
}

/* ===================================
   并行模式样式
   =================================== */

.parallel-progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.progress-title {
  font-weight: bold;
  font-size: 1.1em;
  color: #2c3e50;
}

.progress-overall {
  font-size: 1.5em;
  font-weight: bold;
  color: #4a9eff;
}

.overall-bar {
  margin-bottom: 12px;
}

.overall-bar .progress {
  background: linear-gradient(90deg, #4a9eff 0%, #00d4aa 100%);
}

.parallel-stats {
  display: flex;
  justify-content: center;
  gap: 20px;
  font-size: 0.9em;
  color: #666;
  margin-bottom: 16px;
}

.pools-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.pool-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 8px;
  background: #f0f4f8;
  border-radius: 8px;
  font-size: 12px;
  transition: all 0.2s ease;
}

.pool-icon {
  font-size: 20px;
  margin-bottom: 4px;
}

.pool-name {
  font-weight: 500;
  color: #2c3e50;
  margin-bottom: 2px;
}

.pool-status {
  color: #888;
  font-size: 11px;
}

.pool-processing {
  background: rgba(74, 158, 255, 0.15);
  border: 1px solid rgba(74, 158, 255, 0.3);
}

.pool-processing .pool-status {
  color: #4a9eff;
  font-weight: 500;
}

.pool-waiting-lock {
  background: rgba(255, 193, 7, 0.15);
  border: 1px solid rgba(255, 193, 7, 0.3);
}

.pool-waiting-lock .pool-status {
  color: #ffc107;
}

</style>
