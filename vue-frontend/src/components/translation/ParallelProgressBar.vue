<script setup lang="ts">
/**
 * 并行翻译多进度条组件
 * 
 * 显示6个池子的实时进度，每个池子一行
 */

import { computed } from 'vue'
import type { ParallelProgress, PoolStatus } from '@/composables/translation/parallel/types'

const props = defineProps<{
  progress: ParallelProgress
  visible: boolean
}>()

const emit = defineEmits<{
  cancel: []
}>()

const totalProgress = computed(() => {
  if (props.progress.totalPages === 0) return 0
  return Math.round((props.progress.totalCompleted / props.progress.totalPages) * 100)
})

const formatTime = (seconds: number): string => {
  if (seconds <= 0) return '--'
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (minutes > 0) {
    return `${minutes}分${secs}秒`
  }
  return `${secs}秒`
}

const getPoolStatusText = (pool: PoolStatus): string => {
  if (pool.isWaitingLock) return '等待锁'
  if (pool.processing) return `处理第${pool.currentPage}页`
  if (pool.waiting > 0) return `等待中(${pool.waiting})`
  if (pool.completed > 0) return `已完成${pool.completed}`
  return '空闲'
}

const getPoolStatusClass = (pool: PoolStatus): string => {
  if (pool.isWaitingLock) return 'status-waiting-lock'
  if (pool.processing) return 'status-processing'
  if (pool.waiting > 0) return 'status-waiting'
  return 'status-idle'
}
</script>

<template>
  <div v-if="visible" class="parallel-progress-overlay">
    <div class="parallel-progress-container">
      <!-- 标题 -->
      <div class="progress-header">
        <span class="title">🚀 并行翻译进度</span>
        <span class="overall-progress">{{ totalProgress }}%</span>
      </div>

      <!-- 总体进度条 -->
      <div class="overall-progress-bar">
        <div 
          class="overall-progress-fill" 
          :style="{ width: `${totalProgress}%` }"
        />
      </div>

      <!-- 统计信息 -->
      <div class="progress-stats">
        <span>✅ {{ progress.totalCompleted }}/{{ progress.totalPages }}</span>
        <span v-if="progress.totalFailed > 0" class="failed">❌ {{ progress.totalFailed }} 失败</span>
        <span>⏱️ 剩余 {{ formatTime(progress.estimatedTimeRemaining) }}</span>
      </div>

      <!-- 池子进度列表 -->
      <div class="pools-container">
        <div 
          v-for="pool in progress.pools" 
          :key="pool.name" 
          class="pool-item"
          :class="getPoolStatusClass(pool)"
        >
          <span class="pool-icon">{{ pool.icon }}</span>
          <span class="pool-name">{{ pool.name }}</span>
          <span class="pool-status">{{ getPoolStatusText(pool) }}</span>
          <span class="pool-completed">{{ pool.completed }}</span>
        </div>
      </div>

      <!-- 取消按钮 -->
      <button class="cancel-btn" @click="emit('cancel')">
        取消翻译
      </button>
    </div>
  </div>
</template>

<style scoped>
.parallel-progress-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.parallel-progress-container {
  background: var(--bg-color, #1e1e1e);
  border-radius: 12px;
  padding: 24px;
  min-width: 400px;
  max-width: 500px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-color, #fff);
}

.overall-progress {
  font-size: 24px;
  font-weight: bold;
  color: var(--primary-color, #4a9eff);
}

.overall-progress-bar {
  height: 8px;
  background: var(--border-color, #333);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
}

.overall-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4a9eff, #00d4aa);
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-stats {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--text-secondary, #aaa);
  margin-bottom: 20px;
}

.progress-stats .failed {
  color: #ff6b6b;
}

.pools-container {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 20px;
}

.pool-item {
  display: grid;
  grid-template-columns: 30px 60px 1fr 40px;
  align-items: center;
  padding: 10px 12px;
  background: var(--bg-secondary, #2a2a2a);
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.2s ease;
}

.pool-icon {
  font-size: 18px;
}

.pool-name {
  font-weight: 500;
  color: var(--text-color, #fff);
}

.pool-status {
  color: var(--text-secondary, #aaa);
  font-size: 12px;
}

.pool-completed {
  text-align: right;
  font-weight: 600;
  color: var(--primary-color, #4a9eff);
}

/* 状态样式 */
.status-processing {
  background: rgba(74, 158, 255, 0.15);
  border: 1px solid rgba(74, 158, 255, 0.3);
}

.status-processing .pool-status {
  color: #4a9eff;
}

.status-waiting-lock {
  background: rgba(255, 193, 7, 0.15);
  border: 1px solid rgba(255, 193, 7, 0.3);
}

.status-waiting-lock .pool-status {
  color: #ffc107;
}

.status-waiting {
  background: rgba(108, 117, 125, 0.15);
}

.status-idle {
  opacity: 0.6;
}

.cancel-btn {
  width: 100%;
  padding: 12px;
  background: rgba(255, 107, 107, 0.2);
  border: 1px solid rgba(255, 107, 107, 0.4);
  color: #ff6b6b;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn:hover {
  background: rgba(255, 107, 107, 0.3);
}
</style>
