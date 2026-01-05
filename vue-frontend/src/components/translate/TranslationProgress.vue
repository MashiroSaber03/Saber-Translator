<template>
  <!-- 翻译进度组件 - 使用原版进度条样式 -->
  <div v-if="showProgress" id="translationProgressBar" class="translation-progress-bar">
    <!-- 进度标签 -->
    <div class="progress-bar-label">
      {{ progressLabel }}
      <template v-if="failedCount > 0">
        <span class="failed-count">（{{ failedCount }} 张失败）</span>
      </template>
    </div>
    
    <!-- 进度条 -->
    <div class="progress-bar">
      <div class="progress" :style="{ width: `${progressPercent}%` }"></div>
    </div>
  </div>
</template>


<script setup lang="ts">
/**
 * 翻译进度组件
 * 显示翻译进度条、当前处理图片序号
 */

import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useTranslation, type TranslationProgress } from '@/composables/useTranslation'

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
const translation = useTranslation()

// ============================================================
// 计算属性
// ============================================================

/** 当前进度数据 */
const currentProgress = computed(() => {
  return props.progress || translation.progress.value
})

/** 是否显示进度条 */
const showProgress = computed(() => {
  return currentProgress.value.isInProgress || imageStore.isBatchTranslationInProgress
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

</style>
