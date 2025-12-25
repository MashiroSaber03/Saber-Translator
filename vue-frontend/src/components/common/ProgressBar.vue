<template>
  <!-- 通用进度条组件 - 完全复刻原版样式 -->
  <div v-if="visible" class="translation-progress-bar">
    <div class="progress-bar-label">
      {{ label }}
    </div>
    <div class="progress-bar">
      <div 
        class="progress" 
        :style="{ width: `${percentage}%` }"
      ></div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 通用进度条组件
 * 完全复刻原版 #translationProgressBar 的样式和行为
 */

interface Props {
  /** 是否显示进度条 */
  visible?: boolean
  /** 进度百分比 (0-100) */
  percentage: number
  /** 进度条标签文本 */
  label?: string
}

withDefaults(defineProps<Props>(), {
  visible: true,
  label: '进度'
})
</script>

<style scoped>
/* 完全复刻原版进度条样式 */

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

/* 暗色主题适配 */
[data-theme="dark"] .translation-progress-bar {
  background-color: #2d3748;
}

[data-theme="dark"] .progress-bar-label {
  color: #e2e8f0;
}

[data-theme="dark"] .progress-bar {
  background-color: #1a202c;
}
</style>
