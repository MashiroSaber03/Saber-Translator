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
  margin: 20px auto;
  padding: 20px;
  border: none;
  border-radius: 8px;
  background-color: #f8fafc;
  text-align: center;
  width: 85%;
  box-shadow: 0 2px 8px rgb(0,0,0,0.05);
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

.progress-bar .progress::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: linear-gradient(
      -45deg,
      rgb(255, 255, 255, .2) 25%,
      transparent 25%,
      transparent 50%,
      rgb(255, 255, 255, .2) 50%,
      rgb(255, 255, 255, .2) 75%,
      transparent 75%,
      transparent
  );
  background-size: 30px 30px;
  animation: move 2s linear infinite;
  border-radius: 20px;
  overflow: hidden;
}

</style>
