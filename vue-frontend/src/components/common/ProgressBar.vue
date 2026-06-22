<template>
  <!-- 通用进度条组件 -->
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
 * 翻译流程和批处理任务共用的线性进度条。
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
.translation-progress-bar {
  /* owner tokens: progress-bar */
  --progress-bar-accent-primary: rgba(99, 102, 241, .1);
  --progress-bar-shadow-default: rgba(0, 0, 0, .05);
  --progress-bar-shadow-raised: rgba(0, 0, 0, .1);
  --progress-bar-surface-base: #edf2f7;
  --progress-bar-surface-raised: #4cae4c;
  --progress-bar-surface-muted: #5cb85c;

  margin: 20px auto;
  padding: 20px;
  border: none;
  border-radius: 8px;
  background-color: var(--color-surface-quiet);
  text-align: center;
  width: 85%;
  box-shadow: 0 2px 8px var(--progress-bar-shadow-default);
}

.progress-bar-label {
  margin-bottom: 15px;
  font-weight: bold;
  font-size: 1.1em;
  color: var(--color-text-heading);
}

.progress-bar {
  width: 100%;
  height: 25px;
  background-color: var(--progress-bar-surface-base);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px var(--progress-bar-shadow-raised);
}

.progress-bar .progress {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--progress-bar-surface-raised) 0%, var(--progress-bar-surface-muted) 100%);
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
      var(--progress-bar-accent-primary) 25%,
      transparent 25%,
      transparent 50%,
      var(--progress-bar-accent-primary) 50%,
      var(--progress-bar-accent-primary) 75%,
      transparent 75%,
      transparent
  );
  background-size: 30px 30px;
  animation: move 2s linear infinite;
  border-radius: 20px;
  overflow: hidden;
}

</style>
