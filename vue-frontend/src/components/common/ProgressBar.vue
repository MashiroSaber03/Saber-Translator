<template>
  <div v-if="visible" class="translation-progress-bar">
    <div class="progress-bar-label">
      {{ label }}
    </div>
    <div
      class="progress-bar"
      role="progressbar"
      :aria-label="label"
      aria-valuemin="0"
      aria-valuemax="100"
      :aria-valuenow="clampedPercentage"
    >
      <div 
        class="progress" 
        :style="{ width: `${clampedPercentage}%` }"
      ></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  /** 是否显示进度条 */
  visible?: boolean
  /** 进度百分比 (0-100) */
  percentage: number
  /** 进度条标签文本 */
  label?: string
}

const props = withDefaults(defineProps<Props>(), {
  visible: true,
  label: '进度'
})

const clampedPercentage = computed(() => {
  if (!Number.isFinite(props.percentage)) return 0
  return Math.min(100, Math.max(0, props.percentage))
})
</script>

<style scoped>
.translation-progress-bar {
  --progress-bar-stripe-color: var(--color-focus-brand-soft);
  --progress-bar-shadow: var(--shadow-soft);
  --progress-bar-track-shadow: var(--shadow-medium);
  --progress-bar-track-background: var(--color-surface-muted);
  --progress-bar-fill-start: var(--color-status-success-hover);
  --progress-bar-fill-end: var(--color-status-success);

  margin: 20px auto;
  padding: 20px;
  border: none;
  border-radius: 8px;
  background-color: var(--color-surface-quiet);
  text-align: center;
  width: 85%;
  box-shadow: 0 2px 8px var(--progress-bar-shadow);
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
  background-color: var(--progress-bar-track-background);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px var(--progress-bar-track-shadow);
}

.progress-bar .progress {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--progress-bar-fill-start) 0%, var(--progress-bar-fill-end) 100%);
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
      var(--progress-bar-stripe-color) 25%,
      transparent 25%,
      transparent 50%,
      var(--progress-bar-stripe-color) 50%,
      var(--progress-bar-stripe-color) 75%,
      transparent 75%,
      transparent
  );
  background-size: 30px 30px;
  animation: move 2s linear infinite;
  border-radius: 20px;
  overflow: hidden;
}

</style>
