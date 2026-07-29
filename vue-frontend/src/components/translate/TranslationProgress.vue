<template>
  <div v-if="showProgress" class="translation-progress">
    <UiProgressBar
      :value="progressPercent"
      label="翻译进度"
      size="lg"
      striped
      animated
    >
      <span class="translation-progress__label">
        {{ progressLabel }}
        <span v-if="failedCount > 0" class="translation-progress__failed-count">
          （{{ failedCount }} 张失败）
        </span>
      </span>
    </UiProgressBar>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { useTranslation, type TranslationProgress } from '@/composables/useTranslationPipeline'
import { useImageStore } from '@/stores/imageStore'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'

interface Props {
  progress?: TranslationProgress
}

const props = defineProps<Props>()
const imageStore = useImageStore()
const translation = useTranslation()

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, value))
}

const currentProgress = computed(() => props.progress || translation.progress.value)
const showProgress = computed(() => (
  currentProgress.value.isInProgress || imageStore.isBatchTranslationInProgress
))
const failedCount = computed(() => currentProgress.value.failed)
const progressPercent = computed(() => {
  if (currentProgress.value.percentage !== undefined) {
    return clampPercent(currentProgress.value.percentage)
  }
  if (currentProgress.value.total === 0) return 0
  return clampPercent(
    Math.round(currentProgress.value.current / currentProgress.value.total * 100),
  )
})
const progressLabel = computed(() => (
  currentProgress.value.label
  || `后端翻译任务：${currentProgress.value.current} / ${currentProgress.value.total}`
))
</script>

<style scoped>
.translation-progress {
  container: translation-progress / inline-size;
  margin: 20px auto;
  padding: 20px 24px;
  width: 85%;
  border-radius: 12px;
  background: var(--color-surface-quiet);
  box-shadow: 0 2px 12px var(--shadow-soft);
}

.translation-progress__label {
  font-size: 1.1em;
  font-weight: 700;
  color: var(--color-text-heading);
  text-align: center;
}

.translation-progress__failed-count {
  color: var(--color-text-danger-strong);
  font-weight: 500;
}

@container translation-progress (max-width: 520px) {
  .translation-progress__label {
    font-size: 1em;
  }
}

@media (--breakpoint-md-down) {
  .translation-progress {
    width: 95%;
    padding: 16px 20px;
  }
}

@media (--breakpoint-xs-down) {
  .translation-progress {
    width: 100%;
    padding: 12px 16px;
  }
}
</style>
