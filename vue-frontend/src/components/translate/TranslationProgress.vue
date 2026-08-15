<template>
  <div v-if="showProgress" class="translation-progress">
    <UiProgressBar
      :value="progressPercent"
      label="翻译进度"
      size="lg"
      striped
      :animated="isActivelyAdvancing"
    >
      <span class="translation-progress__label">
        {{ progressLabel }}
        <span v-if="failedCount > 0" class="translation-progress__failed-count">
          （{{ failedCount }} 张失败）
        </span>
      </span>
    </UiProgressBar>
    <div
      v-if="isParallelProgress && currentProgress.pools.length > 0"
      class="translation-progress__pools"
      aria-label="后端并行流水线进度"
    >
      <div
        v-for="pool in currentProgress.pools"
        :key="pool.kind"
        class="translation-progress__pool"
      >
        <div class="translation-progress__pool-heading">
          <strong>{{ stepKindLabel(pool.kind) }}</strong>
          <span>
            完成 {{ pool.completed }} / {{ pool.total }}
            · 处理中 {{ pool.processing }}
            · 等待 {{ pool.waiting }}
          </span>
          <span
            v-if="pool.lockWaiting"
            class="translation-progress__lock-waiting"
          >
            等待深度学习锁
          </span>
          <span v-else-if="pool.skipped > 0" class="translation-progress__skipped">
            跳过 {{ pool.skipped }}
          </span>
          <span v-if="pool.cancelled > 0" class="translation-progress__skipped">
            取消 {{ pool.cancelled }}
          </span>
          <span v-if="pool.failed > 0" class="translation-progress__failed-count">
            失败 {{ pool.failed }}
          </span>
        </div>
        <UiProgressBar
          :value="poolPercent(pool)"
          :label="`${stepKindLabel(pool.kind)}流水线`"
          size="sm"
          :striped="pool.processing > 0"
          :animated="isActivelyAdvancing && pool.processing > 0"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import {
  type TranslationPoolProgress,
  type TranslationProgress,
} from '@/composables/useTranslationPipeline'
import { useImageStore } from '@/stores/imageStore'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import { stepKindLabel } from '@/utils/taskDisplay'

interface Props {
  progress: TranslationProgress
}

const props = defineProps<Props>()
const imageStore = useImageStore()

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, value))
}

const currentProgress = computed(() => props.progress)
const showProgress = computed(() => (
  currentProgress.value.isInProgress || imageStore.isTranslationInProgress
))
const failedCount = computed(() => currentProgress.value.failed)
const isParallelProgress = computed(
  () => currentProgress.value.executionMode === 'parallel',
)
const isActivelyAdvancing = computed(() => (
  currentProgress.value.status === 'running'
  || currentProgress.value.status === 'pausing'
  || currentProgress.value.status === 'cancelling'
))
const progressPercent = computed(() => {
  if (currentProgress.value.percentage !== undefined) {
    return clampPercent(currentProgress.value.percentage)
  }
  if (currentProgress.value.total === 0) return 0
  return clampPercent(
    currentProgress.value.current / currentProgress.value.total * 100,
  )
})
const progressLabel = computed(() => {
  const value = currentProgress.value
  const base = value.label
    || `后端翻译任务：${value.current} / ${value.total}`
  const details: string[] = value.total > 0
    ? [`总进度 ${value.current} / ${value.total}`]
    : []
  if (value.status === 'queued' && value.queuePosition) {
    details.push(`队列第 ${value.queuePosition} 位`)
  }
  if (value.executionMode === 'sequential' && value.currentStep) {
    details.push(
      `第 ${value.currentStep.itemOrdinal} 页 · ${stepKindLabel(value.currentStep.kind)}`,
    )
  }
  return details.length > 0 ? `${base}（${details.join('，')}）` : base
})

function poolPercent(pool: TranslationPoolProgress): number {
  if (pool.total === 0) return 0
  return clampPercent(
    (pool.completed + pool.failed + pool.skipped + pool.cancelled) / pool.total * 100,
  )
}
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

.translation-progress__pools {
  display: grid;
  gap: 10px;
  margin-top: 16px;
}

.translation-progress__pool {
  display: grid;
  gap: 5px;
}

.translation-progress__pool-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 10px;
  color: var(--color-text-secondary);
  font-size: 0.85rem;
}

.translation-progress__pool-heading strong {
  min-width: 4.5em;
  color: var(--color-text-heading);
}

.translation-progress__lock-waiting {
  color: var(--color-status-warning);
  font-weight: 700;
}

.translation-progress__skipped {
  color: var(--color-text-muted);
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
