<template>
  <div v-if="showProgress" class="translation-progress">
    <template v-if="isParallelMode && parallelProgress">
      <div class="translation-progress__parallel-header">
        <span class="translation-progress__header-title">
          <UiIcon name="sparkles" />
          <span>并行翻译中：{{ parallelProgress.totalCompleted }}/{{ parallelProgress.totalPages }}</span>
        </span>
      </div>

      <div v-if="parallelProgress.preSave?.isRunning" class="translation-progress__presave-section">
        <UiProgressBar
          :value="getPreSavePercent()"
          label="预保存原始图片进度"
          size="md"
          animated
        >
          <span class="translation-progress__presave-label">
            <UiIcon name="download" />
            <span>预保存原始图片：{{ parallelProgress.preSave.current }}/{{ parallelProgress.preSave.total }}</span>
          </span>
        </UiProgressBar>
      </div>

      <div class="translation-progress__pools-list">
        <div
          v-for="pool in parallelProgress.pools"
          :key="pool.name"
          class="translation-progress__pool-row"
          :class="{
            'translation-progress__pool-row--processing': pool.processing,
            'translation-progress__pool-row--waiting-lock': pool.isWaitingLock
          }"
        >
          <div class="translation-progress__pool-label">
            <UiIcon class="translation-progress__pool-icon" :name="pool.icon" />
            <span class="translation-progress__pool-name">{{ pool.name }}</span>
          </div>

          <div
            class="translation-progress__pool-bar"
            role="progressbar"
            :aria-label="`${pool.name}进度`"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-valuenow="getPoolCompletedPercent(pool)"
          >
            <div
              class="translation-progress__completed-segment"
              :style="{ width: getPoolCompletedPercent(pool) + '%' }"
            ></div>
            <div
              v-if="pool.processing"
              class="translation-progress__processing-segment"
              :style="{
                left: getPoolCompletedPercent(pool) + '%',
                width: getPoolProcessingWidth(pool) + '%'
              }"
            ></div>
          </div>

          <div class="translation-progress__pool-stats">
            <span class="translation-progress__completed-count">{{ pool.completed }}</span>
            <span class="translation-progress__total-count">/ {{ parallelProgress.totalPages }}</span>
            <span v-if="pool.waiting > 0" class="translation-progress__waiting-badge">+{{ pool.waiting }}</span>
            <UiIcon v-if="pool.isWaitingLock" name="lock" class="translation-progress__lock-indicator" title="等待深度学习锁" />
          </div>
        </div>

        <div v-if="parallelProgress.save" class="translation-progress__pool-row translation-progress__pool-row--save">
          <div class="translation-progress__pool-label">
            <UiIcon class="translation-progress__pool-icon" name="save" />
            <span class="translation-progress__pool-name">保存</span>
          </div>
          <UiProgressBar
            :value="getSavePercent()"
            label="保存进度"
            tone="brand"
            size="md"
          />
          <div class="translation-progress__pool-stats">
            <span class="translation-progress__completed-count">{{ parallelProgress.save.completed }}</span>
            <span class="translation-progress__total-count">/ {{ parallelProgress.save.total }}</span>
          </div>
        </div>
      </div>

      <div class="translation-progress__divider"></div>

      <div class="translation-progress__overall-section">
        <div class="translation-progress__overall-label">
          总进度：{{ parallelOverallPercent }}%
          <span v-if="parallelProgress.totalFailed > 0" class="translation-progress__failed-text">
            （{{ parallelProgress.totalFailed }} 失败）
          </span>
        </div>
        <UiProgressBar
          :value="parallelOverallPercent"
          label="翻译总进度"
          tone="success"
          size="lg"
          striped
          animated
        />
      </div>
    </template>

    <template v-else>
      <UiProgressBar :value="progressPercent" label="翻译进度">
        <span class="translation-progress__normal-label">
          {{ progressLabel }}
          <template v-if="failedCount > 0">
            <span class="translation-progress__failed-count">（{{ failedCount }} 张失败）</span>
          </template>
        </span>
      </UiProgressBar>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation, type TranslationProgress } from '@/composables/useTranslationPipeline'
import { useParallelTranslation } from '@/composables/translation/parallel'
import type { PoolStatus } from '@/composables/translation/parallel/types'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'

interface Props {
  progress?: TranslationProgress
}

const props = defineProps<Props>()

const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const translation = useTranslation()
const parallelTranslation = useParallelTranslation()

const isParallelMode = computed(() => {
  const isPreSaving = parallelTranslation.progress.value?.preSave?.isRunning
  return settingsStore.settings.parallel?.enabled && (parallelTranslation.isRunning.value || isPreSaving)
})

const parallelProgress = computed(() => parallelTranslation.progress.value)

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, value))
}

const parallelOverallPercent = computed(() => {
  const progress = parallelProgress.value
  if (!progress || progress.totalPages === 0) return 0
  return clampPercent(Math.round((progress.totalCompleted / progress.totalPages) * 100))
})

function getPoolCompletedPercent(pool: PoolStatus): number {
  const total = parallelProgress.value?.totalPages || 0
  if (total === 0) return 0
  return clampPercent(Math.round((pool.completed / total) * 100))
}

function getPoolProcessingWidth(pool: PoolStatus): number {
  const total = parallelProgress.value?.totalPages || 0
  if (total === 0) return 0
  const completedPercent = getPoolCompletedPercent(pool)
  const oneTaskPercent = Math.max(3, Math.round((1 / total) * 100))
  return clampPercent(Math.min(100 - completedPercent, oneTaskPercent))
}

function getPreSavePercent(): number {
  const preSave = parallelProgress.value?.preSave
  if (!preSave || preSave.total === 0) return 0
  return clampPercent(Math.round((preSave.current / preSave.total) * 100))
}

function getSavePercent(): number {
  const save = parallelProgress.value?.save
  if (!save || save.total === 0) return 0
  return clampPercent(Math.round((save.completed / save.total) * 100))
}

const currentProgress = computed(() => {
  return props.progress || translation.progress.value
})

const showProgress = computed(() => {
  return currentProgress.value.isInProgress || imageStore.isBatchTranslationInProgress || isParallelMode.value
})

const currentIndex = computed(() => currentProgress.value.current)
const totalCount = computed(() => currentProgress.value.total)
const failedCount = computed(() => currentProgress.value.failed)

const progressPercent = computed(() => {
  if (currentProgress.value.percentage !== undefined) {
    return clampPercent(currentProgress.value.percentage)
  }
  if (totalCount.value === 0) return 0
  return clampPercent(Math.round((currentIndex.value / totalCount.value) * 100))
})

const progressLabel = computed(() => {
  if (currentProgress.value.label) {
    return currentProgress.value.label
  }
  return `翻译中：${currentIndex.value} / ${totalCount.value}`
})
</script>

<style scoped>
.translation-progress {
  margin: 20px auto;
  padding: 20px 24px;
  border: none;
  border-radius: 12px;
  background-color: var(--color-surface-quiet);
  width: 85%;
  box-shadow: 0 2px 12px var(--shadow-soft);
  container: translation-progress / inline-size;
}

.translation-progress__parallel-header {
  text-align: center;
  margin-bottom: 20px;
}

.translation-progress__header-title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 1.2em;
  font-weight: 600;
  color: var(--color-text-heading);
}

.translation-progress__pools-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.translation-progress__pool-row {
  display: grid;
  grid-template-columns: minmax(0, 80px) minmax(0, 1fr) max-content;
  align-items: center;
  gap: 12px;
  padding: 4px 0;
}

.translation-progress__pool-label {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.translation-progress__pool-icon {
  width: 16px;
  height: 16px;
}

.translation-progress__pool-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.translation-progress__pool-bar {
  position: relative;
  min-width: 0;
  height: 12px;
  background: var(--color-border-muted);
  border-radius: 6px;
  overflow: hidden;
}

.translation-progress__completed-segment {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: linear-gradient(90deg, var(--color-status-success), var(--color-status-success-hover));
  border-radius: 6px;
  transition: width 0.3s ease;
}

.translation-progress__processing-segment {
  position: absolute;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, var(--color-action-primary-soft), var(--color-action-primary-hover));
  border-radius: 6px;
  transition: left 0.3s ease, width 0.3s ease;
  animation: pulse 1.5s ease-in-out infinite;
}

.translation-progress__pool-stats {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  min-width: 0;
  flex-wrap: nowrap;
}

.translation-progress__completed-count {
  font-weight: 600;
  color: var(--color-text-heading);
  text-align: right;
  min-width: 28px;
}

.translation-progress__total-count {
  color: var(--color-text-disabled);
  min-width: 38px;
}

.translation-progress__waiting-badge {
  margin-left: 4px;
  padding: 1px 6px;
  background: var(--color-status-warning);
  color: var(--color-text-inverse);
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.4;
  white-space: nowrap;
}

.translation-progress__lock-indicator {
  margin-left: 2px;
  width: 13px;
  height: 13px;
  animation: lockPulse 1s ease-in-out infinite;
  line-height: 1;
}

@keyframes lockPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.translation-progress__pool-row--processing .translation-progress__pool-name {
  color: var(--color-action-primary-hover);
}

.translation-progress__pool-row--waiting-lock .translation-progress__pool-name {
  color: var(--color-status-warning-hover);
}

.translation-progress__presave-section {
  margin-bottom: 16px;
  padding: 12px;
  background: linear-gradient(135deg, var(--color-surface-quiet) 0%, var(--color-surface-interactive-hover) 100%);
  border-radius: 8px;
  border: 1px solid var(--color-border-info);
}

.translation-progress__presave-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-link-strong);
}

.translation-progress__pool-row--save .translation-progress__pool-name {
  color: var(--color-text-brand);
}

.translation-progress__divider {
  height: 1px;
  background: var(--color-border-muted);
  margin: 20px 0;
}

.translation-progress__overall-section {
  margin-top: 8px;
}

.translation-progress__overall-label {
  font-size: 14px;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
}

.translation-progress__failed-text {
  color: var(--color-status-error-hover);
  font-weight: 500;
}

.translation-progress__normal-label {
  font-weight: 700;
  font-size: 1.1em;
  color: var(--color-text-heading);
  text-align: center;
}

.translation-progress__failed-count {
  color: var(--color-text-danger-strong);
  font-weight: 500;
}

@container translation-progress (max-width: 520px) {
  .translation-progress__pool-row {
    grid-template-columns: 1fr;
    align-items: stretch;
    gap: 6px;
  }

  .translation-progress__pool-stats {
    justify-content: flex-start;
  }
}

@media (--breakpoint-md-down) {
  .translation-progress {
    width: 95%;
    padding: 16px 20px;
  }

  .translation-progress__pool-row {
    gap: 10px;
  }

  .translation-progress__pool-label {
    gap: 4px;
  }

  .translation-progress__pool-icon {
    width: 14px;
    height: 14px;
  }

  .translation-progress__pool-name {
    font-size: 13px;
  }

  .translation-progress__pool-stats {
    font-size: 13px;
  }

  .translation-progress__completed-count {
    min-width: 24px;
  }

  .translation-progress__total-count {
    min-width: 34px;
  }

  .translation-progress__waiting-badge {
    font-size: 10px;
    padding: 1px 5px;
  }

  .translation-progress__lock-indicator {
    width: 12px;
    height: 12px;
  }
}

@media (--breakpoint-xs-down) {
  .translation-progress {
    width: 100%;
    padding: 12px 16px;
  }

  .translation-progress__header-title {
    font-size: 1em;
  }

  .translation-progress__pool-row {
    gap: 8px;
  }

  .translation-progress__pool-label {
    gap: 3px;
  }

  .translation-progress__pool-name {
    font-size: 12px;
  }

  .translation-progress__pool-stats {
    font-size: 12px;
    gap: 2px;
  }

  .translation-progress__completed-count {
    min-width: 22px;
  }

  .translation-progress__total-count {
    min-width: 32px;
  }

  .translation-progress__waiting-badge {
    font-size: 9px;
    padding: 1px 4px;
    margin-left: 2px;
  }

  .translation-progress__lock-indicator {
    width: 11px;
    height: 11px;
    margin-left: 1px;
  }
}
</style>
