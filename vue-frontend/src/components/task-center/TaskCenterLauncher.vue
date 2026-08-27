<script setup lang="ts">
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const store = useTaskCenterStore()
</script>

<template>
  <OverlayLayer level="mobile-overlay" passthrough>
    <UiButton
      class="task-center-launcher"
      variant="ghost"
      size="sm"
      :aria-label="`打开任务中心，${store.activeCount} 个当前任务，${store.queuedCount} 个排队中，${store.interruptedCount} 个中断待处理`"
      @click="store.open"
    >
      <span
        class="task-center-launcher__signal"
        :class="{
          'task-center-launcher__signal--active': store.currentJobs.some(job => job.status === 'running'),
          'task-center-launcher__signal--paused': (
            !store.currentJobs.some(job => job.status === 'running')
            && store.currentJobs.some(job => job.status === 'paused')
          ),
        }"
      />
      <span class="task-center-launcher__label">任务中心</span>
      <span v-if="store.activeCount + store.queuedCount" class="task-center-launcher__badge">
        {{ store.activeCount + store.queuedCount }}
      </span>
      <span
        v-if="store.interruptedCount"
        class="task-center-launcher__interrupted"
        title="有中断任务待处理"
      >
        待处理 {{ store.interruptedCount }}
      </span>
    </UiButton>
  </OverlayLayer>
</template>

<style scoped>
.task-center-launcher {
  position: absolute;
  top: 12px;
  right: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 0 12px;
  color: var(--color-text-heading);
  font: inherit;
  font-size: 13px;
  font-weight: 650;
  background: color-mix(in srgb, var(--color-surface-base) 92%, transparent);
  border: 1px solid var(--color-border-default);
  border-radius: 999px;
  box-shadow: 0 4px 12px var(--shadow-medium);
  backdrop-filter: blur(12px);
  cursor: pointer;
}

.task-center-launcher__signal {
  width: 8px;
  height: 8px;
  background: var(--color-text-muted);
  border-radius: 50%;
}

.task-center-launcher__signal--active {
  background: var(--color-status-success);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--color-status-success) 18%, transparent);
}

.task-center-launcher__signal--paused {
  background: var(--color-status-warning);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--color-status-warning) 18%, transparent);
}

.task-center-launcher__badge {
  min-width: 20px;
  padding: 1px 6px;
  color: var(--color-text-inverse);
  text-align: center;
  background: var(--color-action-primary);
  border-radius: 999px;
}

.task-center-launcher__interrupted {
  padding: 1px 6px;
  color: var(--color-text-danger);
  font-size: 12px;
  background: var(--color-surface-danger-soft);
  border-radius: 999px;
}

@media (--breakpoint-md-down) {
  .task-center-launcher {
    right: 12px;
    gap: 6px;
    padding: 0 10px;
  }

  .task-center-launcher__label {
    display: none;
  }
}
</style>
