<script setup lang="ts">
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const store = useTaskCenterStore()
</script>

<template>
  <button
    class="task-center-launcher"
    type="button"
    :aria-label="`打开任务中心，${store.activeCount} 个运行中，${store.queuedCount} 个排队中`"
    @click="store.open"
  >
    <span class="task-center-launcher__signal" :class="{ 'task-center-launcher__signal--active': store.activeCount > 0 }" />
    <span>任务中心</span>
    <span v-if="store.activeCount + store.queuedCount" class="task-center-launcher__badge">
      {{ store.activeCount + store.queuedCount }}
    </span>
  </button>
</template>

<style scoped>
.task-center-launcher {
  position: fixed;
  z-index: calc(var(--z-modal, 1000) - 2);
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
  background: color-mix(in srgb, var(--color-surface) 92%, transparent);
  border: 1px solid var(--color-border);
  border-radius: 999px;
  box-shadow: var(--shadow-md);
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
  background: var(--color-success, #22c55e);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--color-success, #22c55e) 18%, transparent);
}

.task-center-launcher__badge {
  min-width: 20px;
  padding: 1px 6px;
  color: var(--color-on-primary, #fff);
  text-align: center;
  background: var(--color-primary);
  border-radius: 999px;
}
</style>
