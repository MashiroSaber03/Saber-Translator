<script setup lang="ts">
import type { PlotThread } from './timelineTypes'

defineProps<{
  threads: PlotThread[]
}>()

function isResolved(thread: PlotThread): boolean {
  return thread.status === '已解决'
}
</script>

<template>
  <div class="plot-threads-list">
    <div
      v-for="thread in threads"
      :key="thread.id"
      class="plot-thread-item"
      :class="{ resolved: isResolved(thread) }"
    >
      <div class="thread-header">
        <span class="thread-name">{{ thread.name || '未命名线索' }}</span>
        <span class="thread-status" :class="{ resolved: isResolved(thread) }">
          {{ thread.status || '进行中' }}
        </span>
      </div>
      <p v-if="thread.description" class="thread-desc">{{ thread.description }}</p>
      <span v-if="thread.introduced_at" class="thread-intro">第 {{ thread.introduced_at }} 页引入</span>
    </div>
  </div>
</template>

<style scoped>
.plot-threads-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plot-thread-item {
  padding: 14px;
  border-left: 3px solid var(--color-status-warning);
  border-radius: 10px;
  background: var(--insight-bg-secondary);
}

.plot-thread-item.resolved {
  border-left-color: var(--color-status-success);
  opacity: 0.8;
}

.thread-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.thread-name {
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 14px;
}

.thread-status {
  flex-shrink: 0;
  padding: 3px 10px;
  border-radius: 10px;
  background: var(--color-status-warning);
  color: white;
  font-size: 11px;
}

.thread-status.resolved {
  background: var(--color-status-success);
}

.thread-desc {
  margin: 0 0 8px;
  color: var(--insight-text-secondary);
  font-size: 13px;
  line-height: 1.5;
}

.thread-intro {
  color: var(--insight-text-muted);
  font-size: 12px;
}
</style>
