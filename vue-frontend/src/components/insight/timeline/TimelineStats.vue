<script setup lang="ts">
import type { TimelineStats } from './timelineTypes'

defineProps<{
  stats?: TimelineStats
  totalEvents: number
  totalPages: number
}>()
</script>

<template>
  <div class="timeline-stats">
    <span v-if="stats?.total_arcs" class="stat-badge">🎭 {{ stats.total_arcs }} 个剧情弧</span>
    <span class="stat-badge">📊 {{ totalEvents }} 个事件</span>
    <span v-if="stats?.total_characters" class="stat-badge">👥 {{ stats.total_characters }} 个角色</span>
    <span v-if="stats?.total_threads" class="stat-badge">🔗 {{ stats.total_threads }} 条线索</span>
    <span class="stat-badge">📄 {{ totalPages }} 页</span>
  </div>
</template>

<style scoped>
.timeline-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--color-border-muted);
}

.stat-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  background: var(--insight-bg-tertiary);
  border-radius: 16px;
  font-size: 13px;
  color: var(--insight-text-secondary);
}
</style>
