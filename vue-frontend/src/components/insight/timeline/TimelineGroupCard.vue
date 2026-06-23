<script setup lang="ts">
import type { TimelineGroup } from './timelineTypes'

defineProps<{
  expanded: boolean
  group: TimelineGroup
  thumbnailUrl: string
}>()

defineEmits<{
  (event: 'showPage', pageNum: number): void
  (event: 'toggle', id: string): void
}>()

function hideFailedImage(event: Event): void {
  const image = event.target as HTMLImageElement
  image.style.display = 'none'
}
</script>

<template>
  <div class="timeline-card">
    <div class="timeline-card-header" @click="$emit('toggle', group.id)">
      <img
        class="timeline-thumbnail"
        :src="thumbnailUrl"
        :alt="`第${group.page_range.start}页`"
        loading="lazy"
        @error="hideFailedImage"
        @click.stop="$emit('showPage', group.page_range.start)"
      >
      <div class="timeline-card-title">
        <span class="timeline-page-range">第 {{ group.page_range.start }}-{{ group.page_range.end }} 页</span>
        <span class="timeline-event-count">{{ group.events.length }} 个事件</span>
      </div>
      <span class="expand-icon">{{ expanded ? '▼' : '▶' }}</span>
    </div>

    <div v-if="group.summary" class="timeline-summary">{{ group.summary }}</div>

    <ul v-if="expanded && group.events.length > 0" class="timeline-events-list">
      <li
        v-for="(event, index) in group.events"
        :key="index"
        class="timeline-event-item"
      >
        {{ event }}
      </li>
    </ul>
  </div>
</template>

<style scoped>
.timeline-card {
  flex: 1;
  background: var(--insight-bg-secondary);
  border-radius: 12px;
  border: 1px solid var(--color-border-muted);
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.timeline-card:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 12px var(--timeline-panel-card-shadow);
}

.timeline-card-header {
  display: flex;
  gap: 12px;
  padding: 12px;
  background: var(--insight-bg-tertiary);
  border-bottom: 1px solid var(--color-border-muted);
  cursor: pointer;
}

.timeline-thumbnail {
  width: 60px;
  height: 80px;
  object-fit: cover;
  border-radius: 6px;
  cursor: pointer;
  transition: transform 0.2s;
  background: var(--insight-bg-primary);
}

.timeline-thumbnail:hover {
  transform: scale(1.05);
}

.timeline-card-title {
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
}

.timeline-page-range {
  font-weight: 600;
  font-size: 15px;
  color: var(--insight-text-primary);
}

.timeline-event-count {
  width: fit-content;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--insight-bg-primary);
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.expand-icon {
  align-self: center;
  color: var(--insight-text-secondary);
  font-size: 10px;
}

.timeline-summary {
  padding: 12px;
  border-bottom: 1px solid var(--color-border-muted);
  color: var(--insight-text-secondary);
  font-size: 14px;
  line-height: 1.6;
}

.timeline-events-list {
  margin: 0;
  padding: 12px 12px 12px 28px;
  list-style: none;
}

.timeline-event-item {
  position: relative;
  padding: 6px 0;
  color: var(--insight-text-primary);
  font-size: 13px;
  line-height: 1.5;
}

.timeline-event-item::before {
  position: absolute;
  top: 12px;
  left: -16px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--insight-color-primary);
  content: '';
}

.timeline-event-item:not(:last-child) {
  border-bottom: 1px dashed var(--color-border-muted);
}
</style>
