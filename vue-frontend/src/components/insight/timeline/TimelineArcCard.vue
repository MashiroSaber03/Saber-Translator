<script setup lang="ts">
import { computed } from 'vue'
import type { TimelineArc } from './timelineTypes'

const props = defineProps<{
  arc: TimelineArc
  arcId: string
  expanded: boolean
  thumbnailUrl: string
}>()

defineEmits<{
  (event: 'showPage', pageNum: number): void
  (event: 'toggle', id: string): void
}>()

const startPage = computed(() => props.arc.page_range?.start || props.arc.start_page || 1)
const endPage = computed(() => props.arc.page_range?.end || props.arc.end_page || '?')

function hideFailedImage(event: Event): void {
  const image = event.target as HTMLImageElement
  image.style.display = 'none'
}
</script>

<template>
  <div class="timeline-card">
    <div class="timeline-card-header" @click="$emit('toggle', arcId)">
      <img
        class="timeline-thumbnail"
        :src="thumbnailUrl"
        :alt="`第${startPage}页`"
        loading="lazy"
        @error="hideFailedImage"
        @click.stop="$emit('showPage', startPage)"
      >
      <div class="timeline-card-title">
        <span class="timeline-page-range">第 {{ startPage }}-{{ endPage }} 页</span>
        <span class="timeline-event-count">{{ arc.name }}</span>
      </div>
      <span class="expand-icon">{{ expanded ? '▼' : '▶' }}</span>
    </div>

    <div v-if="arc.description" class="timeline-summary">{{ arc.description }}</div>

    <div v-if="arc.mood" class="timeline-mood">
      <span class="label">🎨 氛围：</span>{{ arc.mood }}
    </div>
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
  min-width: 0;
}

.timeline-page-range {
  font-weight: 600;
  font-size: 15px;
  color: var(--insight-text-primary);
}

.timeline-event-count {
  width: fit-content;
  max-width: 100%;
  padding: 2px 8px;
  overflow: hidden;
  border-radius: 10px;
  background: var(--insight-bg-primary);
  color: var(--insight-text-secondary);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
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

.timeline-mood {
  padding: 0 12px 12px;
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.label {
  color: var(--insight-text-secondary);
}
</style>
