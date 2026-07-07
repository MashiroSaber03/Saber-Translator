<script setup lang="ts">
import TimelineEventCardShell from './TimelineEventCardShell.vue'
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
</script>

<template>
  <TimelineEventCardShell
    :badge-label="`${group.events.length} 个事件`"
    :expanded="expanded"
    :page-range-label="`第 ${group.page_range.start}-${group.page_range.end} 页`"
    :thumbnail-page="group.page_range.start"
    :thumbnail-url="thumbnailUrl"
    :toggle-aria-label="`切换第 ${group.page_range.start}-${group.page_range.end} 页事件`"
    @show-page="$emit('showPage', $event)"
    @toggle="$emit('toggle', group.id)"
  >
    <template v-if="group.summary" #summary>
      {{ group.summary }}
    </template>

    <ul v-if="expanded && group.events.length > 0" class="timeline-group-card__events">
      <li
        v-for="(event, index) in group.events"
        :key="index"
        class="timeline-group-card__event-item"
      >
        {{ event }}
      </li>
    </ul>
  </TimelineEventCardShell>
</template>

<style scoped>
.timeline-group-card__events {
  margin: 0;
  padding: 12px 12px 12px 28px;
  list-style: none;
}

.timeline-group-card__event-item {
  position: relative;
  padding: 6px 0;
  color: var(--insight-text-primary);
  font-size: 13px;
  line-height: 1.5;
}

.timeline-group-card__event-item::before {
  position: absolute;
  top: 12px;
  left: -16px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--insight-action-primary);
  content: '';
}

.timeline-group-card__event-item:not(:last-child) {
  border-bottom: 1px dashed var(--color-border-muted);
}
</style>
