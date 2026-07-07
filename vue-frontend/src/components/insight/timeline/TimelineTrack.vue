<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import TimelineArcCard from './TimelineArcCard.vue'
import TimelineGroupCard from './TimelineGroupCard.vue'
import type { TimelineArc, TimelineGroup } from './timelineTypes'

const props = defineProps<{
  expandedIds: string[]
  groups: TimelineGroup[]
  isEnhancedData: boolean
  plotArcs: TimelineArc[]
  thumbnailUrlFor: (pageNum: number) => string
}>()

defineEmits<{
  (event: 'showPage', pageNum: number): void
  (event: 'toggle', id: string): void
}>()

function arcId(arc: TimelineArc, index: number): string {
  return arc.id || `arc-${index}`
}

function arcStartPage(arc: TimelineArc): number {
  return arc.page_range?.start || arc.start_page || 1
}

function isExpanded(id: string): boolean {
  return props.expandedIds.includes(id)
}
</script>

<template>
  <div v-if="isEnhancedData && plotArcs.length > 0" class="timeline-track">
    <div
      v-for="(arc, index) in plotArcs"
      :key="arcId(arc, index)"
      class="timeline-track__group"
      :class="{ 'timeline-track__group--expanded': isExpanded(arcId(arc, index)) }"
    >
      <div class="timeline-track__node">
        <UiButton
          variant="toolbar"
          type="button"
          class="timeline-track__node-dot"
          :aria-label="`切换剧情弧 ${arc.name}`"
          :aria-expanded="String(isExpanded(arcId(arc, index)))"
          @click="$emit('toggle', arcId(arc, index))"
        ></UiButton>
        <div class="timeline-track__node-line"></div>
      </div>
      <TimelineArcCard
        :arc="arc"
        :arc-id="arcId(arc, index)"
        :expanded="isExpanded(arcId(arc, index))"
        :thumbnail-url="thumbnailUrlFor(arcStartPage(arc))"
        @show-page="$emit('showPage', $event)"
        @toggle="$emit('toggle', $event)"
      />
    </div>
  </div>

  <div v-else-if="groups.length > 0" class="timeline-track">
    <div
      v-for="group in groups"
      :key="group.id"
      class="timeline-track__group"
      :class="{ 'timeline-track__group--expanded': isExpanded(group.id) }"
    >
      <div class="timeline-track__node">
        <UiButton
          variant="toolbar"
          type="button"
          class="timeline-track__node-dot"
          :aria-label="`切换第 ${group.page_range.start}-${group.page_range.end} 页事件`"
          :aria-expanded="String(isExpanded(group.id))"
          @click="$emit('toggle', group.id)"
        ></UiButton>
        <div class="timeline-track__node-line"></div>
      </div>
      <TimelineGroupCard
        :expanded="isExpanded(group.id)"
        :group="group"
        :thumbnail-url="thumbnailUrlFor(group.thumbnail_page || group.page_range.start)"
        @show-page="$emit('showPage', $event)"
        @toggle="$emit('toggle', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
.timeline-track {
  position: relative;
  padding-left: 20px;
}

.timeline-track__group {
  display: flex;
  gap: 16px;
  position: relative;
  margin-bottom: 24px;
}

.timeline-track__node {
  display: flex;
  flex-shrink: 0;
  flex-direction: column;
  align-items: center;
  width: 20px;
}

.timeline-track__node-dot {
  width: 14px;
  height: 14px;
  padding: 0;
  border: 3px solid var(--insight-surface-page);
  border-radius: 50%;
  background: var(--insight-action-primary);
  box-shadow: 0 0 0 2px var(--insight-action-primary);
  cursor: pointer;
  transition: transform 0.2s;
  z-index: var(--z-local);
}

.timeline-track__node-dot:hover {
  transform: scale(1.2);
}

.timeline-track__node-line {
  flex: 1;
  width: 2px;
  margin-top: 4px;
  background: linear-gradient(180deg, var(--insight-action-primary), var(--color-border-muted));
}

.timeline-track__group:last-child .timeline-track__node-line {
  display: none;
}
</style>
