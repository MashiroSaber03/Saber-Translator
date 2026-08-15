<script setup lang="ts">
import { computed } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import TimelineEventCardShell from './TimelineEventCardShell.vue'
import type { TimelineArc } from '@/types/insight'

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

const startPage = computed(() => props.arc.page_range.start)
const endPage = computed(() => props.arc.page_range.end)
</script>

<template>
  <TimelineEventCardShell
    :badge-label="arc.name"
    :expanded="expanded"
    :page-range-label="`第 ${startPage}-${endPage} 页`"
    :thumbnail-page="startPage"
    :thumbnail-url="thumbnailUrl"
    :toggle-aria-label="`切换剧情弧 ${arc.name}`"
    @show-page="$emit('showPage', $event)"
    @toggle="$emit('toggle', arcId)"
  >
    <template #summary>
      {{ arc.description }}
    </template>

    <div v-if="expanded && arc.mood" class="timeline-arc-card__mood">
      <span class="timeline-arc-card__mood-label">
        <UiIcon name="palette" size="13" />
        <span>氛围：</span>
      </span>
      <span>{{ arc.mood }}</span>
    </div>
  </TimelineEventCardShell>
</template>

<style scoped>
.timeline-arc-card__mood {
  padding: 0 12px 12px;
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.timeline-arc-card__mood-label {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--insight-text-secondary);
}
</style>
