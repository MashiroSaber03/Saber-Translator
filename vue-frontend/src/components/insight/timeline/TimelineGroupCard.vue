<script setup lang="ts">
import TimelineEventCardShell from './TimelineEventCardShell.vue'
import type { TimelineGroup } from '@/types/insight'

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
    :thumbnail-page="group.thumbnail_page"
    :thumbnail-url="thumbnailUrl"
    :toggle-aria-label="`切换第 ${group.page_range.start}-${group.page_range.end} 页事件`"
    @show-page="$emit('showPage', $event)"
    @toggle="$emit('toggle', group.id)"
  >
    <template #summary>
      {{ group.summary }}
    </template>
  </TimelineEventCardShell>
</template>
