<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { JobStatusSummary } from '@/types/bookshelf'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const props = withDefaults(defineProps<{
  bookId?: string
  chapterId?: string
  summary?: JobStatusSummary
}>(), {
  bookId: '',
  chapterId: '',
  summary: () => ({}),
})

const store = useTaskCenterStore()

const liveJobs = computed(() => (
  [...store.queue, ...store.history].filter(job => (
    (!props.bookId || job.bookId === props.bookId)
    && (!props.chapterId || job.chapterId === props.chapterId)
    && (!props.chapterId || job.kind === 'translation')
    && [
      'queued',
      'running',
      'paused',
      'interrupted',
      'failed',
    ].includes(job.status)
  ))
))

const resolved = computed(() => {
  const liveCounts = liveJobs.value.reduce<JobStatusSummary>((result, job) => {
    result[job.status as keyof JobStatusSummary] = (
      result[job.status as keyof JobStatusSummary] || 0
    ) + 1
    return result
  }, {})
  const counts: JobStatusSummary = store.snapshotLoaded || liveJobs.value.length > 0
    ? {
        ...liveCounts,
        interrupted: Math.max(liveCounts.interrupted || 0, props.summary.interrupted || 0),
        failed: Math.max(liveCounts.failed || 0, props.summary.failed || 0),
      }
    : props.summary
  const entries = [
    { statuses: ['interrupted'], tone: 'danger', label: '中断' },
    { statuses: ['failed'], tone: 'danger', label: '失败' },
    { statuses: ['paused'], tone: 'warning', label: '暂停' },
    { statuses: ['running'], tone: 'success', label: '进行中' },
    { statuses: ['queued'], tone: 'neutral', label: '排队' },
  ] as const
  for (const entry of entries) {
    const count = entry.statuses.reduce((sum, status) => sum + (counts[status] || 0), 0)
    if (count) return { ...entry, count }
  }
  return null
})

function openTaskCenter() {
  store.open({
    bookId: props.bookId || undefined,
    chapterId: props.chapterId || undefined,
  })
}
</script>

<template>
  <UiButton
    v-if="resolved"
    class="task-status-badge"
    size="xs"
    variant="ghost"
    :tone="resolved.tone"
    :title="`在任务中心查看${resolved.label}任务`"
    @click.stop="openTaskCenter"
  >
    {{ resolved.label }} {{ resolved.count }}
  </UiButton>
</template>

<style scoped>
.task-status-badge {
  min-height: 24px;
  padding: 2px 8px;
  border-radius: 999px;
}
</style>
