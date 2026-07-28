import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { jobsApi, type V2Job, type V2JobEvent } from '@/api/v2/jobs'
import { groupJobsByBatch } from '@/stores/taskCenterProjection'

const EVENT_TYPES = [
  'job_created',
  'job_reordered',
  'job_started',
  'job_request_pause',
  'job_request_cancel',
  'job_resume',
  'job_continue',
  'job_paused',
  'job_cancelled',
  'job_finished',
  'job_failed',
  'chapter_write_intent_created',
  'chapter_write_lock_acquired',
  'step_started',
  'step_completed',
  'page_completed',
  'page_failed',
  'drain_acknowledged',
]

export const useTaskCenterStore = defineStore('taskCenter', () => {
  const queue = ref<V2Job[]>([])
  const history = ref<V2Job[]>([])
  const queueRevision = ref(1)
  const drawerOpen = ref(false)
  const loading = ref(false)
  const connected = ref(false)
  const lastEventId = ref(0)
  const latestEvent = ref<V2JobEvent | null>(null)
  let eventSource: EventSource | null = null
  let refreshTimer: ReturnType<typeof setTimeout> | null = null

  const activeCount = computed(() => (
    queue.value.filter(job => ['running', 'pausing', 'cancelling'].includes(job.status)).length
  ))
  const queuedCount = computed(() => queue.value.filter(job => job.status === 'queued').length)
  const queueBatches = computed(() => groupJobsByBatch(queue.value))
  const historyBatches = computed(() => groupJobsByBatch(history.value))

  async function refresh(): Promise<void> {
    loading.value = true
    try {
      const [queueResult, historyResult] = await Promise.all([
        jobsApi.list('queue'),
        jobsApi.list('history'),
      ])
      queue.value = queueResult.items
      history.value = historyResult.items
      queueRevision.value = queueResult.queueRevision
    } finally {
      loading.value = false
    }
  }

  function scheduleRefresh(): void {
    if (refreshTimer) return
    refreshTimer = setTimeout(() => {
      refreshTimer = null
      void refresh()
    }, 100)
  }

  function receiveEvent(event: MessageEvent<string>): void {
    try {
      const parsed = JSON.parse(event.data) as V2JobEvent
      latestEvent.value = parsed
      lastEventId.value = Math.max(lastEventId.value, parsed.eventId)
      scheduleRefresh()
    } catch {
      scheduleRefresh()
    }
  }

  function connect(): void {
    if (eventSource) return
    eventSource = new EventSource(`/api/v2/jobs/events?after=${lastEventId.value}`)
    eventSource.onopen = () => {
      connected.value = true
    }
    eventSource.onerror = () => {
      connected.value = false
    }
    for (const eventType of EVENT_TYPES) {
      eventSource.addEventListener(eventType, receiveEvent as EventListener)
    }
  }

  async function initialize(): Promise<void> {
    await refresh()
    connect()
  }

  function disconnect(): void {
    if (refreshTimer) clearTimeout(refreshTimer)
    refreshTimer = null
    eventSource?.close()
    eventSource = null
    connected.value = false
  }

  async function runCommand(command: () => Promise<unknown>): Promise<void> {
    await command()
    await refresh()
  }

  return {
    queue,
    history,
    queueRevision,
    drawerOpen,
    loading,
    connected,
    latestEvent,
    activeCount,
    queuedCount,
    queueBatches,
    historyBatches,
    initialize,
    disconnect,
    refresh,
    open: () => { drawerOpen.value = true },
    close: () => { drawerOpen.value = false },
    pause: (jobId: string) => runCommand(() => jobsApi.pause(jobId)),
    resume: (jobId: string) => runCommand(() => jobsApi.resume(jobId)),
    continueJob: (jobId: string) => runCommand(() => jobsApi.continue(jobId)),
    cancel: (jobId: string) => runCommand(() => jobsApi.cancel(jobId)),
    cancelQueued: () => runCommand(() => jobsApi.cancelQueued()),
    clearHistory: () => runCommand(() => jobsApi.clearHistory()),
  }
})
