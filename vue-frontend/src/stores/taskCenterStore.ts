import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import {
  jobsApi,
  type JobRetryAccepted,
  type V2Job,
  type V2JobDetail,
  type V2JobEvent,
  type V2JobStatus,
} from '@/api/v2/jobs'
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

export interface TaskCenterFocus {
  jobId?: string
  batchId?: string
  bookId?: string
  chapterId?: string
}

export const useTaskCenterStore = defineStore('taskCenter', () => {
  const queue = ref<V2Job[]>([])
  const history = ref<V2Job[]>([])
  const queueRevision = ref(1)
  const drawerOpen = ref(false)
  const focusTarget = ref<TaskCenterFocus | null>(null)
  const loading = ref(false)
  const connected = ref(false)
  const workerOnline = ref(true)
  const lastEventId = ref(0)
  const latestEvent = ref<V2JobEvent | null>(null)
  const selectedDetail = ref<V2JobDetail | null>(null)
  const detailLoading = ref(false)
  const olderEventsLoading = ref(false)
  const olderEventsExhausted = ref(false)
  const statusFilter = ref<'' | V2JobStatus>('')
  const kindFilter = ref<'' | V2Job['kind']>('')
  const bookFilter = ref('')
  let eventSource: EventSource | null = null
  let refreshTimer: ReturnType<typeof setTimeout> | null = null

  const activeCount = computed(() => (
    queue.value.filter(
      job => ['running', 'pausing', 'paused', 'cancelling'].includes(job.status),
    ).length
  ))
  const queuedCount = computed(() => queue.value.filter(job => job.status === 'queued').length)
  const interruptedCount = computed(() => (
    history.value.filter(job => job.status === 'interrupted').length
  ))
  const currentJobs = computed(() => queue.value.filter(
    job => ['running', 'pausing', 'paused', 'cancelling'].includes(job.status),
  ))
  const waitingJobs = computed(() => queue.value.filter(job => job.status === 'queued'))
  const queueBatches = computed(() => groupJobsByBatch(queue.value))
  const waitingBatches = computed(() => groupJobsByBatch(waitingJobs.value))
  const historyBatches = computed(() => groupJobsByBatch(history.value))

  async function refresh(): Promise<void> {
    loading.value = true
    try {
      const [queueResult, historyResult] = await Promise.all([
        jobsApi.list('queue'),
        jobsApi.list('history', {
          ...(statusFilter.value ? { status: statusFilter.value } : {}),
          ...(kindFilter.value ? { type: kindFilter.value } : {}),
          ...(bookFilter.value ? { bookId: bookFilter.value } : {}),
        }),
      ])
      queue.value = queueResult.items
      history.value = historyResult.items
      queueRevision.value = queueResult.queueRevision
      workerOnline.value = queueResult.workerOnline !== false
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
      void refresh()
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

  async function runCommand<T>(command: () => Promise<T>): Promise<T> {
    const result = await command()
    await refresh()
    return result
  }

  async function loadDetail(jobId: string): Promise<void> {
    detailLoading.value = true
    olderEventsExhausted.value = false
    try {
      selectedDetail.value = await jobsApi.get(jobId)
    } finally {
      detailLoading.value = false
    }
  }

  async function loadOlderEvents(): Promise<void> {
    const detail = selectedDetail.value
    const firstEvent = detail?.recentEvents[0]
    if (!detail || !firstEvent || olderEventsExhausted.value) return
    olderEventsLoading.value = true
    try {
      const page = await jobsApi.events(
        detail.jobId,
        { before: firstEvent.eventId, limit: 50 },
      )
      if (!page.items.length) {
        olderEventsExhausted.value = true
        return
      }
      const known = new Set(detail.recentEvents.map(event => event.eventId))
      selectedDetail.value = {
        ...detail,
        recentEvents: [
          ...page.items.filter(event => !known.has(event.eventId)),
          ...detail.recentEvents,
        ],
      }
    } finally {
      olderEventsLoading.value = false
    }
  }

  async function retry(
    jobId: string,
    strategy: 'current' | 'original' = 'current',
  ): Promise<JobRetryAccepted> {
    return runCommand(() => jobsApi.retry(jobId, strategy))
  }

  async function retryFailed(
    jobId: string,
    strategy: 'current' | 'original' = 'current',
  ): Promise<JobRetryAccepted> {
    return runCommand(() => jobsApi.retryFailed(jobId, strategy))
  }

  async function retryLatestFailed(
    chapterId: string,
    kinds: V2Job['kind'][],
    strategy: 'current' | 'original' = 'current',
  ): Promise<JobRetryAccepted | null> {
    await refresh()
    const source = history.value.find(job => (
      job.chapterId === chapterId
      && job.status === 'completed_with_errors'
      && kinds.includes(job.kind)
    ))
    return source ? retryFailed(source.jobId, strategy) : null
  }

  async function moveQueued(jobId: string, delta: -1 | 1): Promise<void> {
    const sortable = queue.value.filter(job => (
      job.status === 'queued' && !job.blockedReason
    ))
    const index = sortable.findIndex(job => job.jobId === jobId)
    const target = index + delta
    if (index < 0 || target < 0 || target >= sortable.length) return
    const ordered = sortable.map(job => job.jobId)
    ;[ordered[index], ordered[target]] = [ordered[target]!, ordered[index]!]
    await runCommand(() => jobsApi.reorder(ordered, queueRevision.value))
  }

  async function prioritizeQueued(jobId: string): Promise<void> {
    const sortable = queue.value.filter(job => (
      job.status === 'queued' && !job.blockedReason
    ))
    const index = sortable.findIndex(job => job.jobId === jobId)
    if (index <= 0) return
    const ordered = sortable.map(job => job.jobId)
    ordered.splice(index, 1)
    ordered.unshift(jobId)
    await runCommand(() => jobsApi.reorder(ordered, queueRevision.value))
  }

  return {
    queue,
    history,
    queueRevision,
    drawerOpen,
    focusTarget,
    loading,
    connected,
    workerOnline,
    latestEvent,
    selectedDetail,
    detailLoading,
    olderEventsLoading,
    olderEventsExhausted,
    statusFilter,
    kindFilter,
    bookFilter,
    activeCount,
    queuedCount,
    interruptedCount,
    currentJobs,
    queueBatches,
    waitingBatches,
    historyBatches,
    initialize,
    disconnect,
    refresh,
    open: (target?: TaskCenterFocus) => {
      focusTarget.value = target || null
      drawerOpen.value = true
      void refresh()
    },
    close: () => { drawerOpen.value = false },
    pause: (jobId: string) => runCommand(() => jobsApi.pause(jobId)),
    resume: (jobId: string) => runCommand(() => jobsApi.resume(jobId)),
    continueJob: (jobId: string) => runCommand(() => jobsApi.continue(jobId)),
    cancel: (jobId: string) => runCommand(() => jobsApi.cancel(jobId)),
    retry,
    retryFailed,
    retryLatestFailed,
    loadDetail,
    loadOlderEvents,
    moveQueued,
    prioritizeQueued,
    cancelBatch: (batchId: string) => runCommand(() => jobsApi.cancelBatch(batchId)),
    prioritizeBatch: (batchId: string) => (
      runCommand(() => jobsApi.prioritizeBatch(batchId, queueRevision.value))
    ),
    continueBatch: (batchId: string) => runCommand(() => jobsApi.continueBatch(batchId)),
    cancelQueued: () => runCommand(() => jobsApi.cancelQueued()),
    clearHistory: () => runCommand(() => jobsApi.clearHistory()),
  }
})
