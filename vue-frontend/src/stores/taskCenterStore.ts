import { computed, ref, watch, type WatchStopHandle } from 'vue'
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
  'job_interrupted',
  'job_finished',
  'job_failed',
  'chapter_write_intent_created',
  'chapter_write_lock_acquired',
  'step_started',
  'step_checkpointed',
  'step_completed',
  'pipeline_progress',
  'page_completed',
  'page_failed',
  'page_skipped',
  'drain_acknowledged',
  'plugin_stage_completed',
  'plugin_log',
  'plugin_hook_failed',
  'plugin_hook_completed',
  'plugin_agent_state',
  'plugin_agent_assistant_delta',
  'plugin_agent_assistant',
  'plugin_agent_tool_call',
  'plugin_agent_tool_result',
  'plugin_agent_validation',
  'plugin_agent_done',
  'plugin_agent_error',
  'web_import_agent_log',
]

type TaskCenterEventListener = (event: V2JobEvent) => void

const QUEUE_STATUSES = new Set<V2JobStatus>([
  'queued',
  'running',
  'pausing',
  'paused',
  'cancelling',
])
const HISTORY_BATCH_LIMIT = 200

export interface WaitForJobOptions {
  onProgress?: (progress: V2Job['progress']) => void
  signal?: AbortSignal
}

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
  let eventStreamOpenedOnce = false
  let refreshTimer: ReturnType<typeof setTimeout> | null = null
  let refreshPromise: Promise<void> | null = null
  let eventRefreshInFlight = false
  let eventRefreshDirty = false
  let projectionVersion = 0
  const eventListeners = new Set<TaskCenterEventListener>()

  const activeCount = computed(
    () =>
      queue.value.filter(job => ['running', 'pausing', 'paused', 'cancelling'].includes(job.status))
        .length
  )
  const queuedCount = computed(() => queue.value.filter(job => job.status === 'queued').length)
  const interruptedCount = computed(
    () => history.value.filter(job => job.status === 'interrupted').length
  )
  const currentJobs = computed(() =>
    queue.value.filter(job => ['running', 'pausing', 'paused', 'cancelling'].includes(job.status))
  )
  const waitingJobs = computed(() => queue.value.filter(job => job.status === 'queued'))
  const waitingBatches = computed(() => groupJobsByBatch(waitingJobs.value))
  const filteredHistory = computed(() =>
    history.value.filter(
      job =>
        (!statusFilter.value || job.status === statusFilter.value) &&
        (!kindFilter.value || job.kind === kindFilter.value) &&
        (!bookFilter.value || job.bookId === bookFilter.value)
    )
  )
  const historyBatches = computed(() => groupJobsByBatch(filteredHistory.value))

  function refresh(): Promise<void> {
    if (refreshPromise) return refreshPromise
    const startedProjectionVersion = projectionVersion
    loading.value = true
    refreshPromise = Promise.all([
      jobsApi.list('queue'),
      jobsApi.list('history'),
    ]).then(([queueResult, historyResult]) => {
      if (projectionVersion !== startedProjectionVersion) {
        scheduleRefresh()
        return
      }
      queue.value = queueResult.items
      history.value = historyResult.items
      queueRevision.value = queueResult.queueRevision
      workerOnline.value = queueResult.workerOnline !== false
      if (
        !eventSource &&
        lastEventId.value === 0 &&
        typeof queueResult.eventCursor === 'number' &&
        typeof historyResult.eventCursor === 'number'
      ) {
        // The two snapshots are independent reads.  Starting from the older
        // cursor guarantees that an event committed between them is replayed.
        lastEventId.value = Math.min(queueResult.eventCursor, historyResult.eventCursor)
      }
    }).finally(() => {
      loading.value = false
      refreshPromise = null
    })
    return refreshPromise
  }

  function queueOrder(left: V2Job, right: V2Job): number {
    const leftRank = left.queueRank ?? Number.MAX_SAFE_INTEGER
    const rightRank = right.queueRank ?? Number.MAX_SAFE_INTEGER
    if (leftRank !== rightRank) return leftRank - rightRank
    return String(left.createdAt || '').localeCompare(String(right.createdAt || ''))
  }

  function trimHistoryBatches(items: V2Job[]): V2Job[] {
    const retained = new Set<string>()
    return items.filter(job => {
      const key = job.batchId || `job:${job.jobId}`
      if (retained.has(key)) return true
      if (retained.size >= HISTORY_BATCH_LIMIT) return false
      retained.add(key)
      return true
    })
  }

  function applyJobProjection(job: V2Job): void {
    const queueWithoutJob = queue.value.filter(item => item.jobId !== job.jobId)
    const historyWithoutJob = history.value.filter(item => item.jobId !== job.jobId)
    if (QUEUE_STATUSES.has(job.status)) {
      queue.value = [...queueWithoutJob, job].sort(queueOrder)
      history.value = historyWithoutJob
      return
    }

    queue.value = queueWithoutJob
    const previousIndex = history.value.findIndex(item => item.jobId === job.jobId)
    if (previousIndex >= 0) {
      historyWithoutJob.splice(previousIndex, 0, job)
    } else if (job.batchId) {
      const batchIndex = historyWithoutJob.findIndex(item => item.batchId === job.batchId)
      historyWithoutJob.splice(batchIndex >= 0 ? batchIndex : 0, 0, job)
    } else {
      historyWithoutJob.unshift(job)
    }
    history.value = trimHistoryBatches(historyWithoutJob)
  }

  function hasJobProjection(value: unknown): value is V2Job {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return false
    const candidate = value as Partial<V2Job>
    return typeof candidate.jobId === 'string' && typeof candidate.status === 'string'
  }

  function scheduleRefresh(): void {
    eventRefreshDirty = true
    if (refreshTimer || eventRefreshInFlight) return
    refreshTimer = setTimeout(async () => {
      refreshTimer = null
      eventRefreshInFlight = true
      eventRefreshDirty = false
      try {
        await refresh()
      } finally {
        eventRefreshInFlight = false
        if (eventRefreshDirty) scheduleRefresh()
      }
    }, 250)
  }

  function receiveEvent(event: MessageEvent<string>): void {
    try {
      const parsed = JSON.parse(event.data) as V2JobEvent
      if (!Number.isInteger(parsed.eventId) || parsed.eventId < 1) {
        scheduleRefresh()
        return
      }
      if (parsed.eventId <= lastEventId.value) return
      const cursorGap = lastEventId.value > 0 && parsed.eventId !== lastEventId.value + 1
      latestEvent.value = parsed
      lastEventId.value = parsed.eventId
      projectionVersion += 1
      for (const listener of eventListeners) listener(parsed)
      if (hasJobProjection(parsed.job) && parsed.job.jobId === parsed.jobId) {
        applyJobProjection(parsed.job)
        if (typeof parsed.queueRevision === 'number') {
          queueRevision.value = Math.max(queueRevision.value, parsed.queueRevision)
        }
      } else {
        scheduleRefresh()
      }
      if (cursorGap) scheduleRefresh()
    } catch {
      scheduleRefresh()
    }
  }

  function connect(): void {
    if (eventSource) return
    eventSource = new EventSource(`/api/v2/jobs/events?after=${lastEventId.value}`)
    eventSource.onopen = () => {
      connected.value = true
      if (eventStreamOpenedOnce) void refresh()
      eventStreamOpenedOnce = true
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
    eventRefreshDirty = false
    eventSource?.close()
    eventSource = null
    eventStreamOpenedOnce = false
    connected.value = false
  }

  function subscribeEvents(listener: TaskCenterEventListener): () => void {
    eventListeners.add(listener)
    return () => eventListeners.delete(listener)
  }

  async function waitForJob(jobId: string, options: WaitForJobOptions = {}): Promise<V2JobDetail> {
    options.signal?.throwIfAborted()
    await refresh()
    options.signal?.throwIfAborted()

    return new Promise<V2JobDetail>((resolve, reject) => {
      let settled = false
      let terminalDetailLoading = false
      let stop: WatchStopHandle | null = null

      const finish = (action: () => void): void => {
        if (settled) return
        settled = true
        stop?.()
        options.signal?.removeEventListener('abort', abort)
        action()
      }
      const abort = (): void => finish(() => reject(new DOMException('Aborted', 'AbortError')))

      stop = watch(
        () => [...queue.value, ...history.value].find(job => job.jobId === jobId),
        job => {
          if (!job) return
          options.onProgress?.(job.progress)
          if (
            !['completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'].includes(
              job.status
            )
          )
            return
          if (terminalDetailLoading) return
          terminalDetailLoading = true

          void jobsApi.get(jobId).then(
            detail => {
              if (detail.status === 'completed' || detail.status === 'completed_with_errors') {
                finish(() => resolve(detail))
                return
              }
              const rawError = detail.error
              const message =
                typeof rawError === 'string'
                  ? rawError
                  : rawError && typeof rawError === 'object' && typeof rawError.message === 'string'
                    ? rawError.message
                    : detail.status === 'interrupted'
                      ? '任务已中断，请在任务中心继续或取消'
                      : `任务${detail.status === 'cancelled' ? '已取消' : '失败'}`
              finish(() => reject(new Error(message)))
            },
            error => finish(() => reject(error))
          )
        },
        { immediate: true, deep: true }
      )
      options.signal?.addEventListener('abort', abort, { once: true })
      if (settled) stop()
    })
  }

  async function runCommand<T>(command: () => Promise<T>): Promise<T> {
    const result = await command()
    if (refreshPromise) await refreshPromise
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
      const page = await jobsApi.events(detail.jobId, { before: firstEvent.eventId, limit: 50 })
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
    strategy: 'current' | 'original' = 'current'
  ): Promise<JobRetryAccepted> {
    return runCommand(() => jobsApi.retry(jobId, strategy))
  }

  async function retryFailed(
    jobId: string,
    strategy: 'current' | 'original' = 'current'
  ): Promise<JobRetryAccepted> {
    return runCommand(() => jobsApi.retryFailed(jobId, strategy))
  }

  async function retryLatestFailed(
    chapterId: string,
    kinds: V2Job['kind'][],
    strategy: 'current' | 'original' = 'current'
  ): Promise<JobRetryAccepted | null> {
    await refresh()
    const source = history.value.find(
      job =>
        job.chapterId === chapterId &&
        job.status === 'completed_with_errors' &&
        kinds.includes(job.kind)
    )
    return source ? retryFailed(source.jobId, strategy) : null
  }

  async function moveQueued(jobId: string, delta: -1 | 1): Promise<void> {
    const sortable = queue.value.filter(job => job.status === 'queued' && !job.blockedReason)
    const index = sortable.findIndex(job => job.jobId === jobId)
    const target = index + delta
    if (index < 0 || target < 0 || target >= sortable.length) return
    const ordered = sortable.map(job => job.jobId)
    ;[ordered[index], ordered[target]] = [ordered[target]!, ordered[index]!]
    await runCommand(() => jobsApi.reorder(ordered, queueRevision.value))
  }

  async function prioritizeQueued(jobId: string): Promise<void> {
    const sortable = queue.value.filter(job => job.status === 'queued' && !job.blockedReason)
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
    waitingBatches,
    historyBatches,
    initialize,
    disconnect,
    subscribeEvents,
    waitForJob,
    refresh,
    open: (target?: TaskCenterFocus) => {
      focusTarget.value = target || null
      drawerOpen.value = true
      void refresh()
    },
    close: () => {
      drawerOpen.value = false
    },
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
    prioritizeBatch: (batchId: string) =>
      runCommand(() => jobsApi.prioritizeBatch(batchId, queueRevision.value)),
    continueBatch: (batchId: string) => runCommand(() => jobsApi.continueBatch(batchId)),
    cancelQueued: () => runCommand(() => jobsApi.cancelQueued()),
    clearHistory: () => runCommand(() => jobsApi.clearHistory()),
  }
})
