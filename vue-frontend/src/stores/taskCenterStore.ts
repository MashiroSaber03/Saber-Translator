import { computed, ref, watch, type WatchStopHandle } from 'vue'
import { defineStore } from 'pinia'
import {
  CURRENT_JOB_STATUSES,
  HISTORY_JOB_STATUSES,
  NONTERMINAL_JOB_STATUSES,
  jobsApi,
  type JobRetryAccepted,
  type V2Job,
  type V2JobDetail,
  type V2JobEvent,
  type V2JobStatus,
} from '@/api/v2/jobs'
import { groupJobsByBatch } from '@/stores/taskCenterProjection'
import { TASK_EVENT_TYPES } from '@/utils/taskDisplay'

type TaskCenterEventListener = (event: V2JobEvent) => void

const QUEUE_STATUSES = new Set<V2JobStatus>([
  'queued',
  ...CURRENT_JOB_STATUSES,
])
const HISTORY_BATCH_LIMIT = 200
const JOB_EVENT_FIELDS = new Set(['eventId', 'jobId', 'type', 'payload', 'createdAt'])

function parseJobEvent(value: unknown): V2JobEvent | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const record = value as Record<string, unknown>
  if (
    Object.keys(record).length !== JOB_EVENT_FIELDS.size
    || Object.keys(record).some(key => !JOB_EVENT_FIELDS.has(key))
    || !Number.isInteger(record.eventId)
    || (record.eventId as number) < 1
    || typeof record.jobId !== 'string'
    || record.jobId.length === 0
    || typeof record.type !== 'string'
    || record.type.length === 0
    || !record.payload
    || typeof record.payload !== 'object'
    || Array.isArray(record.payload)
    || !(
      record.createdAt === null
      || (
        typeof record.createdAt === 'string'
        && record.createdAt.length > 0
        && Number.isFinite(Date.parse(record.createdAt))
      )
    )
  ) return null
  return record as unknown as V2JobEvent
}

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
  const snapshotLoaded = ref(false)
  const connected = ref(false)
  const workerOnline = ref(true)
  const lastEventId = ref(0)
  const latestEvent = ref<V2JobEvent | null>(null)
  const selectedDetail = ref<V2JobDetail | null>(null)
  const selectedDetailJobId = ref<string | null>(null)
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
  let projectionTimer: ReturnType<typeof setTimeout> | null = null
  let projectionPromise: Promise<void> | null = null
  let detailProjectionTimer: ReturnType<typeof setTimeout> | null = null
  let detailProjectionPromise: Promise<void> | null = null
  let detailProjectionDirty = false
  let detailRequestVersion = 0
  const pendingProjectionJobIds = new Set<string>()
  const eventListeners = new Set<TaskCenterEventListener>()

  const activeCount = computed(
    () => queue.value.filter(job => CURRENT_JOB_STATUSES.has(job.status)).length
  )
  const queuedCount = computed(() => queue.value.filter(job => job.status === 'queued').length)
  const interruptedCount = computed(
    () => history.value.filter(job => job.status === 'interrupted').length
  )
  const currentJobs = computed(() =>
    queue.value.filter(job => CURRENT_JOB_STATUSES.has(job.status))
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
      snapshotLoaded.value = true
      if (selectedDetail.value) {
        scheduleSelectedDetailRefresh(selectedDetail.value.jobId)
      }
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
    const batchKey = (job: V2Job) => job.batchId || `job:${job.jobId}`
    const interruptedBatches = new Set(
      items.filter(job => job.status === 'interrupted').map(batchKey),
    )
    const retainedTerminalBatches = new Set<string>()
    return items.filter(job => {
      const key = batchKey(job)
      if (interruptedBatches.has(key)) return true
      if (retainedTerminalBatches.has(key)) return true
      if (retainedTerminalBatches.size >= HISTORY_BATCH_LIMIT) return false
      retainedTerminalBatches.add(key)
      return true
    })
  }

  function applyJobProjection(job: V2Job): void {
    scheduleSelectedDetailRefresh(job.jobId)
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

  function mergeDetailEvents(
    previous: V2JobEvent[],
    current: V2JobEvent[],
  ): V2JobEvent[] {
    const byId = new Map<number, V2JobEvent>()
    for (const event of [...previous, ...current]) {
      byId.set(event.eventId, event)
    }
    return [...byId.values()].sort((left, right) => left.eventId - right.eventId)
  }

  function scheduleSelectedDetailRefresh(jobId: string): void {
    if (
      !drawerOpen.value
      || selectedDetail.value?.jobId !== jobId
    ) return
    detailProjectionDirty = true
    if (detailProjectionTimer || detailProjectionPromise) return
    detailProjectionTimer = setTimeout(() => {
      detailProjectionTimer = null
      void flushSelectedDetailRefresh()
    }, 100)
  }

  async function flushSelectedDetailRefresh(): Promise<void> {
    const jobId = selectedDetail.value?.jobId
    if (
      detailProjectionPromise
      || !detailProjectionDirty
      || !drawerOpen.value
      || !jobId
    ) return
    detailProjectionDirty = false
    const request = jobsApi.get(jobId).then((detail) => {
      const previous = selectedDetail.value
      if (previous?.jobId !== jobId) return
      selectedDetail.value = {
        ...detail,
        recentEvents: mergeDetailEvents(previous.recentEvents, detail.recentEvents),
      }
    }).catch(() => {
      // The durable queue/history projection remains usable if the optional
      // expanded-detail refresh races with a transient request failure.
    }).finally(() => {
      detailProjectionPromise = null
      const selectedJobId = selectedDetail.value?.jobId
      if (detailProjectionDirty && selectedJobId) {
        scheduleSelectedDetailRefresh(selectedJobId)
      }
    })
    detailProjectionPromise = request
    await request
  }

  function removeJobProjection(jobId: string): void {
    queue.value = queue.value.filter(item => item.jobId !== jobId)
    history.value = history.value.filter(item => item.jobId !== jobId)
  }

  function scheduleJobProjection(jobId: string): void {
    pendingProjectionJobIds.add(jobId)
    if (projectionTimer || projectionPromise) return
    projectionTimer = setTimeout(() => {
      projectionTimer = null
      void flushJobProjections()
    }, 100)
  }

  async function flushJobProjections(): Promise<void> {
    if (projectionPromise || pendingProjectionJobIds.size === 0) return
    const jobIds = [...pendingProjectionJobIds].slice(0, 200)
    for (const jobId of jobIds) pendingProjectionJobIds.delete(jobId)
    const request = jobsApi.snapshot(jobIds).then(result => {
      const found = new Set(result.items.map(job => job.jobId))
      for (const job of result.items) applyJobProjection(job)
      for (const jobId of jobIds) {
        if (!found.has(jobId)) removeJobProjection(jobId)
      }
      queueRevision.value = Math.max(queueRevision.value, result.queueRevision)
      projectionVersion += 1
    }).catch(() => {
      scheduleRefresh()
    }).finally(() => {
      projectionPromise = null
      if (pendingProjectionJobIds.size > 0) {
        const nextJobId = pendingProjectionJobIds.values().next().value
        if (nextJobId) scheduleJobProjection(nextJobId)
      }
    })
    projectionPromise = request
    await request
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
      } catch {
        // A later SSE event, reconnect, drawer open, or explicit action retries
        // the durable snapshot. Background refresh failures must not surface as
        // unhandled promise rejections.
      } finally {
        eventRefreshInFlight = false
        if (eventRefreshDirty) scheduleRefresh()
      }
    }, 250)
  }

  function receiveEvent(event: MessageEvent<string>, expectedType: string): void {
    try {
      const parsed = parseJobEvent(JSON.parse(event.data))
      if (!parsed || parsed.type !== expectedType) {
        scheduleRefresh()
        return
      }
      if (parsed.eventId <= lastEventId.value) return
      const cursorGap = lastEventId.value > 0 && parsed.eventId !== lastEventId.value + 1
      latestEvent.value = parsed
      lastEventId.value = parsed.eventId
      for (const listener of eventListeners) listener(parsed)
      if (cursorGap) scheduleRefresh()
      else scheduleJobProjection(parsed.jobId)
    } catch {
      scheduleRefresh()
    }
  }

  function connect(): void {
    if (eventSource) return
    eventSource = new EventSource(`/api/v2/jobs/events?after=${lastEventId.value}`)
    eventSource.onopen = () => {
      connected.value = true
      if (eventStreamOpenedOnce || !snapshotLoaded.value) {
        void refresh().catch(() => undefined)
      }
      eventStreamOpenedOnce = true
    }
    eventSource.onerror = () => {
      connected.value = false
    }
    for (const eventType of TASK_EVENT_TYPES) {
      eventSource.addEventListener(
        eventType,
        (event) => receiveEvent(event as MessageEvent<string>, eventType),
      )
    }
  }

  async function initialize(): Promise<void> {
    try {
      await refresh()
    } catch {
      // EventSource reconnects independently and refreshes the durable
      // snapshot on open, so a transient startup failure is recoverable.
    }
    connect()
  }

  function disconnect(): void {
    if (refreshTimer) clearTimeout(refreshTimer)
    refreshTimer = null
    if (projectionTimer) clearTimeout(projectionTimer)
    projectionTimer = null
    if (detailProjectionTimer) clearTimeout(detailProjectionTimer)
    detailProjectionTimer = null
    detailProjectionDirty = false
    pendingProjectionJobIds.clear()
    eventRefreshDirty = false
    eventSource?.close()
    eventSource = null
    eventStreamOpenedOnce = false
    snapshotLoaded.value = false
    connected.value = false
    clearDetail()
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
          if (!HISTORY_JOB_STATUSES.has(job.status)) return
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
    if (refreshPromise) await refreshPromise.catch(() => undefined)
    await refresh().catch(() => undefined)
    return result
  }

  function hasActiveTranslation(
    chapterId: string,
    summary: Partial<Record<V2JobStatus, number>> = {},
  ): boolean {
    const matchesChapter = (job: V2Job) => (
      job.chapterId === chapterId
      && job.kind === 'translation'
      && NONTERMINAL_JOB_STATUSES.has(job.status)
    )
    if (queue.value.some(matchesChapter) || history.value.some(matchesChapter)) {
      return true
    }
    if (snapshotLoaded.value) return false
    return [...NONTERMINAL_JOB_STATUSES]
      .some(status => (summary[status] || 0) > 0)
  }

  async function loadDetail(jobId: string): Promise<void> {
    const requestVersion = ++detailRequestVersion
    selectedDetailJobId.value = jobId
    selectedDetail.value = null
    detailLoading.value = true
    olderEventsExhausted.value = false
    try {
      const detail = await jobsApi.get(jobId)
      if (
        requestVersion === detailRequestVersion
        && selectedDetailJobId.value === jobId
      ) {
        selectedDetail.value = detail
      }
    } catch (error) {
      if (requestVersion === detailRequestVersion) {
        selectedDetailJobId.value = null
        selectedDetail.value = null
      }
      throw error
    } finally {
      if (requestVersion === detailRequestVersion) {
        detailLoading.value = false
      }
    }
  }

  function clearDetail(): void {
    detailRequestVersion += 1
    selectedDetailJobId.value = null
    selectedDetail.value = null
    detailLoading.value = false
    olderEventsLoading.value = false
    olderEventsExhausted.value = false
  }

  async function loadOlderEvents(): Promise<void> {
    const detail = selectedDetail.value
    const firstEvent = detail?.recentEvents[0]
    if (!detail || !firstEvent || olderEventsExhausted.value) return
    const requestVersion = detailRequestVersion
    olderEventsLoading.value = true
    try {
      const page = await jobsApi.events(detail.jobId, { before: firstEvent.eventId, limit: 50 })
      if (
        requestVersion !== detailRequestVersion
        || selectedDetailJobId.value !== detail.jobId
        || selectedDetail.value?.jobId !== detail.jobId
      ) return
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
      if (requestVersion === detailRequestVersion) {
        olderEventsLoading.value = false
      }
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
    snapshotLoaded,
    connected,
    workerOnline,
    latestEvent,
    selectedDetail,
    selectedDetailJobId,
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
    hasActiveTranslation,
    waitForJob,
    refresh,
    open: (target?: TaskCenterFocus) => {
      focusTarget.value = target || null
      drawerOpen.value = true
      void refresh().catch(() => undefined)
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
    clearDetail,
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
