import { computed, ref, watch, type WatchStopHandle } from 'vue'
import { defineStore } from 'pinia'
import {
  CURRENT_JOB_STATUSES,
  HISTORY_JOB_STATUSES,
  NONTERMINAL_JOB_STATUSES,
  jobsApi,
  type JobListResponse,
  type JobRetryAccepted,
  type V2Job,
  type V2JobDetail,
  type V2JobEvent,
  type V2JobStatus,
} from '@/api/v2/jobs'
import { groupJobsByBatch } from '@/stores/taskCenterProjection'
import { TASK_EVENT_TYPES } from '@/utils/taskDisplay'

type TaskCenterEventListener = (event: V2JobEvent) => void
type TaskWaitingReason = JobListResponse['waitingReason']

const QUEUE_STATUSES = new Set<V2JobStatus>([
  'queued',
  ...CURRENT_JOB_STATUSES,
])
const HISTORY_BATCH_LIMIT = 200
const RECONCILE_MS = 15_000
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
    || typeof record.createdAt !== 'string'
    || record.createdAt.length === 0
    || !Number.isFinite(Date.parse(record.createdAt))
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
  const queuePaused = ref(false)
  const drawerOpen = ref(false)
  const focusTarget = ref<TaskCenterFocus | null>(null)
  const loading = ref(false)
  const snapshotLoaded = ref(false)
  const connected = ref(false)
  const workerOnline = ref(false)
  const executorBusy = ref(false)
  const waitingReason = ref<TaskWaitingReason>(null)
  const lastEventId = ref(0)
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
  let reconcileTimer: ReturnType<typeof setTimeout> | null = null
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
  let lifecycleGeneration = 0
  let lifecycleActive = false
  const pendingProjectionJobIds = new Set<string>()
  const eventListeners = new Set<TaskCenterEventListener>()
  const lifecycleResetListeners = new Set<() => void>()

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

  function latestRetryableFailedJob(
    chapterId: string,
    kinds: readonly V2Job['kind'][],
  ): V2Job | null {
    const matching = [...queue.value, ...history.value]
      .filter(job => job.chapterId === chapterId && kinds.includes(job.kind))
    const retriedJobIds = new Set(
      matching.flatMap(job => (
        job.retryOfJobId && !['cancelled', 'failed'].includes(job.status)
          ? [job.retryOfJobId]
          : []
      )),
    )
    return matching
      .filter(job => (
        job.status === 'completed_with_errors' && !retriedJobIds.has(job.jobId)
      ))
      .reduce<V2Job | null>(
        (latest, job) => (
          !latest || job.createdAt > latest.createdAt ? job : latest
        ),
        null,
      )
  }

  function retryableFailedItemCount(
    chapterId: string | null | undefined,
    kinds: readonly V2Job['kind'][] = ['translation'],
  ): number {
    if (!chapterId) return 0
    return latestRetryableFailedJob(chapterId, kinds)?.progress.failedItems ?? 0
  }

  function refresh(): Promise<void> {
    if (refreshPromise) return refreshPromise
    const generation = lifecycleGeneration
    const startedProjectionVersion = projectionVersion
    loading.value = true
    const request = jobsApi.list('all').then((result) => {
      if (generation !== lifecycleGeneration) return
      if (projectionVersion !== startedProjectionVersion) {
        scheduleRefresh()
        return
      }
      queue.value = result.items.filter(job => QUEUE_STATUSES.has(job.status))
      history.value = result.items.filter(job => HISTORY_JOB_STATUSES.has(job.status))
      queuePaused.value = result.queuePaused
      workerOnline.value = result.workerOnline
      executorBusy.value = result.executorBusy
      waitingReason.value = result.waitingReason
      snapshotLoaded.value = true
      if (selectedDetail.value) {
        scheduleSelectedDetailRefresh(selectedDetail.value.jobId)
      }
      if (
        !eventSource &&
        lastEventId.value === 0 &&
        typeof result.eventCursor === 'number'
      ) {
        lastEventId.value = result.eventCursor
      }
    }).finally(() => {
      if (refreshPromise === request) {
        loading.value = false
        refreshPromise = null
      }
    })
    refreshPromise = request
    return request
  }

  function queueOrder(left: V2Job, right: V2Job): number {
    const leftRank = left.queueRank ?? Number.MAX_SAFE_INTEGER
    const rightRank = right.queueRank ?? Number.MAX_SAFE_INTEGER
    if (leftRank !== rightRank) return leftRank - rightRank
    return left.createdAt.localeCompare(right.createdAt)
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
    const generation = lifecycleGeneration
    const request = jobsApi.get(jobId).then((detail) => {
      if (generation !== lifecycleGeneration) return
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
      if (detailProjectionPromise === request) {
        detailProjectionPromise = null
        const selectedJobId = selectedDetail.value?.jobId
        if (detailProjectionDirty && selectedJobId) {
          scheduleSelectedDetailRefresh(selectedJobId)
        }
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

  function stopReconciliation(): void {
    if (reconcileTimer) clearTimeout(reconcileTimer)
    reconcileTimer = null
  }

  function scheduleReconciliation(): void {
    const hasNonterminalJob = queue.value.some(job =>
      NONTERMINAL_JOB_STATUSES.has(job.status)
    )
    if (
      reconcileTimer
      || !eventSource
      || !eventStreamOpenedOnce
      || (!hasNonterminalJob && !drawerOpen.value)
    ) return
    reconcileTimer = setTimeout(() => {
      reconcileTimer = null
      if (!eventSource || !eventStreamOpenedOnce) return
      const activeJobIds = queue.value
        .filter(job => NONTERMINAL_JOB_STATUSES.has(job.status))
        .map(job => job.jobId)
      if (activeJobIds.length) {
        for (const jobId of activeJobIds) scheduleJobProjection(jobId)
      } else if (drawerOpen.value) {
        void refresh().catch(() => undefined)
      }
      scheduleReconciliation()
    }, RECONCILE_MS)
  }

  watch(
    [
      () => queue.value.map(job => `${job.jobId}:${job.status}`).join('|'),
      () => drawerOpen.value,
    ],
    () => scheduleReconciliation(),
  )

  async function flushJobProjections(): Promise<void> {
    if (projectionPromise || pendingProjectionJobIds.size === 0) return
    const generation = lifecycleGeneration
    const jobIds = [...pendingProjectionJobIds].slice(0, 200)
    for (const jobId of jobIds) pendingProjectionJobIds.delete(jobId)
    const request = jobsApi.snapshot(jobIds).then(result => {
      if (generation !== lifecycleGeneration) return
      const found = new Set(result.items.map(job => job.jobId))
      for (const job of result.items) applyJobProjection(job)
      for (const jobId of jobIds) {
        if (!found.has(jobId)) removeJobProjection(jobId)
      }
      queuePaused.value = result.queuePaused
      workerOnline.value = result.workerOnline
      executorBusy.value = result.executorBusy
      waitingReason.value = result.waitingReason
      projectionVersion += 1
    }).catch(() => {
      if (generation === lifecycleGeneration) scheduleRefresh()
    }).finally(() => {
      if (projectionPromise === request) {
        projectionPromise = null
        if (pendingProjectionJobIds.size > 0) {
          const nextJobId = pendingProjectionJobIds.values().next().value
          if (nextJobId) scheduleJobProjection(nextJobId)
        }
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
      const generation = lifecycleGeneration
      eventRefreshInFlight = true
      eventRefreshDirty = false
      try {
        await refresh()
      } catch {
        // A later SSE event, reconnect, drawer open, or explicit action retries
        // the durable snapshot. Background refresh failures must not surface as
        // unhandled promise rejections.
      } finally {
        if (generation === lifecycleGeneration) {
          eventRefreshInFlight = false
          if (eventRefreshDirty) scheduleRefresh()
        }
      }
    }, 250)
  }

  function receiveEvent(
    event: MessageEvent<string>,
    expectedType: string,
    generation: number,
  ): void {
    if (generation !== lifecycleGeneration || !lifecycleActive) return
    try {
      const parsed = parseJobEvent(JSON.parse(event.data))
      if (!parsed || parsed.type !== expectedType) {
        scheduleRefresh()
        return
      }
      if (parsed.eventId <= lastEventId.value) return
      lastEventId.value = parsed.eventId
      for (const listener of eventListeners) listener(parsed)
      scheduleJobProjection(parsed.jobId)
    } catch {
      scheduleRefresh()
    }
  }

  function connect(generation = lifecycleGeneration): void {
    if (eventSource || !lifecycleActive || generation !== lifecycleGeneration) return
    const source = new EventSource(`/api/v2/jobs/events?after=${lastEventId.value}`)
    eventSource = source
    source.onopen = () => {
      if (
        generation !== lifecycleGeneration
        || !lifecycleActive
        || eventSource !== source
      ) return
      connected.value = true
      if (eventStreamOpenedOnce || !snapshotLoaded.value) {
        void refresh().catch(() => undefined)
      }
      eventStreamOpenedOnce = true
      scheduleReconciliation()
    }
    source.onerror = () => {
      if (generation !== lifecycleGeneration || eventSource !== source) return
      connected.value = false
      scheduleRefresh()
    }
    for (const eventType of TASK_EVENT_TYPES) {
      source.addEventListener(
        eventType,
        (event) => receiveEvent(event as MessageEvent<string>, eventType, generation),
      )
    }
  }

  async function initialize(): Promise<void> {
    if (lifecycleActive) return
    lifecycleActive = true
    const generation = lifecycleGeneration
    try {
      await refresh()
    } catch {
      // EventSource reconnects independently and refreshes the durable
      // snapshot on open, so a transient startup failure is recoverable.
    }
    if (lifecycleActive && generation === lifecycleGeneration) connect(generation)
  }

  function disconnect(): void {
    lifecycleActive = false
    lifecycleGeneration += 1
    for (const listener of [...lifecycleResetListeners]) listener()
    stopReconciliation()
    if (refreshTimer) clearTimeout(refreshTimer)
    refreshTimer = null
    if (projectionTimer) clearTimeout(projectionTimer)
    projectionTimer = null
    if (detailProjectionTimer) clearTimeout(detailProjectionTimer)
    detailProjectionTimer = null
    detailProjectionDirty = false
    pendingProjectionJobIds.clear()
    eventRefreshDirty = false
    eventRefreshInFlight = false
    eventSource?.close()
    eventSource = null
    refreshPromise = null
    projectionPromise = null
    detailProjectionPromise = null
    eventStreamOpenedOnce = false
    projectionVersion = 0
    queue.value = []
    history.value = []
    queuePaused.value = false
    workerOnline.value = false
    executorBusy.value = false
    waitingReason.value = null
    lastEventId.value = 0
    loading.value = false
    snapshotLoaded.value = false
    connected.value = false
    drawerOpen.value = false
    focusTarget.value = null
    statusFilter.value = ''
    kindFilter.value = ''
    bookFilter.value = ''
    clearDetail()
  }

  function subscribeEvents(listener: TaskCenterEventListener): () => void {
    eventListeners.add(listener)
    return () => eventListeners.delete(listener)
  }

  async function waitForJob(jobId: string, options: WaitForJobOptions = {}): Promise<V2JobDetail> {
    const generation = lifecycleGeneration
    options.signal?.throwIfAborted()
    await refresh()
    options.signal?.throwIfAborted()
    if (generation !== lifecycleGeneration) {
      throw new Error('任务上下文已切换')
    }

    return new Promise<V2JobDetail>((resolve, reject) => {
      let settled = false
      let terminalDetailLoading = false
      let stop: WatchStopHandle | null = null

      const finish = (action: () => void): void => {
        if (settled) return
        settled = true
        stop?.()
        options.signal?.removeEventListener('abort', abort)
        lifecycleResetListeners.delete(resetLifecycle)
        action()
      }
      const abort = (): void => finish(() => reject(new DOMException('Aborted', 'AbortError')))
      const resetLifecycle = (): void => finish(() => reject(new Error('任务上下文已切换')))
      lifecycleResetListeners.add(resetLifecycle)

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
    const generation = lifecycleGeneration
    const result = await command()
    if (generation !== lifecycleGeneration) return result
    const commandResult = result && typeof result === 'object'
      ? result as { jobId?: unknown; jobIds?: unknown }
      : null
    const jobIds = Array.isArray(commandResult?.jobIds)
      ? commandResult.jobIds
      : null
    if (typeof commandResult?.jobId === 'string') {
      scheduleJobProjection(commandResult.jobId)
    } else if (
      jobIds
      && jobIds.length > 0
      && jobIds.every((jobId): jobId is string => (
        typeof jobId === 'string' && jobId.length > 0
      ))
    ) {
      for (const jobId of jobIds) scheduleJobProjection(jobId)
    } else {
      scheduleRefresh()
    }
    return result
  }

  function trackJob(jobId: string): void {
    if (jobId) scheduleJobProjection(jobId)
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
  ): Promise<(JobRetryAccepted & { failedItemCount: number }) | null> {
    await refresh()
    const source = latestRetryableFailedJob(chapterId, kinds)
    if (!source) return null
    return {
      ...await retryFailed(source.jobId, strategy),
      failedItemCount: source.progress.failedItems,
    }
  }

  async function moveQueued(jobId: string, delta: -1 | 1): Promise<void> {
    const sortable = queue.value.filter(
      job => job.status === 'queued' && job.blockedReason !== 'retained_chapter_lock'
    )
    const index = sortable.findIndex(job => job.jobId === jobId)
    const target = index + delta
    if (index < 0 || target < 0 || target >= sortable.length) return
    const ordered = sortable.map(job => job.jobId)
    ;[ordered[index], ordered[target]] = [ordered[target]!, ordered[index]!]
    await runCommand(() => jobsApi.reorder(ordered))
  }

  async function prioritizeQueued(jobId: string): Promise<void> {
    const sortable = queue.value.filter(
      job => job.status === 'queued' && job.blockedReason !== 'retained_chapter_lock'
    )
    const index = sortable.findIndex(job => job.jobId === jobId)
    if (index <= 0) return
    const ordered = sortable.map(job => job.jobId)
    ordered.splice(index, 1)
    ordered.unshift(jobId)
    await runCommand(() => jobsApi.reorder(ordered))
  }

  async function setQueuePaused(paused: boolean): Promise<void> {
    const generation = lifecycleGeneration
    const result = await runCommand(() => (
      paused ? jobsApi.pauseQueue() : jobsApi.resumeQueue()
    ))
    if (generation !== lifecycleGeneration) return
    queuePaused.value = result.queuePaused
  }

  return {
    queue,
    history,
    queuePaused,
    drawerOpen,
    focusTarget,
    loading,
    snapshotLoaded,
    connected,
    workerOnline,
    executorBusy,
    waitingReason,
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
    retryableFailedItemCount,
    initialize,
    disconnect,
    subscribeEvents,
    trackJob,
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
    pauseQueue: () => setQueuePaused(true),
    resumeQueue: () => setQueuePaused(false),
    cancelBatch: (batchId: string) => runCommand(() => jobsApi.cancelBatch(batchId)),
    prioritizeBatch: (batchId: string) =>
      runCommand(() => jobsApi.prioritizeBatch(batchId)),
    continueBatch: (batchId: string) => runCommand(() => jobsApi.continueBatch(batchId)),
    cancelQueued: () => runCommand(() => jobsApi.cancelQueued()),
    clearHistory: () => runCommand(() => jobsApi.clearHistory()),
  }
})
