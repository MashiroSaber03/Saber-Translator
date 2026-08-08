<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import TaskBatchAnalysisModal from '@/components/task-center/TaskBatchAnalysisModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'
import type { V2Job } from '@/api/v2/jobs'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import {
  batchProgressCounts,
  batchStatusCounts,
  currentStepLabel,
  describeJobTarget,
  groupJobsByBatch,
  poolProgress,
  progressCounts,
  progressPercent,
  type JobBatchProjection,
} from '@/stores/taskCenterProjection'
import { jobsApi } from '@/api/v2/jobs'
import { triggerUrlDownload } from '@/utils/browserDownload'
import { showToast } from '@/utils/toast'
import { releaseWorkerModelCache } from '@/api/v2/system'
import type { V2AcceptedJob } from '@/api/v2/insight'
import { confirmProductAction } from '@/composables/useProductConfirm'
import {
  eventTypeLabel,
  formatByteSize,
  formatTaskDuration,
  jobKindLabel,
  resourceRoleLabel,
  stepKindLabel,
} from '@/utils/taskDisplay'

const store = useTaskCenterStore()
const tab = ref<'queue' | 'history'>('queue')
const expanded = ref(new Set<string>())
const downloading = ref(new Set<string>())
const analysisModalOpen = ref(false)
const releasingModels = ref(false)

interface TaskGroupView extends JobBatchProjection {
  zone: 'current' | 'waiting' | 'history'
  zoneStart: boolean
}

const groups = computed<TaskGroupView[]>(() => {
  if (tab.value === 'history') {
    return store.historyBatches.map((group, index) => ({
      ...group,
      key: `history:${group.key}`,
      zone: 'history',
      zoneStart: index === 0,
    }))
  }
  const current = groupJobsByBatch(store.currentJobs).map((group, index) => ({
    ...group,
    key: `current:${group.key}`,
    zone: 'current' as const,
    zoneStart: index === 0,
  }))
  const waiting = store.waitingBatches.map((group, index) => ({
    ...group,
    key: `waiting:${group.key}`,
    zone: 'waiting' as const,
    zoneStart: index === 0,
  }))
  return [...current, ...waiting]
})
const availableKinds = computed(() => (
  [...new Set([...store.queue, ...store.history].map(job => job.kind))].sort()
))
const statusOptions = computed<UiSelectOption[]>(() => [
  { label: '全部', value: '' },
  ...Object.entries(statusLabels).map(([value, label]) => ({ label, value })),
])
const kindOptions = computed<UiSelectOption[]>(() => [
  { label: '全部', value: '' },
  ...availableKinds.value.map(kind => ({ label: jobKindLabel(kind), value: kind })),
])
const bookOptions = computed<UiSelectOption[]>(() => {
  const labels = new Map<string, string>()
  for (const job of [...store.queue, ...store.history]) {
    if (!job.bookId) continue
    const target = job.target as Record<string, unknown>
    labels.set(
      job.bookId,
      typeof target.book === 'string' && target.book ? target.book : job.bookId,
    )
  }
  return [
    { label: '全部', value: '' },
    ...[...labels].map(([value, label]) => ({ label, value })),
  ]
})

watch(
  [
    () => store.drawerOpen,
    () => store.focusTarget,
    () => store.queue.length,
    () => store.history.length,
  ],
  async () => {
    const target = store.focusTarget
    if (!store.drawerOpen || !target) return
    const allJobs = [...store.queue, ...store.history]
    const focusedJob = target.jobId
      ? allJobs.find(job => job.jobId === target.jobId)
      : allJobs.find(job => (
          (!target.batchId || job.batchId === target.batchId)
          && (!target.chapterId || job.chapterId === target.chapterId)
          && (!target.bookId || job.bookId === target.bookId)
        ))
    const batchId = target.batchId || focusedJob?.batchId || undefined
    if (
      focusedJob
      && store.history.some(job => job.jobId === focusedJob.jobId)
    ) {
      tab.value = 'history'
    } else {
      tab.value = 'queue'
    }
    await nextTick()
    const group = groups.value.find(item => (
      item.jobs.some(job => job.jobId === focusedJob?.jobId)
    )) || groups.value.find(item => batchId && item.batchId === batchId)
    if (group) {
      expanded.value = new Set([...expanded.value, group.key])
    }
    await nextTick()
    const selector = focusedJob
      ? `[data-task-job-id="${focusedJob.jobId}"]`
      : batchId
        ? `[data-task-batch-id="${batchId}"]`
        : ''
    const element = selector
      ? document.querySelector<HTMLElement>(selector)
      : null
    element?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    if (element) store.focusTarget = null
  },
)

const statusLabels: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  pausing: '正在暂停',
  paused: '已暂停',
  cancelling: '正在取消',
  cancelled: '已取消',
  completed: '已完成',
  completed_with_errors: '部分失败',
  failed: '失败',
  interrupted: '已中断',
}

function toggle(key: string) {
  const next = new Set(expanded.value)
  if (next.has(key)) {
    next.delete(key)
  } else {
    next.add(key)
  }
  expanded.value = next
}

function canCancel(job: V2Job) {
  return ['queued', 'running', 'pausing', 'paused', 'interrupted'].includes(job.status)
}

function hasBatchContinuations(jobs: V2Job[]) {
  return jobs.some(job => ['paused', 'interrupted'].includes(job.status))
}

async function runAction(action: () => Promise<unknown>, success: string) {
  try {
    await action()
    showToast(success, 'success')
  } catch (error) {
    showToast(error instanceof Error ? error.message : '任务命令执行失败', 'error')
  }
}

async function toggleDetail(job: V2Job) {
  if (store.selectedDetail?.jobId === job.jobId) {
    store.selectedDetail = null
    return
  }
  try {
    await store.loadDetail(job.jobId)
  } catch (error) {
    showToast(error instanceof Error ? error.message : '读取任务详情失败', 'error')
  }
}

function retryJob(job: V2Job, strategy: 'current' | 'original') {
  const command = job.status === 'completed_with_errors'
    ? () => store.retryFailed(job.jobId, strategy)
    : () => store.retry(job.jobId, strategy)
  return runAction(command, strategy === 'current' ? '已按当前设置创建重试任务' : '已沿用原快照创建重试任务')
}

function formatPayload(value: unknown): string {
  if (value === null || value === undefined) return '无'
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2)
}

function setStatusFilter(value: UiSelectValue) {
  store.statusFilter = String(value) as typeof store.statusFilter
  void store.refresh()
}

function setKindFilter(value: UiSelectValue) {
  store.kindFilter = String(value) as typeof store.kindFilter
  void store.refresh()
}

function setBookFilter(value: UiSelectValue) {
  store.bookFilter = String(value)
  void store.refresh()
}

function queuePosition(job: V2Job): number | null {
  if (job.status !== 'queued') return null
  const index = store.queue.findIndex(item => item.jobId === job.jobId)
  if (index < 0) return null
  return store.queue
    .slice(0, index + 1)
    .filter(item => item.status === 'queued')
    .length
}

function blockedReasonLabel(job: V2Job): string {
  switch (job.blockedReason) {
    case 'draining_immediate_writes':
      return '正在排空即时写入，新编辑已暂停'
    case 'retained_chapter_lock':
      return '等待重新领取，章节锁已保留'
    case 'blocked_by_job':
      return '等待其他中断任务释放章节锁'
    default:
      return job.blockedReason ? `等待：${job.blockedReason}` : ''
  }
}

function batchProgressLabel(jobs: V2Job[]): string {
  const counts = batchProgressCounts(jobs)
  return `总进度 ${counts.completed} / ${counts.total}`
}

function batchStatusLabel(jobs: V2Job[]): string {
  return batchStatusCounts(jobs)
    .map(([status, count]) => `${statusLabels[status] || status} ${count}`)
    .join(' · ')
}

function formatTimestamp(value?: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString('zh-CN', { hour12: false })
}

function canDownloadArtifact(job: V2Job): boolean {
  if (job.status !== 'completed') return false
  if (job.kind === 'export' || job.kind === 'insight_export') return true
  if (job.kind !== 'continuation') return false
  return (job.target as Record<string, unknown>).continuationAction === 'export'
}

async function cancelJob(job: V2Job) {
  if (['running', 'pausing', 'paused'].includes(job.status)) {
    const confirmed = await confirmProductAction({
      title: '取消当前任务',
      message: '当前任务将在下一个安全点停止。已经完成并持久化的步骤不会丢失，确定取消吗？',
      confirmText: '确认取消',
      tone: 'danger',
    })
    if (!confirmed) return
  }
  await runAction(
    () => store.cancel(job.jobId),
    job.status === 'queued' ? '排队任务已取消' : '已提交取消请求',
  )
}

async function downloadArtifact(job: V2Job) {
  const pending = new Set(downloading.value)
  pending.add(job.jobId)
  downloading.value = pending
  try {
    const detail = await jobsApi.get(job.jobId)
    const artifact = detail.artifacts[0]
    if (!artifact) {
      showToast('该任务没有可下载产物', 'warning')
      return
    }
    const separator = artifact.url.includes('?') ? '&' : '?'
    triggerUrlDownload(
      `${artifact.url}${separator}download=1&filename=${encodeURIComponent(`${job.kind}-${job.jobId}`)}`,
    )
  } catch (error) {
    showToast(
      `读取任务产物失败：${error instanceof Error ? error.message : '未知错误'}`,
      'error',
    )
  } finally {
    const next = new Set(downloading.value)
    next.delete(job.jobId)
    downloading.value = next
  }
}

async function releaseModels() {
  if (releasingModels.value) return
  releasingModels.value = true
  try {
    await releaseWorkerModelCache()
    showToast('已提交显存释放命令，Worker 将在安全点立即执行', 'success')
  } catch (error) {
    const status = (
      error
      && typeof error === 'object'
      && 'status' in error
    ) ? Number(error.status) : 0
    showToast(
      status === 409
        ? '本地模型正在推理，请等待当前模型步骤完成后再试'
        : error instanceof Error
          ? error.message
          : '释放显存失败',
      status === 409 ? 'warning' : 'error',
    )
  } finally {
    releasingModels.value = false
  }
}

async function analysisCreated(result: V2AcceptedJob) {
  await store.refresh()
  store.open({
    batchId: result.batchId,
    jobId: result.jobIds[0],
  })
  showToast('批量分析任务已加入后端队列', 'success')
}
</script>

<template>
  <Teleport to="body">
    <OverlayLayer
      v-if="store.drawerOpen"
      class="task-center"
      level="overlay"
      role="dialog"
      aria-modal="true"
      aria-label="任务中心"
      @backdrop="store.close"
    >
      <aside class="task-center__panel">
        <header class="task-center__header">
          <div>
            <h2>任务中心</h2>
            <p>后端持续执行 · 关闭页面不会停止任务</p>
          </div>
          <div class="task-center__header-actions">
            <UiButton
              size="xs"
              variant="secondary"
              @click="analysisModalOpen = true"
            >
              新建批量分析
            </UiButton>
            <UiButton
              size="xs"
              variant="ghost"
              :disabled="releasingModels"
              @click="releaseModels"
            >
              {{ releasingModels ? '提交中…' : '释放显存' }}
            </UiButton>
            <UiButton size="sm" variant="ghost" class="task-center__close" @click="store.close">
              关闭
            </UiButton>
          </div>
        </header>

        <div v-if="!store.connected" class="task-center__offline">
          任务事件连接正在重连，当前列表来自持久化快照。
        </div>
        <div v-if="!store.workerOnline" class="task-center__offline">
          Worker 离线，排队任务会在 Worker 恢复后自动继续。
        </div>

        <nav class="task-center__tabs" aria-label="任务分区">
          <UiButton
            size="sm"
            variant="tab"
            class="task-center__tab"
            :class="{ 'task-center__tab--active': tab === 'queue' }"
            @click="tab = 'queue'"
          >
            队列 {{ store.queue.length }}
          </UiButton>
          <UiButton
            size="sm"
            variant="tab"
            class="task-center__tab"
            :class="{ 'task-center__tab--active': tab === 'history' }"
            @click="tab = 'history'"
          >
            历史 {{ store.historyBatches.length }}
          </UiButton>
          <span class="task-center__spacer" />
          <UiButton
            v-if="tab === 'queue' && store.queuedCount"
            size="xs"
            variant="ghost"
            tone="danger"
            @click="store.cancelQueued"
          >
            取消全部排队
          </UiButton>
          <UiButton
            v-if="tab === 'history' && store.history.length"
            size="xs"
            variant="ghost"
            @click="store.clearHistory"
          >
            清空历史
          </UiButton>
        </nav>

        <div v-if="tab === 'history'" class="task-center__filters">
          <label>
            状态
            <UiSelect
              size="sm"
              :model-value="store.statusFilter"
              :options="statusOptions"
              @change="setStatusFilter"
            />
          </label>
          <label>
            类型
            <UiSelect
              size="sm"
              :model-value="store.kindFilter"
              :options="kindOptions"
              @change="setKindFilter"
            />
          </label>
          <label>
            书籍
            <UiSelect
              size="sm"
              :model-value="store.bookFilter"
              :options="bookOptions"
              @change="setBookFilter"
            />
          </label>
        </div>

        <main class="task-center__content">
          <p v-if="store.loading && !groups.length" class="task-center__empty">正在读取后端任务…</p>
          <template v-else>
            <div
              v-if="tab === 'queue' && !store.currentJobs.length"
              class="task-zone task-zone--empty"
            >
              <h3>当前任务</h3>
              <p>暂无正在执行的任务</p>
            </div>

            <div
              v-if="tab === 'queue' && store.currentJobs.some(job => job.status === 'paused')"
              class="task-center__paused"
            >
              队列已因暂停阻塞
            </div>

            <p
              v-if="tab === 'history' && !groups.length"
              class="task-center__empty"
            >
              没有符合筛选条件的历史任务
            </p>

            <template v-for="group in groups" :key="group.key">
              <h3 v-if="group.zoneStart" class="task-zone__heading">
                {{ group.zone === 'current' ? '当前任务' : group.zone === 'waiting' ? '等待队列' : '历史' }}
              </h3>
              <section
                class="task-batch"
                :class="{ 'task-batch--standalone': group.jobs.length === 1 }"
                :data-task-batch-id="group.batchId || undefined"
              >
                <UiButton
                  v-if="group.jobs.length > 1"
                  variant="card-action"
                  block
                  class="task-batch__header"
                  @click="toggle(group.key)"
                >
                  <span>
                    <strong>{{ group.displayName }}</strong>
                    <small>{{ group.jobs.length }} 个任务 · {{ batchProgressLabel(group.jobs) }}</small>
                    <small>{{ batchStatusLabel(group.jobs) }}</small>
                  </span>
                  <span>{{ expanded.has(group.key) ? '收起' : '展开' }}</span>
                </UiButton>
                <div
                  v-if="group.zone === 'waiting' && group.batchId && group.jobs.length > 1"
                  class="task-batch__actions"
                >
                  <UiButton
                    size="xs"
                    variant="ghost"
                    @click="runAction(() => store.prioritizeBatch(group.batchId!), '批次已移到普通排队任务前方')"
                  >
                    整批置顶
                  </UiButton>
                  <UiButton
                    v-if="hasBatchContinuations(group.jobs)"
                    size="xs"
                    variant="ghost"
                    @click="runAction(() => store.continueBatch(group.batchId!), '批次中的暂停/中断任务已继续')"
                  >
                    全部继续
                  </UiButton>
                  <UiButton
                    size="xs"
                    variant="ghost"
                    tone="danger"
                    @click="runAction(() => store.cancelBatch(group.batchId!), '批次中仍在排队的任务已取消')"
                  >
                    整批取消
                  </UiButton>
                </div>

                <div
                  v-if="expanded.has(group.key) || group.jobs.length === 1"
                  class="task-batch__jobs"
                >
                  <article
                    v-for="job in group.jobs"
                    :key="job.jobId"
                    class="task-job"
                    :data-task-job-id="job.jobId"
                  >
                    <div class="task-job__top">
                      <div>
                        <strong>{{ describeJobTarget(job) }}</strong>
                        <span>
                          <span :title="job.kind">{{ jobKindLabel(job.kind) }}</span>
                          <template v-if="queuePosition(job)"> · 队列第 {{ queuePosition(job) }} 位</template>
                        </span>
                      </div>
                      <span class="task-job__status" :data-status="job.status">
                        {{ statusLabels[job.status] || job.status }}
                      </span>
                    </div>
                    <div class="task-job__progress">
                      <span :style="{ width: `${progressPercent(job)}%` }" />
                    </div>
                    <div class="task-job__progress-meta">
                      <span>
                        进度 {{ progressCounts(job).completed }} / {{ progressCounts(job).total }}
                      </span>
                      <span v-if="currentStepLabel(job)">当前：{{ currentStepLabel(job) }}</span>
                    </div>
                    <div v-if="poolProgress(job).length" class="task-job__pools">
                      <div v-for="pool in poolProgress(job)" :key="pool.kind">
                        <strong :title="pool.kind">{{ stepKindLabel(pool.kind) }}</strong>
                        <span>
                          等待 {{ pool.waiting }} · 处理中 {{ pool.processing }} · 完成 {{ pool.completed }}
                          <template v-if="pool.lockWaiting"> · 等待深度学习锁</template>
                        </span>
                      </div>
                    </div>
                    <p v-if="job.blockedReason" class="task-job__hint">
                      {{ blockedReasonLabel(job) }}
                    </p>
                    <div class="task-job__actions">
                      <UiButton
                        v-if="canDownloadArtifact(job)"
                        size="xs"
                        variant="secondary"
                        class="task-job__action"
                        :disabled="downloading.has(job.jobId)"
                        @click="downloadArtifact(job)"
                      >
                        {{ downloading.has(job.jobId) ? '读取中…' : '下载产物' }}
                      </UiButton>
                      <UiButton
                        v-if="job.status === 'running'"
                        size="xs"
                        variant="secondary"
                        @click="runAction(() => store.pause(job.jobId), '已提交暂停请求')"
                      >
                        暂停
                      </UiButton>
                      <UiButton
                        v-if="job.status === 'paused'"
                        size="xs"
                        variant="secondary"
                        @click="runAction(() => store.resume(job.jobId), '任务已继续排队')"
                      >
                        继续
                      </UiButton>
                      <UiButton
                        v-if="job.status === 'interrupted'"
                        size="xs"
                        variant="secondary"
                        @click="runAction(() => store.continueJob(job.jobId), '任务已从检查点继续')"
                      >
                        从检查点继续
                      </UiButton>
                      <UiButton
                        v-if="job.status === 'queued' && !job.blockedReason"
                        size="xs"
                        variant="ghost"
                        @click="runAction(() => store.prioritizeQueued(job.jobId), '任务已置顶')"
                      >
                        置顶
                      </UiButton>
                      <UiButton v-if="job.status === 'queued' && !job.blockedReason" size="xs" variant="ghost" @click="store.moveQueued(job.jobId, -1)">上移</UiButton>
                      <UiButton v-if="job.status === 'queued' && !job.blockedReason" size="xs" variant="ghost" @click="store.moveQueued(job.jobId, 1)">下移</UiButton>
                      <UiButton
                        v-if="job.status === 'failed' || job.status === 'completed_with_errors'"
                        size="xs"
                        variant="secondary"
                        @click="retryJob(job, job.kind === 'style_apply' ? 'original' : 'current')"
                      >
                        {{ job.status === 'completed_with_errors' ? '重试失败项' : '重试任务' }}
                      </UiButton>
                      <UiButton
                        v-if="(job.status === 'failed' || job.status === 'completed_with_errors') && job.kind !== 'style_apply'"
                        size="xs"
                        variant="ghost"
                        @click="retryJob(job, 'original')"
                      >
                        沿用原快照
                      </UiButton>
                      <UiButton size="xs" variant="ghost" @click="toggleDetail(job)">
                        {{ store.selectedDetail?.jobId === job.jobId ? '收起详情' : '详情' }}
                      </UiButton>
                      <UiButton v-if="canCancel(job)" size="xs" variant="ghost" tone="danger" @click="cancelJob(job)">取消</UiButton>
                    </div>
                    <div
                      v-if="store.selectedDetail?.jobId === job.jobId"
                      class="task-job__detail"
                    >
                      <p v-if="store.detailLoading">正在读取详情…</p>
                      <template v-else>
                        <dl>
                          <div><dt>状态</dt><dd>{{ statusLabels[store.selectedDetail.status] || store.selectedDetail.status }}</dd></div>
                          <div><dt>队列位置</dt><dd>{{ queuePosition(store.selectedDetail) ? `第 ${queuePosition(store.selectedDetail)} 位` : '—' }}</dd></div>
                          <div><dt>创建时间</dt><dd>{{ formatTimestamp(store.selectedDetail.createdAt) }}</dd></div>
                          <div><dt>开始时间</dt><dd>{{ formatTimestamp(store.selectedDetail.startedAt) }}</dd></div>
                          <div><dt>结束时间</dt><dd>{{ formatTimestamp(store.selectedDetail.finishedAt) }}</dd></div>
                          <div><dt>耗时</dt><dd>{{ formatTaskDuration(store.selectedDetail.durationMs) }}</dd></div>
                          <div><dt>完成</dt><dd>{{ store.selectedDetail.counts.completed }} / {{ store.selectedDetail.counts.total }}</dd></div>
                          <div><dt>失败</dt><dd>{{ store.selectedDetail.counts.failed }}</dd></div>
                          <div><dt>跳过</dt><dd>{{ store.selectedDetail.counts.skipped }}</dd></div>
                        </dl>
                        <section v-if="store.selectedDetail.error">
                          <strong>任务错误</strong>
                          <pre>{{ formatPayload(store.selectedDetail.error) }}</pre>
                        </section>
                        <section v-if="store.selectedDetail.failedItems.length">
                          <strong>失败项</strong>
                          <ul>
                            <li v-for="item in store.selectedDetail.failedItems" :key="item.itemId">
                              #{{ item.ordinal }} · {{ item.stepKind ? stepKindLabel(item.stepKind) : '未知步骤' }} · {{ formatPayload(item.error) }}
                            </li>
                          </ul>
                        </section>
                        <section>
                          <strong>配置摘要（已脱敏）</strong>
                          <pre>{{ formatPayload(store.selectedDetail.configSummary) }}</pre>
                        </section>
                        <section v-if="store.selectedDetail.resources.length">
                          <strong>步骤资源</strong>
                          <ul>
                            <li v-for="resource in store.selectedDetail.resources" :key="`${resource.stepId}:${resource.role}`">
                              <a :href="resource.url" target="_blank" rel="noopener" :title="resource.role">{{ resourceRoleLabel(resource.role) }}</a>
                              · {{ resource.mimeType }} · {{ formatByteSize(resource.byteSize) }}
                            </li>
                          </ul>
                        </section>
                        <section v-if="store.selectedDetail.recentEvents.length">
                          <strong>最近事件</strong>
                          <UiButton
                            v-if="!store.olderEventsExhausted"
                            size="xs"
                            variant="ghost"
                            :disabled="store.olderEventsLoading"
                            @click="store.loadOlderEvents"
                          >
                            {{ store.olderEventsLoading ? '读取中…' : '加载更早事件' }}
                          </UiButton>
                          <ul>
                            <li v-for="event in [...store.selectedDetail.recentEvents].reverse()" :key="event.eventId">
                              <span :title="event.type">#{{ event.eventId }} · {{ eventTypeLabel(event.type) }}</span>
                            </li>
                          </ul>
                        </section>
                      </template>
                    </div>
                  </article>
                </div>
              </section>
            </template>

            <div
              v-if="tab === 'queue' && !store.waitingBatches.length"
              class="task-zone task-zone--empty"
            >
              <h3>等待队列</h3>
              <p>暂无排队任务</p>
            </div>
          </template>
        </main>
      </aside>
    </OverlayLayer>
    <TaskBatchAnalysisModal
      v-model="analysisModalOpen"
      @created="analysisCreated"
    />
  </Teleport>
</template>

<style scoped>
.task-center {
  background: var(--color-overlay-scrim-subtle);
}

.task-center__panel {
  position: absolute;
  top: 0;
  right: 0;
  display: flex;
  flex-direction: column;
  width: min(520px, 100vw);
  height: 100%;
  color: var(--color-text-default);
  background: var(--color-surface-base);
  border-left: 1px solid var(--color-border-default);
  box-shadow: 0 8px 16px var(--shadow-medium);
}

.task-center__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 22px 24px 16px;
  border-bottom: 1px solid var(--color-border-default);
}

.task-center__header h2,
.task-center__header p {
  margin: 0;
}

.task-center__header p {
  margin-top: 4px;
  color: var(--color-text-muted);
  font-size: 13px;
}

.task-center__header-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
}

.task-center__close,
.task-center__tab,
.task-job__action {
  padding: 7px 10px;
  color: var(--color-text-default);
  font: inherit;
  font-size: 12px;
  background: var(--color-surface-muted);
  border: 1px solid var(--color-border-default);
  border-radius: 8px;
  cursor: pointer;
}

.task-center__offline {
  padding: 9px 24px;
  color: var(--color-status-warning);
  font-size: 12px;
  background: var(--color-status-warning-surface);
}

.task-center__tabs {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 12px 24px;
  border-bottom: 1px solid var(--color-border-default);
}

.task-center__filters {
  display: flex;
  gap: 12px;
  padding: 10px 24px;
  border-bottom: 1px solid var(--color-border-default);
}

.task-center__filters label {
  display: grid;
  flex: 1;
  gap: 4px;
  color: var(--color-text-muted);
  font-size: 11px;
}

.task-center__tab--active {
  color: var(--color-action-primary);
  border-color: var(--color-action-primary);
}

.task-center__spacer {
  flex: 1;
}

.task-center__content {
  flex: 1;
  padding: 16px 18px 32px;
  overflow: auto;
}

.task-center__empty {
  padding: 48px 12px;
  color: var(--color-text-muted);
  text-align: center;
}

.task-zone__heading {
  margin: 4px 2px 10px;
  font-size: 15px;
}

.task-zone__heading:not(:first-child) {
  margin-top: 22px;
}

.task-zone--empty {
  margin: 4px 2px 18px;
}

.task-zone--empty h3,
.task-zone--empty p {
  margin: 0;
}

.task-zone--empty h3 {
  font-size: 15px;
}

.task-zone--empty p {
  margin-top: 8px;
  color: var(--color-text-muted);
  font-size: 12px;
}

.task-center__paused {
  margin: 0 0 12px;
  padding: 10px 12px;
  color: var(--color-status-warning);
  font-size: 12px;
  background: var(--color-status-warning-surface);
  border-radius: 8px;
}

.task-batch {
  margin-bottom: 12px;
  overflow: hidden;
  border: 1px solid var(--color-border-default);
  border-radius: 12px;
}

.task-batch__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 12px 14px;
  color: var(--color-text-default);
  text-align: left;
  background: var(--color-surface-muted);
  border: 0;
  cursor: pointer;
}

.task-batch__header span:first-child {
  display: grid;
  gap: 2px;
}

.task-batch__header small {
  color: var(--color-text-muted);
}

.task-batch__jobs {
  display: grid;
}

.task-batch__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
  padding: 8px 12px;
  border-top: 1px solid var(--color-border-default);
}

.task-job {
  padding: 14px;
  border-top: 1px solid var(--color-border-default);
}

.task-batch--standalone .task-job {
  border-top: 0;
}

.task-job__top,
.task-job__actions {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
}

.task-job__top > div {
  display: grid;
  gap: 2px;
}

.task-job__top span,
.task-job__hint {
  color: var(--color-text-muted);
  font-size: 12px;
}

.task-job__status[data-status='running'],
.task-job__status[data-status='completed'] {
  color: var(--color-status-success);
}

.task-job__status[data-status='failed'],
.task-job__status[data-status='interrupted'] {
  color: var(--color-status-error);
}

.task-job__progress {
  height: 5px;
  margin: 12px 0 6px;
  overflow: hidden;
  background: var(--color-surface-muted);
  border-radius: 999px;
}

.task-job__progress span {
  display: block;
  height: 100%;
  background: var(--color-action-primary);
}

.task-job__progress-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  justify-content: space-between;
  margin-bottom: 10px;
  color: var(--color-text-muted);
  font-size: 11px;
}

.task-job__pools {
  display: grid;
  gap: 5px;
  margin-bottom: 10px;
  padding: 8px 10px;
  font-size: 11px;
  background: var(--color-surface-muted);
  border-radius: 7px;
}

.task-job__pools div {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 8px;
  justify-content: space-between;
}

.task-job__pools span {
  color: var(--color-text-muted);
}

.task-job__hint {
  margin: 0 0 10px;
}

.task-job__actions {
  flex-wrap: wrap;
  justify-content: flex-end;
}

.task-job__detail {
  display: grid;
  gap: 12px;
  margin-top: 12px;
  padding: 12px;
  font-size: 12px;
  background: var(--color-surface-muted);
  border-radius: 8px;
}

.task-job__detail dl {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.task-job__detail dl div {
  display: flex;
  justify-content: space-between;
}

.task-job__detail dt,
.task-job__detail dd,
.task-job__detail p,
.task-job__detail ul {
  margin: 0;
}

.task-job__detail section {
  display: grid;
  gap: 6px;
}

.task-job__detail pre {
  max-height: 160px;
  margin: 0;
  padding: 8px;
  overflow: auto;
  white-space: pre-wrap;
  background: var(--color-surface-base);
  border-radius: 6px;
}

.task-job__detail ul {
  display: grid;
  gap: 4px;
  padding-left: 18px;
}
</style>
