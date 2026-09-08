<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import SelectControl from './SelectControl.vue'
import { jobKindLabel, stepKindLabel } from '../../../vue-frontend/src/utils/taskDisplay'
import type { components } from '../../../vue-frontend/src/api/generated/v2'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
type Schema = components['schemas']
const props = defineProps<{ api: PluginSettingsApi; active: boolean }>()
let removed = false
const active = computed(() => props.active)
const queue = ref<Schema['JobList'] | null>(null)
const filter = ref('all')
const details = ref<Record<string, Schema['JobDetail'] | null>>({})
const loadError = ref('')
const commandError = ref('')
const busy = ref(false)
let timer: ReturnType<typeof setTimeout> | undefined
const filters = [
  { value: 'all', label: '全部 Saber 任务' },
  { value: 'active', label: '未完成任务' },
]
const unfinished = ['queued', 'running', 'paused', 'interrupted']
const statusLabels: Record<Schema['JobStatus'], string> = {
  queued: '排队中',
  running: '运行中',
  paused: '已暂停',
  interrupted: '已中断',
  completed: '已完成',
  completed_with_errors: '部分失败',
  failed: '失败',
  cancelled: '已取消',
}
const jobs = computed(
  () =>
    queue.value?.items.filter(job => filter.value === 'all' || unfinished.includes(job.status)) ??
    []
)
const errorText = (error: unknown) => (error instanceof Error ? error.message : '请求失败')
const jobError = (error: Schema['JobDetail']['error']) =>
  typeof error === 'string' ? error : error?.message

async function loadDetail(jobId: string) {
  const detail = await props.api<Schema['JobDetail']>(`/jobs/${jobId}`)
  if (jobId in details.value) details.value[jobId] = detail
}
async function toggleDetail(jobId: string) {
  if (jobId in details.value) {
    delete details.value[jobId]
    return
  }
  details.value[jobId] = null
  try {
    await loadDetail(jobId)
  } catch (error) {
    delete details.value[jobId]
    commandError.value = errorText(error)
  }
}
async function refresh() {
  clearTimeout(timer)
  if (!active.value || busy.value) return
  busy.value = true
  try {
    queue.value = await props.api<Schema['JobList']>('/jobs?scope=all&limit=200')
    for (const id of Object.keys(details.value)) {
      if (!queue.value.items.some(job => job.jobId === id)) delete details.value[id]
    }
    await Promise.all(Object.keys(details.value).map(loadDetail))
    loadError.value = ''
  } catch (error) {
    loadError.value = errorText(error)
  } finally {
    busy.value = false
    if (!removed && active.value) timer = setTimeout(() => void refresh(), 3000)
  }
}
async function command(path: string, payload?: unknown) {
  if (busy.value) return
  busy.value = true
  commandError.value = ''
  clearTimeout(timer)
  try {
    await props.api(path, 'POST', payload)
  } catch (error) {
    commandError.value = errorText(error)
  } finally {
    busy.value = false
    await refresh()
  }
}
function jobCommand(jobId: string, action: string) {
  return command(
    `/jobs/${jobId}/${action}`,
    action.startsWith('retry') ? { strategy: 'current' } : undefined
  )
}

watch(
  active,
  value => {
    clearTimeout(timer)
    if (value) void refresh()
  },
  { immediate: true }
)

onBeforeUnmount(() => {
  removed = true
  clearTimeout(timer)
})
</script>
<template>
  <div class="view-heading">
    <div>
      <h2>任务中心</h2>
      <p>与本机翻译器同步</p>
    </div>
    <button class="icon-button" aria-label="刷新" :disabled="busy" @click="refresh">↻</button>
  </div>
  <div v-if="commandError || loadError" class="notice error" role="status">
    {{ commandError || loadError }}
  </div>
  <div class="queue-tools">
    <SelectControl v-model="filter" :options="filters" label="显示范围" /><button
      v-if="queue"
      class="button"
      :disabled="busy"
      @click="command(`/jobs/queue/${queue.queuePaused ? 'resume' : 'pause'}`)"
    >
      {{ queue.queuePaused ? '恢复全局队列' : '暂停全局队列' }}
    </button>
  </div>
  <div v-if="queue" class="connection-state">
    <span class="status-dot" :class="{ offline: !queue.workerOnline }" />{{
      queue.workerOnline ? '工作进程在线' : '工作进程未就绪，请启动 Saber。'
    }}<span v-if="busy" class="muted">· 同步中</span>
  </div>
  <div v-if="queue?.queuePaused" class="notice">全局队列已暂停，新任务等待执行。</div>
  <div v-if="!queue && !loadError" class="empty-state">
    <span class="empty-icon">◌</span>
    <h3>正在同步任务</h3>
    <p>连接本机 Saber…</p>
  </div>
  <div v-else-if="queue && !jobs.length" class="empty-state">
    <span class="empty-icon">✓</span>
    <h3>这里暂时很安静</h3>
    <p>暂无任务。在漫画页面中选择图片即可开始。</p>
  </div>
  <article v-for="job in jobs" :key="job.jobId" class="job-card">
    <div class="job-title">
      <h3>
        {{ job.target.chapter ?? job.target.book ?? job.batchDisplayName ?? '任务' }}
      </h3>
      <span
        class="badge"
        :class="{
          error: job.status === 'failed',
          success: job.status === 'completed',
        }"
        >{{ statusLabels[job.status] }}</span
      >
    </div>
    <p class="muted">
      {{ jobKindLabel(job.kind) }} · {{ statusLabels[job.status] }} · 成功
      {{ job.progress.completedItems }}/{{ job.progress.totalItems }} · 失败
      {{ job.progress.failedItems }}
    </p>
    <progress
      :max="job.progress.totalItems || 1"
      :value="
        job.progress.completedItems +
        job.progress.failedItems +
        job.progress.cancelledItems +
        job.progress.skippedItems
      "
      aria-label="任务进度"
    />
    <p v-if="job.progress.currentStep" class="muted">
      第 {{ job.progress.currentStep.itemOrdinal }} 张 ·
      {{ stepKindLabel(job.progress.currentStep.kind) }}
    </p>
    <p v-if="job.blockedReason" class="muted">等待同章节的其他任务释放占用</p>
    <div class="actions">
      <button
        v-if="job.status === 'running'"
        class="button"
        :disabled="busy"
        @click="jobCommand(job.jobId, 'pause')"
      >
        暂停
      </button>
      <button
        v-if="job.status === 'paused'"
        class="button"
        :disabled="busy"
        @click="jobCommand(job.jobId, 'resume')"
      >
        恢复
      </button>
      <button
        v-if="job.status === 'interrupted'"
        class="button"
        :disabled="busy"
        @click="jobCommand(job.jobId, 'continue')"
      >
        继续
      </button>
      <button
        v-if="unfinished.includes(job.status)"
        class="button subtle danger"
        :disabled="busy"
        @click="jobCommand(job.jobId, 'cancel')"
      >
        取消
      </button>
      <button
        v-if="job.status === 'completed_with_errors' || job.status === 'failed'"
        class="button"
        :disabled="busy"
        @click="jobCommand(job.jobId, job.status === 'failed' ? 'retry' : 'retry-failed')"
      >
        按当前设置重试{{ job.status === 'completed_with_errors' ? '失败页' : '' }}
      </button>
      <button
        class="text-button"
        :aria-expanded="job.jobId in details"
        @click="toggleDetail(job.jobId)"
      >
        {{ job.jobId in details ? '收起详情' : '查看详情' }}
      </button>
    </div>
    <div v-if="job.jobId in details" class="job-detail">
      <template v-if="details[job.jobId]"
        ><p class="muted">任务 {{ job.jobId.slice(0, 8) }}</p>
        <p v-if="details[job.jobId]!.error" class="error">
          {{ jobError(details[job.jobId]!.error) }}
        </p>
        <p v-for="item in details[job.jobId]!.failedItems" :key="item.ordinal" class="error">
          第 {{ item.ordinal }} 张：{{ jobError(item.error) }}
        </p></template
      >
      <p v-else class="muted">正在读取详情…</p>
    </div>
  </article>
  <p class="footnote">退出漫画页面后，该页面的临时任务会取消并清理。</p>
</template>
