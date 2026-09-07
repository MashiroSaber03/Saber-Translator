<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import BrowserExtensionSettingsForm from '@/components/settings/BrowserExtensionSettingsForm.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { jobKindLabel, stepKindLabel } from '@/utils/taskDisplay'
import type { components } from '@/api/generated/v2'
import type { PluginSettingsApi } from '@/types/browserExtensionSettings'

type Schema = components['schemas']
const props = defineProps<{ api: PluginSettingsApi }>()
const tab = ref(location.hash === '#settings' ? 'settings' : 'tasks')
const settingsVisited = ref(tab.value === 'settings')
const visible = ref(true)
const active = computed(() => visible.value && tab.value === 'tasks')
const queue = ref<Schema['JobList'] | null>(null)
const filter = ref('all')
const details = ref<Record<string, Schema['JobDetail'] | null>>({})
const loadError = ref('')
const commandError = ref('')
const busy = ref(false)
let timer: ReturnType<typeof setTimeout> | undefined
const tabs = [
  { id: 'settings', label: '翻译配置' },
  { id: 'tasks', label: '任务中心' },
]
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

function changeTab() {
  tab.value = location.hash === '#settings' ? 'settings' : 'tasks'
  if (tab.value === 'settings') settingsVisited.value = true
}
function selectTab(value: string) {
  location.hash = value
  changeTab()
}
function openFromFloatingWindow(event: MessageEvent) {
  if (event.source !== window.parent || event.data?.type !== 'saber-open-management') return
  const section = event.data.section
  if (section === 'settings' || section === 'tasks') selectTab(section)
}
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
    if (active.value) timer = setTimeout(() => void refresh(), 3000)
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

// Intersection visibility includes a hidden parent floating window, unlike document.hidden.
const visibility = new IntersectionObserver(([entry]) => {
  visible.value = entry!.isIntersecting
})
watch(
  active,
  value => {
    clearTimeout(timer)
    if (value) void refresh()
  },
  { immediate: true }
)
onMounted(() => {
  visibility.observe(document.documentElement)
  window.addEventListener('hashchange', changeTab)
  window.addEventListener('message', openFromFloatingWindow)
})
onBeforeUnmount(() => {
  visible.value = false
  clearTimeout(timer)
  visibility.disconnect()
  window.removeEventListener('hashchange', changeTab)
  window.removeEventListener('message', openFromFloatingWindow)
})
</script>

<template>
  <nav class="management-tabs">
    <ProductSegmentedTabs
      :tabs="tabs"
      :active-tab="tab"
      aria-label="插件功能"
      appearance="underline"
      @update:active-tab="selectTab"
    />
  </nav>
  <main>
    <BrowserExtensionSettingsForm v-if="settingsVisited" v-show="tab === 'settings'" :api="api" />
    <div v-show="tab === 'tasks'" class="task-center">
      <ProductStatusBanner v-if="commandError || loadError" tone="danger" role="status">{{
        commandError || loadError
      }}</ProductStatusBanner>
      <section class="task-card">
        <p class="hint">与翻译器共用任务队列。退出漫画页面后，临时任务会取消并清理。</p>
        <UiField label="显示范围" control-id="task-filter">
          <UiSelect id="task-filter" v-model="filter" :options="filters" />
        </UiField>
        <p v-if="queue" class="hint">
          {{ queue.workerOnline ? '工作进程在线' : '工作进程未就绪，请启动 Saber。' }}
        </p>
        <p v-if="queue?.queuePaused" class="hint">全局队列已暂停，新任务等待执行。</p>
        <ProductActionRow justify="start">
          <UiButton
            v-if="queue"
            size="sm"
            :disabled="busy"
            @click="command(`/jobs/queue/${queue.queuePaused ? 'resume' : 'pause'}`)"
            >{{ queue.queuePaused ? '恢复全局队列' : '暂停全局队列' }}</UiButton
          >
          <UiButton size="sm" :disabled="busy" @click="refresh">刷新</UiButton>
        </ProductActionRow>
      </section>
      <p v-if="!queue && !loadError" class="empty">正在读取任务…</p>
      <p v-else-if="queue && !jobs.length" class="empty">
        暂无任务。在漫画页面中选择图片即可开始。
      </p>
      <section v-for="job in jobs" :key="job.jobId" class="task-card">
        <h2>{{ job.target.chapter ?? job.target.book ?? job.batchDisplayName ?? '任务' }}</h2>
        <p class="hint">
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
        <p v-if="job.progress.currentStep" class="hint">
          第 {{ job.progress.currentStep.itemOrdinal }} 张 ·
          {{ stepKindLabel(job.progress.currentStep.kind) }}
        </p>
        <p v-if="job.blockedReason" class="hint">等待同章节的其他任务释放占用</p>
        <ProductActionRow justify="start">
          <UiButton
            v-if="job.status === 'running'"
            size="sm"
            :disabled="busy"
            @click="jobCommand(job.jobId, 'pause')"
            >暂停</UiButton
          >
          <UiButton
            v-if="job.status === 'paused'"
            size="sm"
            :disabled="busy"
            @click="jobCommand(job.jobId, 'resume')"
            >恢复</UiButton
          >
          <UiButton
            v-if="job.status === 'interrupted'"
            size="sm"
            :disabled="busy"
            @click="jobCommand(job.jobId, 'continue')"
            >继续</UiButton
          >
          <UiButton
            v-if="unfinished.includes(job.status)"
            size="sm"
            variant="plain-danger"
            :disabled="busy"
            @click="jobCommand(job.jobId, 'cancel')"
            >取消</UiButton
          >
          <UiButton
            v-if="job.status === 'completed_with_errors'"
            size="sm"
            :disabled="busy"
            @click="jobCommand(job.jobId, 'retry-failed')"
            >按当前设置重试失败页</UiButton
          >
          <UiButton
            v-if="job.status === 'failed'"
            size="sm"
            :disabled="busy"
            @click="jobCommand(job.jobId, 'retry')"
            >按当前设置重试</UiButton
          >
          <UiButton
            size="sm"
            :aria-expanded="job.jobId in details"
            @click="toggleDetail(job.jobId)"
            >{{ job.jobId in details ? '收起详情' : '查看详情' }}</UiButton
          >
        </ProductActionRow>
        <div v-if="job.jobId in details" class="job-detail">
          <template v-if="details[job.jobId]">
            <p>任务 {{ job.jobId.slice(0, 8) }}</p>
            <p v-if="details[job.jobId]!.error" class="error">
              {{ jobError(details[job.jobId]!.error) }}
            </p>
            <p v-for="item in details[job.jobId]!.failedItems" :key="item.ordinal" class="error">
              第 {{ item.ordinal }} 张：{{ jobError(item.error) }}
            </p>
          </template>
          <p v-else>正在读取详情…</p>
        </div>
      </section>
    </div>
  </main>
</template>

<style scoped>
.management-tabs {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 10px 12px;
  background: var(--color-surface-app);
}
main {
  padding: 0 12px 16px;
}
.task-center {
  display: grid;
  gap: 12px;
}
.task-card {
  padding: 14px;
  border: 1px solid var(--color-border-default);
  border-radius: 12px;
  background: var(--color-surface-panel);
}
.task-card h2 {
  margin: 0 0 8px;
  font-size: 15px;
  overflow-wrap: anywhere;
}
.hint,
.job-detail {
  margin: 8px 0;
  color: var(--color-text-supporting);
  font-size: 12px;
}
.task-card > .hint:first-child {
  margin-top: 0;
}
.task-card progress {
  width: 100%;
  height: 8px;
  accent-color: var(--color-action-primary);
}
.empty {
  padding: 20px 8px;
  text-align: center;
  color: var(--color-text-supporting);
}
.error {
  color: var(--color-text-danger);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
</style>
