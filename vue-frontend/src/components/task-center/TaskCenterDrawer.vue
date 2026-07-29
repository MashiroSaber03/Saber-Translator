<script setup lang="ts">
import { computed, ref } from 'vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { V2Job } from '@/api/v2/jobs'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { describeJobTarget, progressPercent } from '@/stores/taskCenterProjection'
import { jobsApi } from '@/api/v2/jobs'
import { triggerUrlDownload } from '@/utils/browserDownload'
import { showToast } from '@/utils/toast'

const store = useTaskCenterStore()
const tab = ref<'queue' | 'history'>('queue')
const expanded = ref(new Set<string>())
const downloading = ref(new Set<string>())

const groups = computed(() => tab.value === 'queue' ? store.queueBatches : store.historyBatches)

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
  return ['queued', 'running', 'pausing', 'paused', 'cancelling', 'interrupted'].includes(job.status)
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
          <UiButton size="sm" variant="ghost" class="task-center__close" @click="store.close">
            关闭
          </UiButton>
        </header>

        <div v-if="!store.connected" class="task-center__offline">
          任务事件连接正在重连，当前列表来自持久化快照。
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
            历史 {{ store.history.length }}
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

        <main class="task-center__content">
          <p v-if="store.loading && !groups.length" class="task-center__empty">正在读取后端任务…</p>
          <p v-else-if="!groups.length" class="task-center__empty">这里还没有任务</p>

          <section v-for="group in groups" :key="group.key" class="task-batch">
            <UiButton
              variant="card-action"
              block
              class="task-batch__header"
              @click="toggle(group.key)"
            >
              <span>
                <strong>{{ group.displayName }}</strong>
                <small>{{ group.jobs.length }} 个任务</small>
              </span>
              <span>{{ expanded.has(group.key) || group.jobs.length === 1 ? '收起' : '展开' }}</span>
            </UiButton>

            <div v-if="expanded.has(group.key) || group.jobs.length === 1" class="task-batch__jobs">
              <article v-for="job in group.jobs" :key="job.jobId" class="task-job">
                <div class="task-job__top">
                  <div>
                    <strong>{{ describeJobTarget(job) }}</strong>
                    <span>{{ job.kind }}</span>
                  </div>
                  <span class="task-job__status" :data-status="job.status">
                    {{ statusLabels[job.status] || job.status }}
                  </span>
                </div>
                <div class="task-job__progress">
                  <span :style="{ width: `${progressPercent(job)}%` }" />
                </div>
                <p v-if="job.blockedReason" class="task-job__hint">
                  {{ job.blockedReason === 'draining_immediate_writes' ? '正在排空即时写入，新编辑已暂停' : `等待：${job.blockedReason}` }}
                </p>
                <div class="task-job__actions">
                  <UiButton
                    v-if="job.status === 'completed'"
                    size="xs"
                    variant="secondary"
                    class="task-job__action"
                    :disabled="downloading.has(job.jobId)"
                    @click="downloadArtifact(job)"
                  >
                    {{ downloading.has(job.jobId) ? '读取中…' : '下载产物' }}
                  </UiButton>
                  <UiButton v-if="job.status === 'running'" size="xs" variant="secondary" @click="store.pause(job.jobId)">暂停</UiButton>
                  <UiButton v-if="job.status === 'paused'" size="xs" variant="secondary" @click="store.resume(job.jobId)">继续</UiButton>
                  <UiButton v-if="job.status === 'interrupted'" size="xs" variant="secondary" @click="store.continueJob(job.jobId)">从检查点继续</UiButton>
                  <UiButton v-if="canCancel(job)" size="xs" variant="ghost" tone="danger" @click="store.cancel(job.jobId)">取消</UiButton>
                </div>
              </article>
            </div>
          </section>
        </main>
      </aside>
    </OverlayLayer>
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

.task-job {
  padding: 14px;
  border-top: 1px solid var(--color-border-default);
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
  margin: 12px 0;
  overflow: hidden;
  background: var(--color-surface-muted);
  border-radius: 999px;
}

.task-job__progress span {
  display: block;
  height: 100%;
  background: var(--color-action-primary);
}

.task-job__hint {
  margin: 0 0 10px;
}

.task-job__actions {
  justify-content: flex-end;
}
</style>
