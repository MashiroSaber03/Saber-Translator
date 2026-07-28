<script setup lang="ts">
import { computed, ref } from 'vue'
import type { V2Job } from '@/api/v2/jobs'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { describeJobTarget, progressPercent } from '@/stores/taskCenterProjection'

const store = useTaskCenterStore()
const tab = ref<'queue' | 'history'>('queue')
const expanded = ref(new Set<string>())

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
</script>

<template>
  <Teleport to="body">
    <div v-if="store.drawerOpen" class="task-center" role="dialog" aria-modal="true" aria-label="任务中心">
      <button class="task-center__backdrop" type="button" aria-label="关闭任务中心" @click="store.close" />
      <aside class="task-center__panel">
        <header class="task-center__header">
          <div>
            <h2>任务中心</h2>
            <p>后端持续执行 · 关闭页面不会停止任务</p>
          </div>
          <button type="button" class="task-center__close" @click="store.close">关闭</button>
        </header>

        <div v-if="!store.connected" class="task-center__offline">
          任务事件连接正在重连，当前列表来自持久化快照。
        </div>

        <nav class="task-center__tabs" aria-label="任务分区">
          <button type="button" :class="{ active: tab === 'queue' }" @click="tab = 'queue'">
            队列 {{ store.queue.length }}
          </button>
          <button type="button" :class="{ active: tab === 'history' }" @click="tab = 'history'">
            历史 {{ store.history.length }}
          </button>
          <span class="task-center__spacer" />
          <button v-if="tab === 'queue' && store.queuedCount" type="button" @click="store.cancelQueued">
            取消全部排队
          </button>
          <button v-if="tab === 'history' && store.history.length" type="button" @click="store.clearHistory">
            清空历史
          </button>
        </nav>

        <main class="task-center__content">
          <p v-if="store.loading && !groups.length" class="task-center__empty">正在读取后端任务…</p>
          <p v-else-if="!groups.length" class="task-center__empty">这里还没有任务</p>

          <section v-for="group in groups" :key="group.key" class="task-batch">
            <button type="button" class="task-batch__header" @click="toggle(group.key)">
              <span>
                <strong>{{ group.displayName }}</strong>
                <small>{{ group.jobs.length }} 个任务</small>
              </span>
              <span>{{ expanded.has(group.key) || group.jobs.length === 1 ? '收起' : '展开' }}</span>
            </button>

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
                  <button v-if="job.status === 'running'" type="button" @click="store.pause(job.jobId)">暂停</button>
                  <button v-if="job.status === 'paused'" type="button" @click="store.resume(job.jobId)">继续</button>
                  <button v-if="job.status === 'interrupted'" type="button" @click="store.continueJob(job.jobId)">从检查点继续</button>
                  <button v-if="canCancel(job)" type="button" @click="store.cancel(job.jobId)">取消</button>
                </div>
              </article>
            </div>
          </section>
        </main>
      </aside>
    </div>
  </Teleport>
</template>

<style scoped>
.task-center {
  position: fixed;
  z-index: var(--z-modal, 1000);
  inset: 0;
}

.task-center__backdrop {
  position: absolute;
  inset: 0;
  width: 100%;
  background: rgb(15 23 42 / 42%);
  border: 0;
}

.task-center__panel {
  position: absolute;
  top: 0;
  right: 0;
  display: flex;
  flex-direction: column;
  width: min(520px, 100vw);
  height: 100%;
  color: var(--color-text);
  background: var(--color-surface);
  border-left: 1px solid var(--color-border);
  box-shadow: var(--shadow-xl);
}

.task-center__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 22px 24px 16px;
  border-bottom: 1px solid var(--color-border);
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
.task-center__tabs button,
.task-job__actions button {
  padding: 7px 10px;
  color: var(--color-text);
  font: inherit;
  font-size: 12px;
  background: var(--color-surface-muted);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  cursor: pointer;
}

.task-center__offline {
  padding: 9px 24px;
  color: var(--color-warning-text, #92400e);
  font-size: 12px;
  background: var(--color-warning-subtle, #fef3c7);
}

.task-center__tabs {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 12px 24px;
  border-bottom: 1px solid var(--color-border);
}

.task-center__tabs button.active {
  color: var(--color-primary);
  border-color: var(--color-primary);
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
  border: 1px solid var(--color-border);
  border-radius: 12px;
}

.task-batch__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 12px 14px;
  color: var(--color-text);
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
  border-top: 1px solid var(--color-border);
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
  color: var(--color-success, #15803d);
}

.task-job__status[data-status='failed'],
.task-job__status[data-status='interrupted'] {
  color: var(--color-danger, #dc2626);
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
  background: var(--color-primary);
}

.task-job__hint {
  margin: 0 0 10px;
}

.task-job__actions {
  justify-content: flex-end;
}
</style>
