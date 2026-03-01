<script setup lang="ts">
import { onMounted, ref } from 'vue'
import AppHeader from '@/components/common/AppHeader.vue'
import { cancelJob, getJobs, retryJob } from '@/api/jobs'
import type { JobItem } from '@/types'
import { showToast } from '@/utils/toast'

const loading = ref(false)
const jobs = ref<JobItem[]>([])
const statusFilter = ref('')
const typeFilter = ref('')

async function loadJobs(): Promise<void> {
  loading.value = true
  try {
    const res = await getJobs({
      status: statusFilter.value || undefined,
      type: typeFilter.value || undefined,
      limit: 300
    })
    if (!res.success) throw new Error(res.error || '加载任务失败')
    jobs.value = res.jobs || []
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}

async function handleRetry(job: JobItem): Promise<void> {
  loading.value = true
  try {
    const res = await retryJob(job.id)
    if (!res.success) throw new Error(res.error || '重试失败')
    showToast('任务重试完成', 'success')
    await loadJobs()
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}

async function handleCancel(job: JobItem): Promise<void> {
  loading.value = true
  try {
    const res = await cancelJob(job.id)
    if (!res.success) throw new Error(res.error || '取消失败')
    showToast('任务已取消', 'success')
    await loadJobs()
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}

function formatTime(value?: string | null): string {
  if (!value) return '-'
  return value.replace('T', ' ').replace('Z', '')
}

onMounted(() => {
  loadJobs()
})
</script>

<template>
  <div class="jobs-page">
    <AppHeader variant="bookshelf" />

    <main class="jobs-main">
      <section class="panel">
        <h1>批处理任务中心</h1>
        <p>查看任务状态、失败原因，并支持重试与取消。</p>

        <div class="filters">
          <label for="jobs-status-filter" class="sr-only">按状态筛选</label>
          <select id="jobs-status-filter" v-model="statusFilter">
            <option value="">全部状态</option>
            <option value="pending">pending</option>
            <option value="running">running</option>
            <option value="completed">completed</option>
            <option value="failed">failed</option>
            <option value="cancelled">cancelled</option>
          </select>
          <label for="jobs-type-filter" class="sr-only">按类型筛选</label>
          <select id="jobs-type-filter" v-model="typeFilter">
            <option value="">全部类型</option>
            <option value="quality">quality</option>
            <option value="compare">compare</option>
          </select>
          <button type="button" :disabled="loading" @click="loadJobs">刷新</button>
        </div>
      </section>

      <section class="panel table-panel">
        <table>
          <caption class="sr-only">批处理任务列表</caption>
          <thead>
            <tr>
              <th scope="col">ID</th>
              <th scope="col">类型</th>
              <th scope="col">状态</th>
              <th scope="col">重试次数</th>
              <th scope="col">创建时间</th>
              <th scope="col">结束时间</th>
              <th scope="col">信息</th>
              <th scope="col">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="job in jobs" :key="job.id">
              <td>{{ job.id }}</td>
              <td>{{ job.type }}</td>
              <td>
                <span class="status" :class="`status-${job.status}`">{{ job.status }}</span>
              </td>
              <td>{{ job.retry_count }}</td>
              <td>{{ formatTime(job.created_at) }}</td>
              <td>{{ formatTime(job.finished_at) }}</td>
              <td class="message-cell">
                <div v-if="job.error" class="error">{{ job.error }}</div>
                <div v-else-if="job.result">{{ JSON.stringify(job.result) }}</div>
                <div v-else>-</div>
              </td>
              <td class="actions">
                <button type="button" class="secondary" :disabled="loading" @click="handleRetry(job)">重试</button>
                <button type="button" class="danger" :disabled="loading" @click="handleCancel(job)">取消</button>
              </td>
            </tr>
            <tr v-if="jobs.length === 0">
              <td colspan="8" class="empty">暂无任务</td>
            </tr>
          </tbody>
        </table>
      </section>
    </main>
  </div>
</template>

<style scoped>
.jobs-page {
  min-height: 100vh;
  background: linear-gradient(160deg, #f8fafd 0%, #f5fff8 100%);
}

.jobs-main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  display: grid;
  gap: 16px;
}

.panel {
  background: #fff;
  border: 1px solid #dce7f2;
  border-radius: 14px;
  padding: 16px;
}

.panel h1 {
  margin: 0 0 8px;
}

.panel p {
  margin: 0 0 12px;
  color: #56677b;
}

.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

select,
button {
  border-radius: 10px;
  padding: 8px 10px;
}

select {
  border: 1px solid #c8d6e6;
}

button {
  border: none;
  background: #1677ff;
  color: #fff;
  cursor: pointer;
}

button.secondary {
  background: #5f6f85;
}

button.danger {
  background: #e25544;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.table-panel {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  border-bottom: 1px solid #edf2f8;
  text-align: left;
  padding: 10px 8px;
  font-size: 14px;
  vertical-align: top;
}

.status {
  display: inline-block;
  border-radius: 8px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 600;
}

.status-pending {
  background: #eef3fa;
  color: #4b6077;
}

.status-running {
  background: #ddeeff;
  color: #0f5db5;
}

.status-completed {
  background: #e2f6de;
  color: #2d7a33;
}

.status-failed {
  background: #ffe1e1;
  color: #b63232;
}

.status-cancelled {
  background: #f0e8dc;
  color: #7b5a2f;
}

.message-cell {
  max-width: 280px;
  word-break: break-word;
}

.message-cell .error {
  color: #c03a2c;
}

.actions {
  display: flex;
  gap: 6px;
}

.empty {
  text-align: center;
  color: #5b6d82;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@media (max-width: 768px) {
  .jobs-main {
    padding: 12px;
  }
}
</style>
