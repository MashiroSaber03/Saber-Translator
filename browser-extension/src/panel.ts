import {
  jobKindLabel,
  stepKindLabel,
} from '../../vue-frontend/src/utils/taskDisplay'
import './panel.css'
import { createPluginSettingsEditor, field, section, type Values } from '../../src/shared/browserExtensionSettings'
import { loadSettings } from './storage'
import type { components } from '../../vue-frontend/src/api/generated/v2'

type Schema = components['schemas']
const content = document.querySelector<HTMLElement>('#content')!
const notice = document.querySelector<HTMLElement>('#notice')!
let view = ''
let generation = 0
let taskBusy = false
let taskSnapshot = ''
let timer: ReturnType<typeof setTimeout> | undefined

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = '',
  className = '',
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  node.textContent = text
  node.className = className
  return node
}
function message(text: string, error = false): void {
  notice.textContent = text
  notice.dataset.error = String(error)
}
async function api<T>(
  path: string,
  method = 'GET',
  body?: unknown,
): Promise<T> {
  const settings = await loadSettings()
  if (!settings.token) throw new Error('请先在工具栏弹窗完成配对。')
  const response = await fetch(
    `http://127.0.0.1:${settings.serverPort}/api/v2/browser-extension/manage${path}`,
    {
      method,
      cache: 'no-store',
      credentials: 'omit',
      signal: AbortSignal.timeout(120_000),
      headers: {
        Authorization: `Bearer ${settings.token}`,
        'Content-Type': 'application/json',
        ...(method !== 'GET' ? { 'Idempotency-Key': crypto.randomUUID() } : {}),
      },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    },
  )
  const result = await response.json()
  if (!response.ok)
    throw new Error(
      response.status === 409
        ? '设置或任务状态已在其他窗口变化，请重新读取后再操作。'
        : (result.error?.message ?? `请求失败（${response.status}）`),
    )
  return result as T
}
function button(
  text: string,
  action: () => Promise<unknown>,
  primary = false,
): HTMLButtonElement {
  const node = el('button', text, primary ? 'primary' : '')
  node.type = 'button'
  node.addEventListener('click', () => {
    node.disabled = true
    void action()
      .catch((error) =>
        message(error instanceof Error ? error.message : '操作失败', true),
      )
      .finally(() => {
        node.disabled = false
      })
  })
  return node
}
const settingsContainer = el('div')
const settingsEditor = createPluginSettingsEditor(
  settingsContainer,
  api,
  message,
)

const statuses: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  paused: '已暂停',
  interrupted: '已中断',
  completed: '已完成',
  completed_with_errors: '部分失败',
  failed: '失败',
  cancelled: '已取消',
}
let taskFilter = 'all'
const expanded = new Set<string>()
async function tasksView(version: number): Promise<void> {
  const result = await api<Schema['JobList']>('/jobs?scope=all&limit=200')
  if (version !== generation || taskBusy) return
  const snapshot = JSON.stringify([result, taskFilter])
  if (snapshot === taskSnapshot) return
  taskSnapshot = snapshot
  const fragment = document.createDocumentFragment()
  const controls = section('Saber 任务')
  controls.append(
    el(
      'p',
      '与 GUI、Web 共用队列。退出漫画页面后，该页面临时任务会取消并清理。',
      'hint',
    ),
  )
  const filters: Values = { filter: taskFilter }
  field(
    controls,
    '显示范围',
    filters,
    'filter',
    [
      ['all', '全部 Saber 任务'],
      ['active', '未完成任务'],
    ],
    () => {
      taskFilter = String(filters.filter)
      void refreshTasks(version)
    },
  )
  controls.append(
    el(
      'p',
      result.workerOnline ? '工作进程在线' : '工作进程未就绪，请启动 Saber。',
      'hint',
    ),
  )
  if (result.queuePaused)
    controls.append(el('p', '全局队列已暂停，新任务等待执行。', 'hint'))
  controls.append(
    button(result.queuePaused ? '恢复全局队列' : '暂停全局队列', async () => {
      await taskCommand(
        `/jobs/queue/${result.queuePaused ? 'resume' : 'pause'}`,
        version,
      )
    }),
    button('刷新', () => refreshTasks(version)),
  )
  fragment.append(controls)
  const jobs = result.items.filter(
    (job) =>
      taskFilter === 'all' ||
      ['queued', 'running', 'paused', 'interrupted'].includes(job.status),
  )
  for (const job of jobs) {
    const box = section(
      String(
        job.target.chapter ?? job.target.book ?? job.batchDisplayName ?? '任务',
      ),
    )
    const p = job.progress
    box.append(
      el(
        'div',
        `${jobKindLabel(job.kind)} · ${statuses[job.status]} · 成功 ${p.completedItems}/${p.totalItems} · 失败 ${p.failedItems}`,
        'job-progress',
      ),
    )
    if (p.currentStep)
      box.append(
        el(
          'p',
          `第 ${p.currentStep.itemOrdinal} 张 · ${stepKindLabel(p.currentStep.kind)}`,
          'hint',
        ),
      )
    if (job.blockedReason)
      box.append(el('p', '等待同章节的其他任务释放占用', 'hint'))
    const progress = el('progress')
    progress.max = p.totalItems || 1
    progress.value =
      p.completedItems + p.failedItems + p.cancelledItems + p.skippedItems
    box.append(progress)
    const actions = el('div', '', 'actions')
    const command = (name: string, action: string) =>
      actions.append(
        button(name, async () => {
          await taskCommand(
            `/jobs/${job.jobId}/${action}`,
            version,
            action.startsWith('retry') ? { strategy: 'current' } : undefined,
          )
        }),
      )
    if (job.status === 'running') command('暂停', 'pause')
    if (job.status === 'paused') command('恢复', 'resume')
    if (job.status === 'interrupted') command('继续', 'continue')
    if (['queued', 'running', 'paused', 'interrupted'].includes(job.status))
      command('取消', 'cancel')
    if (job.status === 'completed_with_errors')
      command('按当前设置重试失败页', 'retry-failed')
    if (job.status === 'failed') command('按当前设置重试', 'retry')
    const detail = el('div', '', 'job-detail')
    actions.append(
      button(expanded.has(job.jobId) ? '收起详情' : '查看详情', async () => {
        if (expanded.has(job.jobId)) {
          expanded.delete(job.jobId)
          detail.replaceChildren()
        } else {
          expanded.add(job.jobId)
          await showDetail(job.jobId, detail)
        }
      }),
    )
    box.append(actions, detail)
    if (expanded.has(job.jobId))
      void showDetail(job.jobId, detail).catch(() => undefined)
    fragment.append(box)
  }
  if (!jobs.length)
    fragment.append(
      el('p', '暂无任务。在漫画页面中选择图片即可开始。', 'empty'),
    )
  content.replaceChildren(fragment)
}
async function showDetail(jobId: string, target: HTMLElement): Promise<void> {
  const detail = await api<Schema['JobDetail']>(`/jobs/${jobId}`)
  target.replaceChildren(el('p', `任务 ${jobId.slice(0, 8)}`))
  if (detail.error)
    target.append(
      el(
        'p',
        typeof detail.error === 'string'
          ? detail.error
          : String(detail.error.message ?? '任务失败'),
        'error',
      ),
    )
  for (const item of detail.failedItems)
    target.append(
      el(
        'p',
        `第 ${item.ordinal} 张：${typeof item.error === 'string' ? item.error : (item.error?.message ?? '处理失败')}`,
        'error',
      ),
    )
}
async function taskCommand(
  path: string,
  version: number,
  payload?: unknown,
): Promise<void> {
  if (taskBusy) return
  taskBusy = true
  clearTimeout(timer)
  try {
    await api(path, 'POST', payload)
  } finally {
    taskBusy = false
    await refreshTasks(version)
  }
}
async function refreshTasks(version: number): Promise<void> {
  clearTimeout(timer)
  if (taskBusy || version !== generation) return
  try {
    await tasksView(version)
  } catch (error) {
    if (generation === version)
      message(error instanceof Error ? error.message : '读取任务失败', true)
  } finally {
    if (version === generation && view === 'tasks')
      timer = setTimeout(() => {
        void refreshTasks(version)
      }, 3000)
  }
}
async function navigate(): Promise<void> {
  view = location.hash === '#settings' ? 'settings' : 'tasks'
  const version = ++generation
  taskSnapshot = ''
  clearTimeout(timer)
  message('')
  content.replaceChildren(el('p', '正在读取…', 'hint'))
  for (const name of ['settings', 'tasks'])
    document
      .getElementById(`${name}-tab`)!
      .classList.toggle('active', name === view)
  try {
    if (view === 'settings') {
      content.replaceChildren(settingsContainer)
      if (!settingsContainer.childElementCount) await settingsEditor.load()
    } else await refreshTasks(version)
  } catch (error) {
    if (version === generation)
      message(error instanceof Error ? error.message : '读取失败', true)
  }
}
for (const name of ['settings', 'tasks'])
  document.getElementById(`${name}-tab`)!.addEventListener('click', () => {
    location.hash = name
  })
window.addEventListener('hashchange', () => {
  void navigate()
})
void navigate()
