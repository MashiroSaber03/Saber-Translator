import type { V2Job } from '@/api/v2/jobs'
import { projectInsightPageProgress } from '@/utils/insightJobProgress'
import { jobKindLabel, stepKindLabel } from '@/utils/taskDisplay'

export interface JobBatchProjection {
  key: string
  batchId: string | null
  displayName: string
  jobs: V2Job[]
}

export interface JobProgressCounts {
  completed: number
  total: number
}

export interface JobPoolProjection {
  kind: string
  waiting: number
  processing: number
  completed: number
  lockWaiting: boolean
}

export function groupJobsByBatch(jobs: V2Job[]): JobBatchProjection[] {
  const groups = new Map<string, JobBatchProjection>()
  for (const job of jobs) {
    const key = job.batchId || `job:${job.jobId}`
    let group = groups.get(key)
    if (!group) {
      group = {
        key,
        batchId: job.batchId || null,
        displayName: describeJobBatch(job),
        jobs: [],
      }
      groups.set(key, group)
    }
    group.jobs.push(job)
  }
  return [...groups.values()]
}

function targetText(job: V2Job, key: string): string {
  const value = (job.target as Record<string, unknown>)[key]
  return typeof value === 'string' ? value.trim() : ''
}

function targetPageCount(job: V2Job): number {
  const value = Number((job.target as Record<string, unknown>).pageCount)
  return Number.isInteger(value) && value > 0 ? value : 0
}

function describeJobBatch(job: V2Job): string {
  const book = targetText(job, 'book')
  const chapter = targetText(job, 'chapter')
  if (book && chapter) return `书籍：${book} · 章节：${chapter}`
  if (book) return `书籍：${book}`
  if (chapter) return `章节：${chapter}`
  return job.batchDisplayName || describeJobTarget(job)
}

export function describeJobTarget(job: V2Job): string {
  const chapter = targetText(job, 'chapter')
  const book = targetText(job, 'book')
  const page = targetText(job, 'page')
  const name = targetText(job, 'name')
  const pageCount = targetPageCount(job)
  if (chapter) return `章节：${chapter}${pageCount ? ` · ${pageCount} 页` : ''}`
  if (book) return `书籍：${book}${pageCount ? ` · ${pageCount} 页` : ''}`
  if (page) return `页面：${page}`
  return name || jobKindLabel(job.kind)
}

export function progressPercent(job: V2Job): number {
  const counts = progressCounts(job)
  if (counts.total <= 0) return 0
  return Math.max(0, Math.min(100, counts.completed / counts.total * 100))
}

export function progressCounts(job: V2Job): JobProgressCounts {
  const progress = job.progress as Record<string, unknown>
  const finiteCount = (value: unknown): number => {
    const count = Number(value)
    return Number.isFinite(count) && count > 0 ? count : 0
  }
  const total = finiteCount(progress.totalItems)
  const completed = (
    finiteCount(progress.completedItems)
    + finiteCount(progress.failedItems)
    + finiteCount(progress.skippedItems)
    + finiteCount(progress.cancelledItems)
  )
  if (job.kind === 'insight_analysis') {
    const pageProgress = projectInsightPageProgress(job.progress)
    if (pageProgress.total > 0) {
      return {
        completed: pageProgress.current,
        total: pageProgress.total,
      }
    }
    const pageCount = targetPageCount(job)
    if (pageCount > 0) {
      return {
        completed: Math.min(completed, pageCount),
        total: pageCount,
      }
    }
  }
  return {
    completed: Math.min(completed, total),
    total,
  }
}

export function batchProgressCounts(jobs: V2Job[]): JobProgressCounts {
  return jobs.reduce<JobProgressCounts>(
    (summary, job) => {
      const counts = progressCounts(job)
      summary.completed += counts.completed
      summary.total += counts.total
      return summary
    },
    { completed: 0, total: 0 },
  )
}

export function batchStatusCounts(jobs: V2Job[]): Array<[V2Job['status'], number]> {
  const counts = new Map<V2Job['status'], number>()
  for (const job of jobs) counts.set(job.status, (counts.get(job.status) ?? 0) + 1)
  return [...counts]
}

export function currentStepLabel(job: V2Job): string {
  const progress = job.progress as Record<string, unknown>
  const current = progress.currentStep
  if (!current || typeof current !== 'object' || Array.isArray(current)) return ''
  const step = current as Record<string, unknown>
  const kind = typeof step.kind === 'string' ? step.kind : ''
  const ordinal = Number(step.itemOrdinal)
  const page = typeof step.pageId === 'string' ? step.pageId.slice(0, 8) : ''
  const target = Number.isInteger(ordinal) && ordinal > 0
    ? `第 ${ordinal} 项`
    : page
      ? `页面 ${page}`
      : ''
  return [target, kind ? stepKindLabel(kind) : ''].filter(Boolean).join(' · ')
}

export function poolProgress(job: V2Job): JobPoolProjection[] {
  const progress = job.progress as Record<string, unknown>
  if (progress.executionMode !== 'parallel' || !Array.isArray(progress.pools)) return []
  const count = (value: unknown): number => {
    const parsed = Number(value)
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0
  }
  return progress.pools.flatMap((value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const pool = value as Record<string, unknown>
    if (typeof pool.kind !== 'string' || !pool.kind) return []
    return [{
      kind: pool.kind,
      waiting: count(pool.waiting),
      processing: count(pool.processing),
      completed: count(pool.completed),
      lockWaiting: Boolean(pool.lockWaiting),
    }]
  })
}
