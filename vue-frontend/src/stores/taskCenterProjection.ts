import type { V2Job } from '@/api/v2/jobs'

export interface JobBatchProjection {
  key: string
  batchId: string | null
  displayName: string
  jobs: V2Job[]
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

export function describeJobBatch(job: V2Job): string {
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
  return name || job.kind
}

export function progressPercent(job: V2Job): number {
  const progress = job.progress as Record<string, unknown>
  const total = Number(progress.totalItems || 0)
  const completed = Number(progress.completedItems || 0)
  const failed = Number(progress.failedItems || 0)
  if (!Number.isFinite(total) || total <= 0) return 0
  return Math.max(0, Math.min(100, Math.round(((completed + failed) / total) * 100)))
}
