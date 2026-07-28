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
        displayName: job.batchDisplayName || describeJobTarget(job),
        jobs: [],
      }
      groups.set(key, group)
    }
    group.jobs.push(job)
  }
  return [...groups.values()]
}

export function describeJobTarget(job: V2Job): string {
  const target = job.target as Record<string, unknown>
  const named = target.chapter || target.book || target.page || target.name
  return typeof named === 'string' && named ? named : job.kind
}

export function progressPercent(job: V2Job): number {
  const progress = job.progress as Record<string, unknown>
  const total = Number(progress.totalItems || 0)
  const completed = Number(progress.completedItems || 0)
  const failed = Number(progress.failedItems || 0)
  if (!Number.isFinite(total) || total <= 0) return 0
  return Math.max(0, Math.min(100, Math.round(((completed + failed) / total) * 100)))
}
