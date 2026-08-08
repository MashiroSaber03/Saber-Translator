import type { V2Job } from '@/api/v2/jobs'

export interface InsightPageProgress {
  current: number
  total: number
  currentStepKind: string
}

export type InsightTerminalEventType = 'job_finished' | 'job_failed' | 'job_cancelled'

export function projectInsightPageProgress(
  progress: V2Job['progress'],
): InsightPageProgress {
  const pagePool = progress.pools?.find(
    pool => pool.kind === 'insight_analyze_page',
  )
  const currentStepKind = progress.currentStep?.kind ?? ''

  if (!pagePool) {
    return { current: 0, total: 0, currentStepKind }
  }

  return {
    current: Math.min(
      pagePool.total,
      pagePool.completed + pagePool.failed + pagePool.skipped,
    ),
    total: pagePool.total,
    currentStepKind,
  }
}

export function projectTerminalInsightPageProgress(
  progress: V2Job['progress'],
  eventType: InsightTerminalEventType,
): InsightPageProgress {
  const projected = projectInsightPageProgress(progress)
  return {
    ...projected,
    current: eventType === 'job_finished' ? projected.total : projected.current,
    currentStepKind: '',
  }
}
