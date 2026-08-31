import type { ContinuationState } from './useContinuationState'
import type { V2JobDetail } from '@/api/v2/jobs'

export interface ContinuationMutationOptions<T> {
  state: ContinuationState
  failurePrefix: string
  successMessage?: string
  run: () => Promise<T>
  isCurrent?: () => boolean
  afterSuccess?: (result: T) => Promise<void> | void
  onFailure?: () => Promise<void> | void
}

function toContinuationActionError(error: unknown): string {
  return error instanceof Error ? error.message : '网络错误'
}

function formatContinuationActionError(prefix: string, message: string): string {
  return `${prefix}: ${message}`
}

export async function runContinuationMutation<T>(
  options: ContinuationMutationOptions<T>
): Promise<boolean> {
  try {
    const result = await options.run()
    if (options.isCurrent && !options.isCurrent()) return false

    await options.afterSuccess?.(result)
    if (options.isCurrent && !options.isCurrent()) return false
    if (options.successMessage) {
      options.state.showMessage(options.successMessage, 'success')
    }
    return true
  } catch (error) {
    await options.onFailure?.()
    if (!options.isCurrent || options.isCurrent()) {
      options.state.showMessage(
        formatContinuationActionError(options.failurePrefix, toContinuationActionError(error)),
        'error'
      )
    }
    return false
  }
}

export interface ContinuationJobOutcome {
  completed: number
  failed: number
  skipped: number
  total: number
  firstError: string
}

export function continuationJobOutcome(job: V2JobDetail): ContinuationJobOutcome {
  const rawError = job.failedItems[0]?.error
  const firstError = rawError && typeof rawError === 'object'
    && typeof rawError.message === 'string'
    ? rawError.message
    : ''
  return {
    completed: job.counts.completed,
    failed: job.counts.failed + job.counts.cancelled,
    skipped: job.counts.skipped,
    total: job.counts.total,
    firstError,
  }
}

export function assertContinuationJobCompleted(
  job: V2JobDetail,
  action: string,
): ContinuationJobOutcome {
  const outcome = continuationJobOutcome(job)
  if (
    job.status !== 'completed'
    || outcome.failed > 0
    || outcome.skipped > 0
    || outcome.completed !== outcome.total
  ) {
    const counts = `成功 ${outcome.completed}，失败 ${outcome.failed}，跳过 ${outcome.skipped}`
    throw new Error(
      `${action}未完整完成（${counts}）${outcome.firstError ? `：${outcome.firstError}` : ''}`
    )
  }
  return outcome
}
