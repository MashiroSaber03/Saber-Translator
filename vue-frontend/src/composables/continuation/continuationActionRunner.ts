import type { ContinuationState } from './useContinuationState'

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
