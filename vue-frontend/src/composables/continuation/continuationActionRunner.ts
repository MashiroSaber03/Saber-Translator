import type { ContinuationState } from './useContinuationState'

interface ContinuationMutationOptions<T> {
  state: ContinuationState
  failurePrefix: string
  successMessage?: string
  run: () => Promise<T>
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
): Promise<T | undefined> {
  try {
    const result = await options.run()

    await options.afterSuccess?.(result)
    if (options.successMessage) {
      options.state.showMessage(options.successMessage, 'success')
    }
    return result
  } catch (error) {
    await options.onFailure?.()
    options.state.showMessage(
      formatContinuationActionError(options.failurePrefix, toContinuationActionError(error)),
      'error'
    )
    return undefined
  }
}
