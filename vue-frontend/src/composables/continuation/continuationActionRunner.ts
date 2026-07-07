import type { ContinuationState } from './useContinuationState'

type ContinuationMutationResult = {
  success?: boolean
  error?: string
}

interface ContinuationMutationOptions<T extends ContinuationMutationResult> {
  state: ContinuationState
  failurePrefix: string
  successMessage?: string
  run: () => Promise<T>
  afterSuccess?: (result: T) => Promise<void> | void
  onFailure?: () => Promise<void> | void
}

export function toContinuationActionError(error: unknown): string {
  return error instanceof Error ? error.message : '网络错误'
}

function formatContinuationActionError(prefix: string, message?: string): string {
  return `${prefix}: ${message || '操作失败'}`
}

export async function runContinuationMutation<T extends ContinuationMutationResult>(
  options: ContinuationMutationOptions<T>
): Promise<T | undefined> {
  try {
    const result = await options.run()

    if (result.success) {
      await options.afterSuccess?.(result)
      if (options.successMessage) {
        options.state.showMessage(options.successMessage, 'success')
      }
      return result
    }

    await options.onFailure?.()
    options.state.showMessage(
      formatContinuationActionError(options.failurePrefix, result.error),
      'error'
    )
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
