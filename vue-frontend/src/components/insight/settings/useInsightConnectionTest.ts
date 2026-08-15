import { getCurrentScope, onScopeDispose, ref, watch, type WatchSource } from 'vue'

import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'

type ConnectionTestResult = {
  success: boolean
  message?: string
}

type ConnectionTestOptions<TSnapshot> = {
  sources: WatchSource<unknown>[]
  snapshot: () => TSnapshot
  request: (snapshot: TSnapshot) => Promise<ConnectionTestResult>
  successMessage: string
  emitMessage: (message: string, type: 'success' | 'error') => void
}

export function useInsightConnectionTest<TSnapshot>(
  options: ConnectionTestOptions<TSnapshot>,
) {
  const isTesting = ref(false)
  const requestGuard = useLatestRequestGuard()

  function snapshotKey(snapshot: TSnapshot): string {
    return JSON.stringify(snapshot)
  }

  function invalidate(): void {
    requestGuard.invalidate()
    isTesting.value = false
  }

  async function testConnection(): Promise<void> {
    if (isTesting.value) return
    const snapshot = options.snapshot()
    const key = snapshotKey(snapshot)
    const requestId = requestGuard.next()
    isTesting.value = true

    const isCurrent = () => requestGuard.isCurrent(
      requestId,
      () => snapshotKey(options.snapshot()) === key,
    )

    try {
      const response = await options.request(snapshot)
      if (!isCurrent()) return
      options.emitMessage(
        response.success
          ? options.successMessage
          : '连接失败: ' + (response.message || '未知错误'),
        response.success ? 'success' : 'error',
      )
    } catch (error) {
      if (!isCurrent()) return
      options.emitMessage(
        '测试失败: ' + (error instanceof Error ? error.message : '网络错误'),
        'error',
      )
    } finally {
      if (requestGuard.isCurrent(requestId)) isTesting.value = false
    }
  }

  watch(options.sources, invalidate)
  if (getCurrentScope()) onScopeDispose(invalidate)

  return {
    isTesting,
    testConnection,
  }
}
