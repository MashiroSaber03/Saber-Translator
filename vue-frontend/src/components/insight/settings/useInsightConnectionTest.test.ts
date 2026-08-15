import { nextTick, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import { useInsightConnectionTest } from './useInsightConnectionTest'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe('useInsightConnectionTest', () => {
  it('suppresses a result after the tested settings change', async () => {
    const provider = ref('first')
    const pending = deferred<{ success: boolean }>()
    const request = vi.fn(() => pending.promise)
    const emitMessage = vi.fn()
    const connection = useInsightConnectionTest({
      sources: [provider],
      snapshot: () => ({ provider: provider.value }),
      request,
      successMessage: '连接成功',
      emitMessage,
    })

    const operation = connection.testConnection()
    await connection.testConnection()
    expect(request).toHaveBeenCalledTimes(1)

    provider.value = 'second'
    await nextTick()
    pending.resolve({ success: true })
    await operation

    expect(connection.isTesting.value).toBe(false)
    expect(emitMessage).not.toHaveBeenCalled()
  })

  it('reports the actual request failure detail', async () => {
    const provider = ref('current')
    const emitMessage = vi.fn()
    const connection = useInsightConnectionTest({
      sources: [provider],
      snapshot: () => ({ provider: provider.value }),
      request: () => Promise.reject(new Error('provider unavailable')),
      successMessage: '连接成功',
      emitMessage,
    })

    await connection.testConnection()

    expect(emitMessage).toHaveBeenCalledWith(
      '测试失败: provider unavailable',
      'error',
    )
    expect(connection.isTesting.value).toBe(false)
  })
})
