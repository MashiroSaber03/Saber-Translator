import { afterEach, expect, it, vi } from 'vitest'
import {
  configureBrowserCredentials,
  prepareBrowserCredentialTransaction,
} from '@/services/browserCredentials'

const { upload } = vi.hoisted(() => ({ upload: vi.fn().mockResolvedValue({}) }))
vi.mock('@/api/client', () => ({ apiClient: { put: upload } }))

afterEach(() => {
  configureBrowserCredentials(false)
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

function storageTransaction() {
  const request = {
    result: 'saved-key',
    error: null as Error | null,
    onsuccess: null as (() => void) | null,
    onerror: null as (() => void) | null,
  }
  const put = vi.fn(() => request)
  const transaction = {
    objectStore: () => ({ put }),
    error: null as Error | null,
    oncomplete: null as (() => void) | null,
    onabort: null as (() => void) | null,
    onerror: null as (() => void) | null,
  }
  const close = vi.fn()
  vi.stubGlobal('indexedDB', {
    open: () => {
      const opened = {
        result: { transaction: () => transaction, close },
        onsuccess: null as (() => void) | null,
      }
      queueMicrotask(() => opened.onsuccess?.())
      return opened
    },
  })
  configureBrowserCredentials(true, 'test-user')
  const pending = prepareBrowserCredentialTransaction({
    credentialEdits: [{
      domain: 'translation',
      provider: 'custom',
      secret: { apiKey: 'test-key' },
      baseRevision: 0,
      clientRef: 'test-credential',
    }],
  })
  return { transaction, request, put, close, pending }
}

it('uploads the lease only after the local credential transaction commits', async () => {
  const { transaction, request, put, close, pending } = storageTransaction()
  await vi.waitFor(() => expect(put).toHaveBeenCalledOnce())
  request.onsuccess?.()
  await Promise.resolve()
  expect(upload).not.toHaveBeenCalled()
  expect(close).not.toHaveBeenCalled()
  transaction.oncomplete?.()
  await pending
  expect(upload).toHaveBeenCalledOnce()
  expect(close).toHaveBeenCalledOnce()
})

it('does not upload a lease when the local transaction aborts after request success', async () => {
  const { transaction, request, put, close, pending } = storageTransaction()
  const rejected = expect(pending).rejects.toThrow('commit failed')
  await vi.waitFor(() => expect(put).toHaveBeenCalledOnce())
  request.onsuccess?.()
  transaction.error = new Error('commit failed')
  transaction.onabort?.()
  await rejected
  expect(upload).not.toHaveBeenCalled()
  expect(close).toHaveBeenCalledOnce()
})
