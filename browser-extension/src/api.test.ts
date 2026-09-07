import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { saberRequest } from './api'

beforeEach(() => {
  vi.stubGlobal('chrome', {
    storage: {
      local: {
        get: async () => ({
          'saber-extension-settings-v1': { token: 'test-token', serverPort: 5000, domains: {} },
        }),
      },
    },
  })
})
afterEach(() => vi.unstubAllGlobals())

it('reports a useful connection error for both page and management requests', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
  for (const path of ['/status', '/manage/jobs?scope=all']) {
    await expect(saberRequest(path)).rejects.toMatchObject({
      code: 'saber_unreachable',
      message: '无法连接本机 Saber（端口 5000）',
    })
  }
})

it('keeps an HTTP error when an intermediary returns HTML instead of JSON', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(new Response('<h1>Unavailable</h1>', { status: 503 }))
  )
  await expect(saberRequest('/manage/settings')).rejects.toMatchObject({
    code: 'http_503',
    retryable: true,
  })
})

it('explains conflicting edits and keeps API credentials out of cookies', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValue(
      Response.json(
        { error: { code: 'revision_conflict', message: 'revision mismatch' } },
        { status: 409 }
      )
    )
  vi.stubGlobal('fetch', fetch)
  await expect(
    saberRequest('/manage/settings/transactions', { method: 'PUT', body: '{}' })
  ).rejects.toMatchObject({
    code: 'revision_conflict',
    message: '配置已在其他窗口变化，请重新读取后再修改。',
  })
  expect(fetch.mock.calls[0]![1]).toMatchObject({ credentials: 'omit' })
})
