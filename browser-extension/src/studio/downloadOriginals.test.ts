// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest'
import { downloadOriginals } from './downloadOriginals'
import { RequestFailure } from '../api'
const request = vi.hoisted(() => vi.fn())
vi.mock('../api', async importOriginal => ({ ...await importOriginal<typeof import('../api')>(), saberResponse: request }))
afterEach(() => { request.mockReset(); vi.restoreAllMocks() })
it.each(['http_404', 'not_found'])('explains an old backend instead of silently failing (%s)', async code => {
  request.mockRejectedValue(new RequestFailure(code, 'not found', false))
  await expect(downloadOriginals('session', 'comic')).rejects.toThrow('停止并重新启动后端')
})
it('keeps a missing session error distinct from a missing route', async () => {
  const error = new RequestFailure('not_found', 'browser session not found', false)
  request.mockRejectedValue(error)
  await expect(downloadOriginals('session', 'comic')).rejects.toBe(error)
})
