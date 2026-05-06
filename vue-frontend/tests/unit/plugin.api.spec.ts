import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, deleteMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  deleteMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    delete: deleteMock,
  },
}))

import { refreshPlugins } from '@/api/plugin'

describe('plugin api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    deleteMock.mockReset()
  })

  it('refreshPlugins should post to the refresh route', async () => {
    postMock.mockResolvedValue({ success: true, partial_success: false })

    await refreshPlugins()

    expect(postMock).toHaveBeenCalledWith('/api/plugins/refresh')
  })
})
