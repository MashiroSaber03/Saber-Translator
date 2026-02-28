import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
  },
}))

import { getAnalysisConfig, saveAnalysisConfig } from '@/api/insight'

describe('insight api config routes', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('getAnalysisConfig should call global config route', async () => {
    getMock.mockResolvedValue({ success: true, config: {} })

    await getAnalysisConfig('book-1')

    expect(getMock).toHaveBeenCalledWith('/api/manga-insight/config')
  })

  it('saveAnalysisConfig should call global config route', async () => {
    const config = { vlm: { provider: 'gemini', api_key: '', model: 'x' } }
    postMock.mockResolvedValue({ success: true })

    await saveAnalysisConfig('book-1', config as any)

    expect(postMock).toHaveBeenCalledWith('/api/manga-insight/config', config)
  })
})
