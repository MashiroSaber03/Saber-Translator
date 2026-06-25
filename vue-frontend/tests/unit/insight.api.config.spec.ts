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

import { getGlobalConfig, saveGlobalConfig, type AnalysisConfig } from '@/api/insight'

describe('insight api config routes', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('getGlobalConfig calls the global config route', async () => {
    getMock.mockResolvedValue({ success: true, config: {} })

    await getGlobalConfig()

    expect(getMock).toHaveBeenCalledWith('/api/manga-insight/config')
  })

  it('saveGlobalConfig calls the global config route', async () => {
    const config: AnalysisConfig = { vlm: { provider: 'gemini', api_key: '', model: 'x' } }
    postMock.mockResolvedValue({ success: true })

    await saveGlobalConfig(config)

    expect(postMock).toHaveBeenCalledWith('/api/manga-insight/config', config)
  })
})
