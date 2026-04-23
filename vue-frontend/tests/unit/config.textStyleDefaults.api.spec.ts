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

import {
  getTextStyleDefaults,
  resetTextStyleDefaults,
  saveTextStyleDefaults,
} from '@/api/config'

describe('config text style defaults api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('getTextStyleDefaults should call the text defaults route', async () => {
    getMock.mockResolvedValue({ success: true, defaults: {} })

    await getTextStyleDefaults()

    expect(getMock).toHaveBeenCalledWith('/api/config/text-style-defaults')
  })

  it('saveTextStyleDefaults should post defaults payload', async () => {
    const defaults = { fontSize: 26, textColor: '#000000' }
    postMock.mockResolvedValue({ success: true, defaults })

    await saveTextStyleDefaults(defaults as any)

    expect(postMock).toHaveBeenCalledWith('/api/config/text-style-defaults', { defaults })
  })

  it('resetTextStyleDefaults should call reset route', async () => {
    postMock.mockResolvedValue({ success: true, defaults: {} })

    await resetTextStyleDefaults()

    expect(postMock).toHaveBeenCalledWith('/api/config/text-style-defaults/reset')
  })
})
