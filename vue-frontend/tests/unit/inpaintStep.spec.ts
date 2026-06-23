import { beforeEach, describe, expect, it, vi } from 'vitest'

const { parallelInpaintMock } = vi.hoisted(() => ({
  parallelInpaintMock: vi.fn(),
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelInpaint: parallelInpaintMock,
}))

import { executeInpaint } from '@/composables/translation/core/steps/inpaint'

const settingsSnapshot = {
  textStyle: {
    inpaintMethod: 'solid',
    fillColor: '#ffffff',
  },
  preciseMask: {
    dilateSize: 3,
    boxExpandRatio: 0.1,
  },
} as any

describe('executeInpaint', () => {
  beforeEach(() => {
    parallelInpaintMock.mockReset()
    parallelInpaintMock.mockResolvedValue({
      success: true,
      clean_image: 'clean-image',
    })
  })

  it('sends text and user masks without routine console output', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    let result
    try {
      result = await executeInpaint({
        imageIndex: 0,
        image: {
          originalDataURL: 'data:image/png;base64,original',
        } as any,
        bubbleCoords: [[0, 0, 10, 10]],
        bubblePolygons: [[]],
        textMask: 'text-mask',
        userMask: 'user-mask',
        settingsSnapshot,
      })
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(result).toEqual({ cleanImage: 'clean-image' })
    expect(parallelInpaintMock).toHaveBeenCalledWith(expect.objectContaining({
      raw_mask: 'text-mask',
      user_mask: 'user-mask',
      method: 'solid',
    }))
  })
})
