import { describe, expect, it, vi, beforeEach } from 'vitest'

const { parallelDetectMock } = vi.hoisted(() => ({
  parallelDetectMock: vi.fn()
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelDetect: parallelDetectMock
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => ({
    settings: {
      textDetector: 'ctd',
      enableSaberYoloRefine: true,
      saberYoloRefineOverlapThreshold: 35,
      boxExpand: {
        ratio: 3,
        top: 1,
        bottom: 2,
        left: 4,
        right: 5
      }
    }
  })
}))

vi.mock('@/stores/imageStore', () => ({
  useImageStore: () => ({
    updateImageByIndex: vi.fn()
  })
}))

describe('executeDetection saber yolo refine flags', () => {
  beforeEach(() => {
    parallelDetectMock.mockReset()
  })

  it('passes the current toggle for main detection and disables refinement for mask detection', async () => {
    parallelDetectMock
      .mockResolvedValueOnce({
        success: true,
        bubble_coords: [],
        bubble_angles: [],
        bubble_polygons: [],
        auto_directions: [],
        textlines_per_bubble: []
      })
      .mockResolvedValueOnce({
        success: true,
        raw_mask: 'mask-data'
      })

    const { executeDetection } = await import('@/composables/translation/core/steps/detection')

    await executeDetection({
      imageIndex: 0,
      image: {
        originalDataURL: 'data:image/png;base64,ZmFrZQ==',
        bubbleStates: undefined
      } as any
    })

    expect(parallelDetectMock).toHaveBeenCalledTimes(2)
    expect(parallelDetectMock).toHaveBeenNthCalledWith(1, expect.objectContaining({
      detector_type: 'ctd',
      enable_saber_yolo_refine: true,
      saber_yolo_refine_overlap_threshold: 35
    }))
    expect(parallelDetectMock).toHaveBeenNthCalledWith(2, expect.objectContaining({
      detector_type: 'default',
      enable_saber_yolo_refine: false
    }))
  })
})
