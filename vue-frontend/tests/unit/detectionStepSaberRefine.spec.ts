import { describe, expect, it, vi, beforeEach } from 'vitest'

const { parallelDetectMock } = vi.hoisted(() => ({
  parallelDetectMock: vi.fn()
}))

const detectionSettingsSnapshot = {
  textDetector: 'ctd',
  minTextBlockAreaPercent: 1,
  enableSaberYoloRefine: true,
  saberYoloRefineOverlapThreshold: 35,
  enableAuxYoloDetection: true,
  auxYoloConfThreshold: 0.55,
  auxYoloOverlapThreshold: 0.2,
  boxExpand: {
    ratio: 3,
    top: 1,
    bottom: 2,
    left: 4,
    right: 5
  },
  textStyle: {
    fontSize: 16,
    fontFamily: 'fonts/STSONG.TTF',
    layoutDirection: 'auto',
    textColor: '#000000',
    fillColor: '#ffffff',
    strokeEnabled: false,
    strokeColor: '#000000',
    strokeWidth: 1,
    lineSpacing: 1,
    textAlign: 'start',
    inpaintMethod: 'solid',
  }
} as any

vi.mock('@/api/parallelTranslate', () => ({
  parallelDetect: parallelDetectMock
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => ({
    settings: detectionSettingsSnapshot
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
      } as any,
      settingsSnapshot: detectionSettingsSnapshot,
    })

    expect(parallelDetectMock).toHaveBeenCalledTimes(2)
    expect(parallelDetectMock).toHaveBeenNthCalledWith(1, expect.objectContaining({
      detector_type: 'ctd',
      min_text_block_area_percent: 1,
      enable_saber_yolo_refine: true,
      saber_yolo_refine_overlap_threshold: 35,
      enable_aux_yolo_detection: true,
      aux_yolo_conf_threshold: 0.55,
      aux_yolo_overlap_threshold: 0.2
    }))
    expect(parallelDetectMock).toHaveBeenNthCalledWith(2, expect.objectContaining({
      detector_type: 'default',
      enable_saber_yolo_refine: false,
      enable_aux_yolo_detection: false
    }))
    expect(parallelDetectMock.mock.calls[1]?.[0]).not.toHaveProperty('min_text_block_area_percent')
  })
})
