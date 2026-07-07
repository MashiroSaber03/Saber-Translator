import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { ImageData } from '@/types/image'

const { parallelDetectMock } = vi.hoisted(() => ({
  parallelDetectMock: vi.fn()
}))

const detectionSettingsSnapshot = createDefaultSettings()
detectionSettingsSnapshot.textDetector = 'ctd'
detectionSettingsSnapshot.minTextBlockAreaPercent = 1
detectionSettingsSnapshot.enableSaberYoloRefine = true
detectionSettingsSnapshot.saberYoloRefineOverlapThreshold = 35
detectionSettingsSnapshot.enableAuxYoloDetection = true
detectionSettingsSnapshot.auxYoloConfThreshold = 0.55
detectionSettingsSnapshot.auxYoloOverlapThreshold = 0.2
detectionSettingsSnapshot.boxExpand = {
  ratio: 3,
  top: 1,
  bottom: 2,
  left: 4,
  right: 5
}
detectionSettingsSnapshot.textStyle = {
  ...detectionSettingsSnapshot.textStyle,
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

vi.mock('@/api/parallelTranslate', () => ({
  parallelDetect: parallelDetectMock
}))

function createTestImage(overrides: Partial<ImageData> = {}): ImageData {
  return {
    id: 'image-1',
    fileName: 'page.png',
    originalDataURL: 'data:image/png;base64,ZmFrZQ==',
    translatedDataURL: null,
    cleanImageData: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    fontSize: 18,
    autoFontSize: false,
    fontFamily: 'fonts/STSONG.TTF',
    layoutDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#ffffff',
    inpaintMethod: 'solid',
    strokeEnabled: false,
    strokeColor: '#000000',
    strokeWidth: 1,
    lineSpacing: 1,
    textAlign: 'start',
    useAutoTextColor: false,
    hasUnsavedChanges: false,
    ...overrides,
  }
}

describe('executeDetection saber yolo refine flags', () => {
  beforeEach(() => {
    parallelDetectMock.mockReset()
  })

  it('builds detection bubble states through the shared bubble factory', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/steps/detection.ts'),
      'utf8',
    )

    expect(source).toContain('createBubbleStatesFromResponse')
    expect(source).not.toContain('function createBubbleStates' + 'FromDetection')
    expect(source).not.toContain('createBubble' + 'State({')
  })

  it('keeps detection fixtures typed to the current image and settings schema', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'tests/unit/detectionStepSaberRefine.spec.ts'),
      'utf8',
    )

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
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

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      await executeDetection({
        imageIndex: 0,
        image: createTestImage({ bubbleStates: null }),
        settingsSnapshot: detectionSettingsSnapshot,
      })
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

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

  it('keeps existing and cleared bubble detection skips quiet', async () => {
    const { executeDetection } = await import('@/composables/translation/core/steps/detection')

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      await executeDetection({
        imageIndex: 0,
        image: createTestImage({
          bubbleStates: [
            createBubbleState({
              coords: [0, 0, 10, 10],
              rotationAngle: 0,
              autoTextDirection: 'vertical',
              textDirection: 'vertical',
              originalText: '原文',
              textlines: [],
            })
          ]
        }),
        settingsSnapshot: detectionSettingsSnapshot,
      })

      await executeDetection({
        imageIndex: 1,
        image: createTestImage({ bubbleStates: [] }),
        settingsSnapshot: detectionSettingsSnapshot,
      })

      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })
})
