import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createDefaultSettings } from '@/stores/settings/defaults'
import type { ImageData } from '@/types/image'

const { parallelInpaintMock } = vi.hoisted(() => ({
  parallelInpaintMock: vi.fn(),
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelInpaint: parallelInpaintMock,
}))

import { executeInpaint } from '@/composables/translation/core/steps/inpaint'

const settingsSnapshot = createDefaultSettings()
settingsSnapshot.textStyle = {
  ...settingsSnapshot.textStyle,
  inpaintMethod: 'solid',
  fillColor: '#ffffff',
}
settingsSnapshot.preciseMask = {
  dilateSize: 3,
  boxExpandRatio: 0.1,
}

function createTestImage(overrides: Partial<ImageData> = {}): ImageData {
  return {
    id: 'image-1',
    fileName: 'page.png',
    originalDataURL: 'data:image/png;base64,original',
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

describe('executeInpaint', () => {
  beforeEach(() => {
    parallelInpaintMock.mockReset()
    parallelInpaintMock.mockResolvedValue({
      success: true,
      clean_image: 'clean-image',
    })
  })

  it('keeps inpaint fixtures typed to the current image and settings schema', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/inpaintStep.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('sends text and user masks without routine console output', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    let result
    try {
      result = await executeInpaint({
        imageIndex: 0,
        image: createTestImage(),
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
