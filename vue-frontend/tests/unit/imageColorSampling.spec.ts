import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { sampleImageColor } from '@/utils/imageColorSampling'

const drawImage = vi.fn()
const getImageData = vi.fn()
const context = { drawImage, getImageData, imageSmoothingEnabled: true }

function imageAtScale(scale: number) {
  const image = document.createElement('img')
  Object.defineProperties(image, {
    complete: { value: true, configurable: true },
    naturalWidth: { value: 400 },
    naturalHeight: { value: 300 },
  })
  vi.spyOn(image, 'getBoundingClientRect').mockReturnValue({
    left: 120, top: 80, width: 400 * scale, height: 300 * scale,
  } as DOMRect)
  return image
}

describe('image color sampling', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getImageData.mockReturnValue({ data: new Uint8ClampedArray([18, 52, 86, 255]) })
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D)
  })
  afterEach(() => vi.restoreAllMocks())

  it.each([0.5, 1, 1.5])('reads one decoded pixel after zooming to %s and panning', scale => {
    const image = imageAtScale(scale)
    expect(sampleImageColor(image, { clientX: 120 + 130.5 * scale, clientY: 80 + 150.5 * scale })).toBe('#123456')
    expect(drawImage).toHaveBeenCalledWith(image, 130, 150, 1, 1, 0, 0, 1, 1)
    expect(context.imageSmoothingEnabled).toBe(false)
    expect(getImageData).toHaveBeenCalledWith(0, 0, 1, 1)
  })

  it.each([[119, 80], [120, 79], [520, 80], [120, 380], [NaN, 80]])('rejects coordinates outside the image (%s, %s)', (clientX, clientY) => {
    expect(sampleImageColor(imageAtScale(1), { clientX, clientY })).toBeNull()
    expect(drawImage).not.toHaveBeenCalled()
  })

  it('keeps the last valid pixel and rejects an image still loading', () => {
    const image = imageAtScale(1)
    sampleImageColor(image, { clientX: 519.9, clientY: 379.9 })
    expect(drawImage).toHaveBeenCalledWith(image, 399, 299, 1, 1, 0, 0, 1, 1)
    Object.defineProperty(image, 'complete', { value: false })
    expect(sampleImageColor(image, { clientX: 130, clientY: 90 })).toBeNull()
    expect(drawImage).toHaveBeenCalledTimes(1)
  })

  it('does not interpret transparent pixels as black', () => {
    getImageData.mockReturnValue({ data: new Uint8ClampedArray([0, 0, 0, 0]) })
    expect(sampleImageColor(imageAtScale(1), { clientX: 130, clientY: 90 })).toBeNull()
  })
})
