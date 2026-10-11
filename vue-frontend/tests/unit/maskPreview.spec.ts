import { describe, expect, it } from 'vitest'
import { reactive } from 'vue'
import { maskOverlayPixels, maskVisibility, type MaskStages } from '@/utils/maskPreview'

const size = 40
function region(x1: number, y1: number, x2: number, y2: number): Uint8Array {
  const mask = new Uint8Array(size * size)
  for (let y = y1; y < y2; y++) for (let x = x1; x < x2; x++) mask[y * size + x] = 255
  return mask
}
const at = (pixels: Uint8ClampedArray, x: number, y: number) =>
  Array.from(pixels.slice((y * size + x) * 4, (y * size + x) * 4 + 4))
const processVisibility = { ...maskVisibility(), automatic: true, repair: true, preserve: true }
const stages: MaskStages = {
  raw: region(4, 4, 32, 32), boxes: region(8, 8, 28, 28),
  automatic: region(7, 7, 28, 28),
}

describe('mask display composition', () => {
  it('shows only yellow text-box ranges, without implicitly displaying text masks', () => {
    const manual = new Uint8Array(size * size).fill(127)
    const visible = { ...maskVisibility(), final: false, boxes: true }
    const pixels = maskOverlayPixels(size, size, stages, manual, visible)
    expect(at(pixels, 22, 22)).toEqual([250, 204, 21, 28])
    expect(at(pixels, 16, 16)).toEqual([250, 204, 21, 28])
    expect(at(pixels, 8, 18)).toEqual([250, 204, 21, 240])
    expect(at(pixels, 5, 5)[3]).toBe(0)
  })

  it('hides the range without changing any mask source data', () => {
    const manual = new Uint8Array(size * size).fill(127)
    const visible = { ...maskVisibility(), final: false, boxes: true }
    const before = Object.fromEntries(Object.entries(stages).map(([key, value]) => [key, value?.slice()]))
    const pixels = maskOverlayPixels(size, size, stages, manual, visible)
    expect(at(pixels, 16, 16)).toEqual([250, 204, 21, 28])
    expect(at(pixels, 5, 5)[3]).toBe(0)
    expect(at(pixels, 35, 35)[3]).toBe(0)
    expect(stages).toEqual(before)
    expect(manual.every(value => value === 127)).toBe(true)
    visible.boxes = false
    expect(maskOverlayPixels(size, size, stages, manual, visible).some(Boolean)).toBe(false)
  })

  it('reads reactive switches once per redraw and uses their current values', () => {
    const switches = reactive({ ...processVisibility })
    const reads = new Map<string, number>()
    const observed = new Proxy(switches, {
      get(target, key, receiver) {
        if (typeof key === 'string' && Object.hasOwn(target, key)) {
          reads.set(key, (reads.get(key) ?? 0) + 1)
        }
        return Reflect.get(target, key, receiver)
      },
    })
    const manual = new Uint8Array(size * size).fill(127)
    expect(maskOverlayPixels(size, size, stages, manual, observed))
      .toEqual(maskOverlayPixels(size, size, stages, manual, { ...switches }))
    expect([...reads.values()]).toEqual(Array(6).fill(1))

    for (const key of Object.keys(switches) as (keyof typeof switches)[]) switches[key] = false
    reads.clear()
    expect(maskOverlayPixels(size, size, stages, manual, observed).some(Boolean)).toBe(false)
    expect([...reads.values()]).toEqual(Array(6).fill(1))
  })

  it('shows final coverage while preserving the original manual and automatic data', () => {
    const manual = new Uint8Array(size * size).fill(127)
    manual[16 * size + 16] = 0
    manual[35 * size + 35] = 255
    const before = manual.slice()
    const pixels = maskOverlayPixels(size, size, stages, manual, maskVisibility())
    expect(at(pixels, 16, 16)[3]).toBe(0)
    expect(at(pixels, 20, 20)).toEqual([244, 67, 54, 110])
    expect(at(pixels, 35, 35)).toEqual([244, 67, 54, 110])
    expect(manual).toEqual(before)
  })
  it.each(['raw', 'boxes', 'automatic', 'repair', 'preserve'] as const)(
    'keeps the final translucent fill unchanged when %s is shown', layer => {
      const manual = new Uint8Array(size * size).fill(127)
      manual[35 * size + 35] = 255
      const visible = maskVisibility()
      const original = maskOverlayPixels(size, size, stages, manual, visible)
      visible[layer] = true
      const shown = maskOverlayPixels(size, size, stages, manual, visible)
      expect(at(shown, 20, 20)).toEqual([244, 67, 54, 110])
      expect(at(shown, 35, 35)).toEqual([244, 67, 54, 110])
      visible[layer] = false
      expect(maskOverlayPixels(size, size, stages, manual, visible)).toEqual(original)
    },
  )
  it.each(['raw', 'automatic', 'repair', 'preserve', 'final'] as const)(
    'draws text-box boundaries above the %s layer', layer => {
      const manual = new Uint8Array(size * size).fill(layer === 'repair' ? 255 : layer === 'preserve' ? 0 : 127)
      const visible = { ...maskVisibility(), final: false, boxes: true, [layer]: true }
      const pixels = maskOverlayPixels(size, size, stages, manual, visible)
      expect(at(pixels, 8, 18)).toEqual([250, 204, 21, 240])
      expect(at(pixels, 8, 12)).not.toEqual([250, 204, 21, 240]) // Dashed gap shows the mask underneath.
      visible.boxes = false
      expect(at(maskOverlayPixels(size, size, stages, manual, visible), 8, 18))
        .not.toEqual([250, 204, 21, 240])
    },
  )
  it('distinguishes automatic fill and manual overrides when final coverage is hidden', () => {
    const manual = new Uint8Array(size * size).fill(127)
    for (let y = 12; y < 22; y++) for (let x = 12; x < 22; x++) manual[y * size + x] = 0
    const visible = { ...processVisibility, final: false }
    const pixels = maskOverlayPixels(size, size, stages, manual, visible)
    expect(at(pixels, 17, 17).slice(0, 3)).toEqual([33, 150, 243])
    expect(at(pixels, 17, 24).slice(0, 3)).toEqual([251, 146, 60])
    expect(at(pixels, 7, 17).slice(0, 3)).toEqual([251, 146, 60])
    manual.fill(255, 12 * size, 22 * size)
    expect(at(maskOverlayPixels(size, size, stages, manual, visible), 17, 17).slice(0, 3))
      .toEqual([76, 175, 80])
  })
  it('shows original text only when explicitly enabled alongside the box boundaries', () => {
    const visible = { ...maskVisibility(), final: false, raw: true, boxes: true }
    const pixels = maskOverlayPixels(size, size, stages, new Uint8Array(size * size).fill(127), visible)
    expect(at(pixels, 5, 5).slice(0, 3)).toEqual([180, 191, 209])
    expect(at(pixels, 16, 16).slice(0, 3)).toEqual([180, 191, 209])
    expect(at(pixels, 8, 18).slice(0, 3)).toEqual([250, 204, 21])
    for (const key of Object.keys(visible) as (keyof typeof visible)[]) visible[key] = false
    expect(maskOverlayPixels(size, size, stages, new Uint8Array(size * size).fill(127), visible).some(Boolean)).toBe(false)
  })
  it('keeps tiny protected holes visible within the final fill alongside other layers', () => {
    const manual = new Uint8Array(size * size).fill(127)
    manual[16 * size + 16] = 0
    const visible = { ...processVisibility, boxes: true }
    const pixels = maskOverlayPixels(size, size, stages, manual, visible)
    expect(at(pixels, 16, 16).slice(0, 3)).toEqual([33, 150, 243])
    expect(at(pixels, 16, 15).slice(0, 3)).toEqual([244, 67, 54])
    manual[10 * size + 12] = 0
    expect(at(maskOverlayPixels(size, size, stages, manual, visible), 12, 10).slice(0, 3))
      .toEqual([33, 150, 243])
  })
})
