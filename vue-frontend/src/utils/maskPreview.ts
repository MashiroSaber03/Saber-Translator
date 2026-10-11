/** Display-only layers. Repair always submits the backend's final binary mask. */
export const maskLayerOptions = [
  { value: 'raw', label: '原始文字范围', text: '原始', operator: '', color: [180, 191, 209] },
  { value: 'boxes', label: '文本框范围', text: '文本框范围', operator: '→', color: [250, 204, 21] },
  { value: 'automatic', label: '自动生效区域', text: '自动', operator: '→', color: [251, 146, 60] },
  { value: 'repair', label: '手动修复区域', text: '手动修复', operator: '+', color: [76, 175, 80] },
  { value: 'preserve', label: '手动保留区域', text: '手动保留', operator: '−', color: [33, 150, 243] },
  { value: 'final', label: '最终修复范围', text: '最终', operator: '=', color: [244, 67, 54] },
] as const

export type MaskLayer = (typeof maskLayerOptions)[number]['value']
export type MaskVisibility = Record<MaskLayer, boolean>
export type MaskStages = Record<'raw' | 'boxes' | 'automatic', Uint8Array | null>

export function maskVisibility(): MaskVisibility {
  return {
    raw: false, boxes: false, automatic: false,
    repair: false, preserve: false, final: true,
  }
}

export function maskOverlayPixels(
  width: number, height: number, stages: MaskStages, manual: Uint8Array, visibility: MaskVisibility
): Uint8ClampedArray {
  // Read reactive switches once, instead of through their proxy for every pixel.
  const visible = { ...visibility }
  const pixels = new Uint8ClampedArray(width * height * 4)
  const colors = Object.fromEntries(maskLayerOptions.map(item => [item.value, item.color])) as
    Record<MaskLayer, (typeof maskLayerOptions)[number]['color']>
  if (!maskLayerOptions.some(item => visible[item.value] &&
    (item.value !== 'raw' || stages.raw))) return pixels
  const finalAt = (index: number) => manual[index] === 255 ||
    (manual[index] === 127 && Boolean(stages.automatic?.[index]))
  // Raster outlines use the exact backend ranges, including expanded/rotated boxes.
  function edge(index: number, x: number, y: number, selected: (i: number) => boolean, radius: number): boolean {
    for (let distance = 1; distance <= radius; distance++) {
      if (x < distance || y < distance || x >= width - distance || y >= height - distance ||
          !selected(index - distance) || !selected(index + distance) ||
          !selected(index - width * distance) || !selected(index + width * distance)) return true
    }
    return false
  }
  const inBoxes = (index: number) => Boolean(stages.boxes?.[index])
  for (let i = 0; i < manual.length; i++) {
    const x = i % width, y = Math.floor(i / width)
    let color: readonly number[] | null = null
    let alpha = 110
    if (visible.raw && (stages.raw?.[i] ?? 0) >= 128) {
      color = colors.raw
      alpha = (x + y) % 12 < 4 ? 160 : 45
    }
    if (visible.boxes && inBoxes(i) && !color) { color = colors.boxes; alpha = 28 }
    if (visible.automatic && stages.automatic?.[i]) { color = colors.automatic; alpha = 110 }
    if (visible.repair && manual[i] === 255) { color = colors.repair; alpha = 140 }
    if (visible.preserve && manual[i] === 0) {
      color = colors.preserve
      alpha = (x + y) % 12 < 4 ? 200 : 85
    }
    if (visible.final && finalAt(i)) {
      color = colors.final; alpha = 110
    }
    if (visible.boxes && inBoxes(i) && edge(i, x, y, inBoxes, 3) && (x + y) % 12 < 8) {
      color = colors.boxes; alpha = 240
    }
    if (color) {
      pixels[i * 4] = color[0]!
      pixels[i * 4 + 1] = color[1]!
      pixels[i * 4 + 2] = color[2]!
      pixels[i * 4 + 3] = alpha
    }
  }
  return pixels
}
