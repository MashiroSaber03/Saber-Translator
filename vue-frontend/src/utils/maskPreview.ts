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
export type MaskOverlay = { canvas: HTMLCanvasElement }

export function maskVisibility(): MaskVisibility {
  return {
    raw: false, boxes: false, automatic: false,
    repair: false, preserve: false, final: true,
  }
}

// Compile geometry and manual overrides only when mask data changes. Display switches
// then use a tiny palette over one byte per pixel, without repeating edge searches.
export function maskOverlayClasses(
  width: number, height: number, stages: MaskStages, manual: Uint8Array
): Uint8Array {
  const classes = new Uint8Array(width * height)
  const inBoxes = (index: number) => Boolean(stages.boxes?.[index])
  function edge(index: number, x: number, y: number): boolean {
    for (let distance = 1; distance <= 3; distance++) {
      if (x < distance || y < distance || x >= width - distance || y >= height - distance ||
          !inBoxes(index - distance) || !inBoxes(index + distance) ||
          !inBoxes(index - width * distance) || !inBoxes(index + width * distance)) return true
    }
    return false
  }
  for (let i = 0; i < manual.length; i++) {
    const raw = (stages.raw?.[i] ?? 0) >= 128
    const box = inBoxes(i)
    const automatic = Boolean(stages.automatic?.[i])
    const repair = manual[i] === 255, preserve = manual[i] === 0
    let flags = (raw ? 1 : 0) | (box ? 2 : 0) | (automatic ? 4 : 0)
      | (repair ? 8 : 0) | (preserve ? 16 : 0)
      | (repair || (manual[i] === 127 && automatic) ? 32 : 0)
    if (raw || box || preserve) {
      const x = i % width, y = Math.floor(i / width), pattern = (x + y) % 12
      if (pattern < 4) flags |= 128
      if (box && pattern < 8 && edge(i, x, y)) flags |= 64
    }
    classes[i] = flags
  }
  return classes
}

export function maskOverlayPixels(
  classes: Uint8Array, visibility: MaskVisibility,
  pixels = new Uint8ClampedArray(classes.length * 4)
): Uint8ClampedArray {
  const visible = { ...visibility }
  const colors = Object.fromEntries(maskLayerOptions.map(item => [item.value, item.color])) as
    Record<MaskLayer, (typeof maskLayerOptions)[number]['color']>
  const palette = new Uint8ClampedArray(256 * 4)
  for (let flags = 0; flags < 256; flags++) {
    let color: readonly number[] | null = null
    let alpha = 110
    if (visible.raw && (flags & 1)) { color = colors.raw; alpha = flags & 128 ? 160 : 45 }
    if (visible.boxes && (flags & 2) && !color) { color = colors.boxes; alpha = 28 }
    if (visible.automatic && (flags & 4)) { color = colors.automatic; alpha = 110 }
    if (visible.repair && (flags & 8)) { color = colors.repair; alpha = 140 }
    if (visible.preserve && (flags & 16)) { color = colors.preserve; alpha = flags & 128 ? 200 : 85 }
    if (visible.final && (flags & 32)) { color = colors.final; alpha = 110 }
    if (visible.boxes && (flags & 64)) { color = colors.boxes; alpha = 240 }
    if (color) {
      palette[flags * 4] = color[0]!
      palette[flags * 4 + 1] = color[1]!
      palette[flags * 4 + 2] = color[2]!
      palette[flags * 4 + 3] = alpha
    }
  }
  // Read and write the RGBA bytes in native order; this does not assume endianness.
  const packed = new Uint32Array(palette.buffer)
  const target = new Uint32Array(pixels.buffer, pixels.byteOffset, classes.length)
  for (let i = 0; i < classes.length; i++) target[i] = packed[classes[i]!]!
  return pixels
}
