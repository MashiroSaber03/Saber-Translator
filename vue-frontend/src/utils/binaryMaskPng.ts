export interface BinaryMaskCircle {
  x: number
  y: number
  radius: number
}

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])
const IHDR = new Uint8Array([73, 72, 68, 82])
const IDAT = new Uint8Array([73, 68, 65, 84])
const IEND = new Uint8Array([73, 69, 78, 68])

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let value = 0; value < table.length; value += 1) {
    let remainder = value
    for (let bit = 0; bit < 8; bit += 1) {
      remainder = (remainder & 1) !== 0
        ? 0xedb88320 ^ (remainder >>> 1)
        : remainder >>> 1
    }
    table[value] = remainder >>> 0
  }
  return table
})()

function writeUint32(target: Uint8Array, offset: number, value: number): void {
  new DataView(target.buffer, target.byteOffset, target.byteLength)
    .setUint32(offset, value >>> 0)
}

function crc32(type: Uint8Array, payload: Uint8Array): number {
  let value = 0xffffffff
  for (const bytes of [type, payload]) {
    for (const byte of bytes) {
      value = CRC32_TABLE[(value ^ byte) & 0xff]! ^ (value >>> 8)
    }
  }
  return (value ^ 0xffffffff) >>> 0
}

function pngChunk(type: Uint8Array, payload: Uint8Array): Uint8Array<ArrayBuffer> {
  const chunk = new Uint8Array(12 + payload.byteLength)
  writeUint32(chunk, 0, payload.byteLength)
  chunk.set(type, 4)
  chunk.set(payload, 8)
  writeUint32(chunk, 8 + payload.byteLength, crc32(type, payload))
  return chunk
}

function maskSize(width: number, height: number): number {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error('修复掩膜尺寸无效')
  }
  if (!Number.isSafeInteger(width * height)) throw new Error('修复掩膜尺寸过大')
  return width * height
}

export function paintMaskCircles(
  pixels: Uint8Array, width: number, height: number,
  circles: readonly BinaryMaskCircle[], value: 0 | 127 | 255,
): boolean {
  if (pixels.length !== maskSize(width, height)) throw new Error('修复掩膜尺寸无效')
  if (circles.length === 0) throw new Error('修复掩膜没有绘制区域')
  let changed = false
  for (const circle of circles) {
    if (
      !Number.isFinite(circle.x)
      || !Number.isFinite(circle.y)
      || !Number.isFinite(circle.radius)
      || circle.radius <= 0
    ) {
      throw new Error('修复笔刷轨迹无效')
    }
    const minX = Math.max(0, Math.floor(circle.x - circle.radius))
    const maxX = Math.min(width - 1, Math.ceil(circle.x + circle.radius))
    const minY = Math.max(0, Math.floor(circle.y - circle.radius))
    const maxY = Math.min(height - 1, Math.ceil(circle.y + circle.radius))
    const radiusSquared = circle.radius * circle.radius
    for (let y = minY; y <= maxY; y += 1) {
      const deltaY = y + 0.5 - circle.y
      const rowOffset = y * width
      for (let x = minX; x <= maxX; x += 1) {
        const deltaX = x + 0.5 - circle.x
        if (deltaX * deltaX + deltaY * deltaY <= radiusSquared && pixels[rowOffset + x] !== value) {
          pixels[rowOffset + x] = value
          changed = true
        }
      }
    }
  }
  return changed
}

async function deflate(scanlines: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const compression = new CompressionStream('deflate')
  const compressed = new Response(compression.readable).arrayBuffer()
  const writer = compression.writable.getWriter()
  await writer.write(scanlines)
  await writer.close()
  return new Uint8Array(await compressed)
}

/** Even-odd polygon fill at pixel centres, clipped to the source image. */
export function paintMaskPolygon(pixels: Uint8Array, width: number, height: number, points: readonly { x: number; y: number }[], value: 0 | 127 | 255): boolean {
  if (pixels.length !== maskSize(width, height)) throw new Error('修复掩膜尺寸无效')
  if (points.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) throw new Error('选区坐标无效')
  if (points.length < 3) return false
  let changed = false
  let low = height, high = 0
  for (const point of points) { low = Math.min(low, point.y); high = Math.max(high, point.y) }
  const minY = Math.max(0, Math.floor(low))
  const maxY = Math.min(height - 1, Math.ceil(high))
  for (let y = minY; y <= maxY; y++) {
    const crossings: number[] = []
    const cy = y + 0.5
    for (let i = 0; i < points.length; i++) {
      const a = points[i]!, b = points[(i + 1) % points.length]!
      if ((a.y > cy) !== (b.y > cy)) crossings.push(a.x + (cy - a.y) * (b.x - a.x) / (b.y - a.y))
    }
    crossings.sort((a, b) => a - b)
    for (let i = 0; i + 1 < crossings.length; i += 2) {
      const left = Math.max(0, Math.ceil(crossings[i]! - 0.5))
      const right = Math.min(width - 1, Math.ceil(crossings[i + 1]! - 0.5) - 1)
      for (let x = left; x <= right; x++) {
        if (pixels[y * width + x] !== value) { pixels[y * width + x] = value; changed = true }
      }
    }
  }
  return changed
}

export async function encodeGrayscaleMaskPng(
  width: number, height: number, pixels: Uint8Array,
): Promise<Blob> {
  if (pixels.length !== maskSize(width, height)) throw new Error('修复掩膜尺寸无效')
  const scanlines = new Uint8Array((width + 1) * height)
  for (let y = 0; y < height; y += 1) {
    scanlines.set(pixels.subarray(y * width, (y + 1) * width), y * (width + 1) + 1)
  }
  const compressed = await deflate(scanlines)
  const header = new Uint8Array(13)
  writeUint32(header, 0, width)
  writeUint32(header, 4, height)
  header[8] = 8
  header[9] = 0
  return new Blob([
    PNG_SIGNATURE,
    pngChunk(IHDR, header),
    pngChunk(IDAT, compressed),
    pngChunk(IEND, new Uint8Array()),
  ], { type: 'image/png' })
}

export async function encodeBinaryMaskPng(
  width: number, height: number, circles: readonly BinaryMaskCircle[],
): Promise<Blob> {
  const pixels = new Uint8Array(maskSize(width, height))
  paintMaskCircles(pixels, width, height, circles, 255)
  return encodeGrayscaleMaskPng(width, height, pixels)
}
