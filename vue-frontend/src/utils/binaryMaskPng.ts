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

function createScanlines(
  width: number,
  height: number,
  circles: readonly BinaryMaskCircle[],
): Uint8Array<ArrayBuffer> {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error('修复掩膜尺寸无效')
  }
  if (circles.length === 0) throw new Error('修复掩膜没有绘制区域')
  const stride = width + 1
  const byteLength = stride * height
  if (!Number.isSafeInteger(byteLength)) throw new Error('修复掩膜尺寸过大')
  const scanlines = new Uint8Array(byteLength)

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
      const rowOffset = y * stride + 1
      for (let x = minX; x <= maxX; x += 1) {
        const deltaX = x + 0.5 - circle.x
        if (deltaX * deltaX + deltaY * deltaY <= radiusSquared) {
          scanlines[rowOffset + x] = 255
        }
      }
    }
  }
  return scanlines
}

async function deflate(scanlines: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const compression = new CompressionStream('deflate')
  const compressed = new Response(compression.readable).arrayBuffer()
  const writer = compression.writable.getWriter()
  await writer.write(scanlines)
  await writer.close()
  return new Uint8Array(await compressed)
}

export async function encodeBinaryMaskPng(
  width: number,
  height: number,
  circles: readonly BinaryMaskCircle[],
): Promise<Blob> {
  const scanlines = createScanlines(width, height, circles)
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
