import { inflateSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'

import { encodeBinaryMaskPng } from '@/utils/binaryMaskPng'

function readBlobBytes(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('读取掩膜失败'))
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer))
    reader.readAsArrayBuffer(blob)
  })
}

async function decodePng(blob: Blob) {
  const bytes = await readBlobBytes(blob)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const chunks: string[] = []
  const idat: Uint8Array[] = []
  let offset = 8
  while (offset < bytes.byteLength) {
    const length = view.getUint32(offset)
    const type = new TextDecoder().decode(bytes.slice(offset + 4, offset + 8))
    const payload = bytes.slice(offset + 8, offset + 8 + length)
    chunks.push(type)
    if (type === 'IDAT') idat.push(payload)
    offset += 12 + length
  }
  const scanlines = new Uint8Array(inflateSync(Buffer.concat(idat.map(value => Buffer.from(value)))))
  return {
    bitDepth: bytes[24],
    chunks,
    colorType: bytes[25],
    height: view.getUint32(20),
    scanlines,
    signature: [...bytes.slice(0, 8)],
    width: view.getUint32(16),
  }
}

describe('binary mask PNG encoder', () => {
  it('encodes a single-frame 8-bit grayscale binary PNG', async () => {
    const blob = await encodeBinaryMaskPng(16, 12, [{ x: 8, y: 6, radius: 3 }])
    const png = await decodePng(blob)

    expect(blob.type).toBe('image/png')
    expect(png.signature).toEqual([137, 80, 78, 71, 13, 10, 26, 10])
    expect(png.width).toBe(16)
    expect(png.height).toBe(12)
    expect(png.bitDepth).toBe(8)
    expect(png.colorType).toBe(0)
    expect(png.chunks).toEqual(['IHDR', 'IDAT', 'IEND'])

    const pixels: number[] = []
    const stride = 17
    for (let y = 0; y < 12; y += 1) {
      expect(png.scanlines[y * stride]).toBe(0)
      pixels.push(...png.scanlines.slice(y * stride + 1, (y + 1) * stride))
    }
    expect(new Set(pixels)).toEqual(new Set([0, 255]))
    expect(png.scanlines[6 * stride + 1 + 8]).toBe(255)
    expect(png.scanlines[1]).toBe(0)
  })

  it('rejects empty drawing input instead of uploading an empty mask', async () => {
    await expect(encodeBinaryMaskPng(16, 12, [])).rejects.toThrow('没有绘制区域')
  })
})
