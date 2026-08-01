import { expect, test } from '@playwright/test'

test('Chromium creates a backend-compatible grayscale repair mask', async ({ page }) => {
  await page.goto('/')

  const mask = await page.evaluate(async () => {
    const moduleUrl = new URL('/src/utils/binaryMaskPng.ts', window.location.origin).href
    const { encodeBinaryMaskPng } = await import(moduleUrl)
    const blob = await encodeBinaryMaskPng(16, 12, [{ x: 8, y: 6, radius: 3 }])
    const bytes = new Uint8Array(await blob.arrayBuffer())
    const bitmap = await createImageBitmap(blob)
    const result = {
      bitDepth: bytes[24],
      colorType: bytes[25],
      height: bitmap.height,
      mimeType: blob.type,
      signature: [...bytes.slice(0, 8)],
      width: bitmap.width,
    }
    bitmap.close()
    return result
  })

  expect(mask).toEqual({
    bitDepth: 8,
    colorType: 0,
    height: 12,
    mimeType: 'image/png',
    signature: [137, 80, 78, 71, 13, 10, 26, 10],
    width: 16,
  })
})
