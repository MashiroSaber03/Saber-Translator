/** Read decoded image pixels, independently of DOM overlays and display zoom. */
export function sampleImageColor(
  image: HTMLImageElement,
  point: { clientX: number; clientY: number },
): string | null {
  if (!image.complete || !image.naturalWidth || !image.naturalHeight) return null
  const rect = image.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return null
  const x = Math.floor((point.clientX - rect.left) * image.naturalWidth / rect.width)
  const y = Math.floor((point.clientY - rect.top) * image.naturalHeight / rect.height)
  if (!Number.isFinite(x) || !Number.isFinite(y)
    || x < 0 || y < 0 || x >= image.naturalWidth || y >= image.naturalHeight) return null

  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 1
  const context = canvas.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' })
  if (!context) throw new Error('当前浏览器无法读取图片颜色')
  context.imageSmoothingEnabled = false
  context.drawImage(image, x, y, 1, 1, 0, 0, 1, 1)
  const pixel = context.getImageData(0, 0, 1, 1).data
  // Transparent pixels have no visible image color to assign to an opaque text style.
  if (pixel[3] === 0) return null
  return `#${Array.from(pixel.slice(0, 3), channel => channel.toString(16).padStart(2, '0')).join('')}`
}
