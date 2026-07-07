import type { BubbleCoords } from '@/types/bubble'

export function calculateDraggedCoords(
  coords: BubbleCoords,
  deltaX: number,
  deltaY: number,
  imageWidth: number,
  imageHeight: number
): BubbleCoords {
  const [x1, y1, x2, y2] = coords
  const width = x2 - x1
  const height = y2 - y1
  const safeWidth = Math.min(width, imageWidth)
  const safeHeight = Math.min(height, imageHeight)
  const nextX1 = Math.max(0, Math.min(Math.round(x1 + deltaX), imageWidth - safeWidth))
  const nextY1 = Math.max(0, Math.min(Math.round(y1 + deltaY), imageHeight - safeHeight))

  return [nextX1, nextY1, nextX1 + safeWidth, nextY1 + safeHeight]
}
