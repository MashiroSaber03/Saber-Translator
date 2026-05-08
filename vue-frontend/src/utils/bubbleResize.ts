import type { BubbleCoords } from '@/types/bubble'

export type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'

interface ResizeHandleDescriptor {
  sx: -1 | 0 | 1
  sy: -1 | 0 | 1
}

export interface ResizeCoordsOptions {
  rotationAngle?: number
  minSize?: number
  imageWidth?: number
  imageHeight?: number
  clampToImage?: boolean
  round?: boolean
}

const HANDLE_DESCRIPTORS: Record<ResizeHandle, ResizeHandleDescriptor> = {
  nw: { sx: -1, sy: -1 },
  n: { sx: 0, sy: -1 },
  ne: { sx: 1, sy: -1 },
  e: { sx: 1, sy: 0 },
  se: { sx: 1, sy: 1 },
  s: { sx: 0, sy: 1 },
  sw: { sx: -1, sy: 1 },
  w: { sx: -1, sy: 0 },
}

function project(x: number, y: number, axisX: number, axisY: number): number {
  return x * axisX + y * axisY
}

function buildCoords(centerX: number, centerY: number, width: number, height: number): BubbleCoords {
  const halfWidth = width / 2
  const halfHeight = height / 2
  return [
    centerX - halfWidth,
    centerY - halfHeight,
    centerX + halfWidth,
    centerY + halfHeight,
  ]
}

/**
 * 根据旋转矩形几何计算缩放后的坐标。
 * 坐标始终存储为未旋转矩形 + 独立 rotationAngle，因此需要先在旋转局部坐标系中求解，
 * 再转换回 axis-aligned coords。
 */
export function calculateResizedCoords(
  coords: BubbleCoords,
  handle: ResizeHandle,
  deltaX: number,
  deltaY: number,
  options: ResizeCoordsOptions = {}
): BubbleCoords | null {
  const descriptor = HANDLE_DESCRIPTORS[handle]
  const rotationAngle = options.rotationAngle ?? 0
  const minSize = options.minSize ?? 10

  const [x1, y1, x2, y2] = coords
  const width = x2 - x1
  const height = y2 - y1
  const centerX = (x1 + x2) / 2
  const centerY = (y1 + y2) / 2
  const halfWidth = width / 2
  const halfHeight = height / 2

  const angleRad = rotationAngle * Math.PI / 180
  const cos = Math.cos(angleRad)
  const sin = Math.sin(angleRad)

  // 旋转后的局部 X / Y 轴单位向量
  const axisX = { x: cos, y: sin }
  const axisY = { x: -sin, y: cos }

  const handleOffsetX =
    descriptor.sx * halfWidth * axisX.x + descriptor.sy * halfHeight * axisY.x
  const handleOffsetY =
    descriptor.sx * halfWidth * axisX.y + descriptor.sy * halfHeight * axisY.y

  const anchorOffsetX =
    -descriptor.sx * halfWidth * axisX.x - descriptor.sy * halfHeight * axisY.x
  const anchorOffsetY =
    -descriptor.sx * halfWidth * axisX.y - descriptor.sy * halfHeight * axisY.y

  const handleX = centerX + handleOffsetX
  const handleY = centerY + handleOffsetY
  const anchorX = centerX + anchorOffsetX
  const anchorY = centerY + anchorOffsetY

  const mouseX = handleX + deltaX
  const mouseY = handleY + deltaY
  const anchorToMouseX = mouseX - anchorX
  const anchorToMouseY = mouseY - anchorY

  let nextWidth = width
  let nextHeight = height
  let widthSign: -1 | 0 | 1 = descriptor.sx
  let heightSign: -1 | 0 | 1 = descriptor.sy

  if (descriptor.sx !== 0) {
    const projectedWidth = descriptor.sx * project(
      anchorToMouseX,
      anchorToMouseY,
      axisX.x,
      axisX.y
    )
    nextWidth = Math.abs(projectedWidth)
    widthSign = projectedWidth >= 0 ? descriptor.sx : (descriptor.sx * -1) as -1 | 1
  }

  if (descriptor.sy !== 0) {
    const projectedHeight = descriptor.sy * project(
      anchorToMouseX,
      anchorToMouseY,
      axisY.x,
      axisY.y
    )
    nextHeight = Math.abs(projectedHeight)
    heightSign = projectedHeight >= 0 ? descriptor.sy : (descriptor.sy * -1) as -1 | 1
  }

  if (nextWidth < minSize || nextHeight < minSize) {
    return null
  }

  let nextCenterX = anchorX
  let nextCenterY = anchorY

  if (descriptor.sx !== 0) {
    nextCenterX += widthSign * (nextWidth / 2) * axisX.x
    nextCenterY += widthSign * (nextWidth / 2) * axisX.y
  }

  if (descriptor.sy !== 0) {
    nextCenterX += heightSign * (nextHeight / 2) * axisY.x
    nextCenterY += heightSign * (nextHeight / 2) * axisY.y
  }

  let nextCoords = buildCoords(nextCenterX, nextCenterY, nextWidth, nextHeight)

  if (options.clampToImage) {
    const maxWidth = options.imageWidth ?? Number.POSITIVE_INFINITY
    const maxHeight = options.imageHeight ?? Number.POSITIVE_INFINITY
    nextCoords = [
      Math.max(0, nextCoords[0]),
      Math.max(0, nextCoords[1]),
      Math.min(maxWidth, nextCoords[2]),
      Math.min(maxHeight, nextCoords[3]),
    ]
  }

  if (options.round) {
    nextCoords = nextCoords.map(value => Math.round(value)) as BubbleCoords
  }

  if (nextCoords[2] - nextCoords[0] < minSize || nextCoords[3] - nextCoords[1] < minSize) {
    return null
  }

  return nextCoords
}
