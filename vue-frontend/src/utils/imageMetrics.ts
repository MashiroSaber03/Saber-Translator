export interface ImageDisplayMetrics {
  visualContentWidth: number
  visualContentHeight: number
  visualContentOffsetX: number
  visualContentOffsetY: number
  scaleX: number
  scaleY: number
  naturalWidth: number
  naturalHeight: number
  elementWidth: number
  elementHeight: number
}

type Point = {
  x: number
  y: number
}

type RectCoords = [number, number, number, number]

function resolveContainedImageSize(
  naturalWidth: number,
  naturalHeight: number,
  elementWidth: number,
  elementHeight: number,
): { width: number; height: number } {
  const naturalAspectRatio = naturalWidth / naturalHeight
  const elementAspectRatio = elementWidth / elementHeight

  if (naturalAspectRatio > elementAspectRatio) {
    return {
      width: elementWidth,
      height: elementWidth / naturalAspectRatio,
    }
  }

  return {
    width: elementHeight * naturalAspectRatio,
    height: elementHeight,
  }
}

export function calculateImageDisplayMetrics(
  imageElement: HTMLImageElement | null | undefined,
): ImageDisplayMetrics | null {
  if (!imageElement) {
    return null
  }

  if (!imageElement.complete || imageElement.naturalWidth === 0 || imageElement.naturalHeight === 0) {
    return null
  }

  const naturalWidth = imageElement.naturalWidth
  const naturalHeight = imageElement.naturalHeight
  const elementWidth = imageElement.clientWidth
  const elementHeight = imageElement.clientHeight
  const containedSize = resolveContainedImageSize(naturalWidth, naturalHeight, elementWidth, elementHeight)
  const offsetXInsideElement = (elementWidth - containedSize.width) / 2
  const offsetYInsideElement = (elementHeight - containedSize.height) / 2
  const visualContentOffsetX = imageElement.offsetLeft + offsetXInsideElement
  const visualContentOffsetY = imageElement.offsetTop + offsetYInsideElement
  const scaleX = naturalWidth > 0 ? containedSize.width / naturalWidth : 0
  const scaleY = naturalHeight > 0 ? containedSize.height / naturalHeight : 0

  return {
    visualContentWidth: containedSize.width,
    visualContentHeight: containedSize.height,
    visualContentOffsetX,
    visualContentOffsetY,
    scaleX,
    scaleY,
    naturalWidth,
    naturalHeight,
    elementWidth,
    elementHeight,
  }
}

export function imageToScreenCoords(
  imageX: number,
  imageY: number,
  metrics: ImageDisplayMetrics,
): Point {
  return {
    x: imageX * metrics.scaleX + metrics.visualContentOffsetX,
    y: imageY * metrics.scaleY + metrics.visualContentOffsetY,
  }
}

export function screenToImageCoords(
  screenX: number,
  screenY: number,
  metrics: ImageDisplayMetrics,
): Point {
  if (metrics.scaleX === 0 || metrics.scaleY === 0) {
    return { x: 0, y: 0 }
  }

  return {
    x: (screenX - metrics.visualContentOffsetX) / metrics.scaleX,
    y: (screenY - metrics.visualContentOffsetY) / metrics.scaleY,
  }
}

export function bubbleCoordsToScreen(
  coords: RectCoords,
  metrics: ImageDisplayMetrics,
): RectCoords {
  const topLeft = imageToScreenCoords(coords[0], coords[1], metrics)
  const bottomRight = imageToScreenCoords(coords[2], coords[3], metrics)
  return [topLeft.x, topLeft.y, bottomRight.x, bottomRight.y]
}

export function screenCoordsToBubble(
  screenCoords: RectCoords,
  metrics: ImageDisplayMetrics,
): RectCoords {
  const topLeft = screenToImageCoords(screenCoords[0], screenCoords[1], metrics)
  const bottomRight = screenToImageCoords(screenCoords[2], screenCoords[3], metrics)

  return [
    Math.round(topLeft.x),
    Math.round(topLeft.y),
    Math.round(bottomRight.x),
    Math.round(bottomRight.y),
  ]
}

export function polygonToScreen(
  polygon: number[][],
  metrics: ImageDisplayMetrics,
): number[][] {
  return polygon.map(point => {
    const x = point[0] ?? 0
    const y = point[1] ?? 0
    const screenPoint = imageToScreenCoords(x, y, metrics)
    return [screenPoint.x, screenPoint.y]
  })
}

export function screenPolygonToImage(
  screenPolygon: number[][],
  metrics: ImageDisplayMetrics,
): number[][] {
  return screenPolygon.map(point => {
    const x = point[0] ?? 0
    const y = point[1] ?? 0
    const imagePoint = screenToImageCoords(x, y, metrics)
    return [Math.round(imagePoint.x), Math.round(imagePoint.y)]
  })
}

export function scaleSize(
  width: number,
  height: number,
  metrics: ImageDisplayMetrics,
): { width: number; height: number } {
  return {
    width: width * metrics.scaleX,
    height: height * metrics.scaleY,
  }
}

export function isPointInVisualContent(
  screenX: number,
  screenY: number,
  metrics: ImageDisplayMetrics,
): boolean {
  const { visualContentOffsetX, visualContentOffsetY, visualContentWidth, visualContentHeight } = metrics

  return (
    screenX >= visualContentOffsetX &&
    screenX <= visualContentOffsetX + visualContentWidth &&
    screenY >= visualContentOffsetY &&
    screenY <= visualContentOffsetY + visualContentHeight
  )
}
