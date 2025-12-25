/**
 * 图片显示指标计算工具函数
 * 用于计算图片在容器中的实际显示尺寸、缩放比例和偏移量
 * 主要用于气泡坐标转换（图片坐标 ↔ 屏幕坐标）
 */

/**
 * 图片显示指标接口
 */
export interface ImageDisplayMetrics {
  /** 图像内容在屏幕上的实际渲染宽度 */
  visualContentWidth: number
  /** 图像内容在屏幕上的实际渲染高度 */
  visualContentHeight: number
  /** 图像内容左上角相对于容器的X轴偏移 */
  visualContentOffsetX: number
  /** 图像内容左上角相对于容器的Y轴偏移 */
  visualContentOffsetY: number
  /** 水平缩放比例 (visualContentWidth / naturalWidth) */
  scaleX: number
  /** 垂直缩放比例 (visualContentHeight / naturalHeight) */
  scaleY: number
  /** 图像的原始宽度 */
  naturalWidth: number
  /** 图像的原始高度 */
  naturalHeight: number
  /** img 元素本身的宽度 */
  elementWidth: number
  /** img 元素本身的高度 */
  elementHeight: number
}

/**
 * 计算图像内容在其 img 元素中的实际显示指标
 * 考虑到 object-fit: contain 的影响
 * 
 * @param imageElement - 图片 DOM 元素（HTMLImageElement 或 jQuery 对象）
 * @returns 图片显示指标对象，如果图片无效或未加载完成则返回 null
 * 
 * @example
 * ```typescript
 * const img = document.getElementById('myImage') as HTMLImageElement
 * const metrics = calculateImageDisplayMetrics(img)
 * if (metrics) {
 *   // 将图片坐标转换为屏幕坐标
 *   const screenX = imageX * metrics.scaleX + metrics.visualContentOffsetX
 *   const screenY = imageY * metrics.scaleY + metrics.visualContentOffsetY
 * }
 * ```
 */
export function calculateImageDisplayMetrics(
  imageElement: HTMLImageElement | null | undefined
): ImageDisplayMetrics | null {
  // 参数验证
  if (!imageElement) {
    console.error('calculateImageDisplayMetrics: 图像元素无效')
    return null
  }

  // 检查图片是否加载完成
  if (!imageElement.complete || imageElement.naturalWidth === 0 || imageElement.naturalHeight === 0) {
    console.warn('calculateImageDisplayMetrics: 图像未完全加载或尺寸为0')
    return null
  }

  const naturalWidth = imageElement.naturalWidth
  const naturalHeight = imageElement.naturalHeight

  // img 元素在屏幕上的实际渲染尺寸
  const elementWidth = imageElement.clientWidth
  const elementHeight = imageElement.clientHeight

  // 计算实际显示尺寸（考虑 object-fit: contain）
  let visualContentWidth: number
  let visualContentHeight: number
  const naturalAspectRatio = naturalWidth / naturalHeight
  const elementAspectRatio = elementWidth / elementHeight

  if (naturalAspectRatio > elementAspectRatio) {
    // 图片比元素框更"宽"，宽度填满，高度按比例缩放（上下留白）
    visualContentWidth = elementWidth
    visualContentHeight = elementWidth / naturalAspectRatio
  } else {
    // 图片比元素框更"高"，高度填满，宽度按比例缩放（左右留白）
    visualContentHeight = elementHeight
    visualContentWidth = elementHeight * naturalAspectRatio
  }

  // 图像内容在其元素框内的偏移（由于 object-fit: contain，内容会居中）
  const offsetXInsideElement = (elementWidth - visualContentWidth) / 2
  const offsetYInsideElement = (elementHeight - visualContentHeight) / 2

  // img 元素本身相对于其 offsetParent 的偏移
  const elementOffsetX = imageElement.offsetLeft
  const elementOffsetY = imageElement.offsetTop

  // 最终，图像内容左上角相对于容器的偏移
  const finalVisualContentOffsetX = elementOffsetX + offsetXInsideElement
  const finalVisualContentOffsetY = elementOffsetY + offsetYInsideElement

  // 计算缩放比例
  const finalScaleX = naturalWidth > 0 ? visualContentWidth / naturalWidth : 0
  const finalScaleY = naturalHeight > 0 ? visualContentHeight / naturalHeight : 0

  return {
    visualContentWidth,
    visualContentHeight,
    visualContentOffsetX: finalVisualContentOffsetX,
    visualContentOffsetY: finalVisualContentOffsetY,
    scaleX: finalScaleX,
    scaleY: finalScaleY,
    naturalWidth,
    naturalHeight,
    elementWidth,
    elementHeight
  }
}

/**
 * 将图片坐标转换为屏幕坐标
 * 
 * @param imageX - 图片上的 X 坐标
 * @param imageY - 图片上的 Y 坐标
 * @param metrics - 图片显示指标
 * @returns 屏幕坐标 { x, y }
 */
export function imageToScreenCoords(
  imageX: number,
  imageY: number,
  metrics: ImageDisplayMetrics
): { x: number; y: number } {
  return {
    x: imageX * metrics.scaleX + metrics.visualContentOffsetX,
    y: imageY * metrics.scaleY + metrics.visualContentOffsetY
  }
}

/**
 * 将屏幕坐标转换为图片坐标
 * 
 * @param screenX - 屏幕上的 X 坐标（相对于容器）
 * @param screenY - 屏幕上的 Y 坐标（相对于容器）
 * @param metrics - 图片显示指标
 * @returns 图片坐标 { x, y }
 */
export function screenToImageCoords(
  screenX: number,
  screenY: number,
  metrics: ImageDisplayMetrics
): { x: number; y: number } {
  // 避免除以零
  if (metrics.scaleX === 0 || metrics.scaleY === 0) {
    return { x: 0, y: 0 }
  }
  
  return {
    x: (screenX - metrics.visualContentOffsetX) / metrics.scaleX,
    y: (screenY - metrics.visualContentOffsetY) / metrics.scaleY
  }
}

/**
 * 将气泡矩形坐标（图片坐标系）转换为屏幕坐标系
 * 
 * @param coords - 气泡坐标 [x1, y1, x2, y2]
 * @param metrics - 图片显示指标
 * @returns 屏幕坐标 [x1, y1, x2, y2]
 */
export function bubbleCoordsToScreen(
  coords: [number, number, number, number],
  metrics: ImageDisplayMetrics
): [number, number, number, number] {
  const topLeft = imageToScreenCoords(coords[0], coords[1], metrics)
  const bottomRight = imageToScreenCoords(coords[2], coords[3], metrics)
  
  return [topLeft.x, topLeft.y, bottomRight.x, bottomRight.y]
}

/**
 * 将屏幕坐标系的矩形转换为图片坐标系
 * 
 * @param screenCoords - 屏幕坐标 [x1, y1, x2, y2]
 * @param metrics - 图片显示指标
 * @returns 图片坐标 [x1, y1, x2, y2]
 */
export function screenCoordsToBubble(
  screenCoords: [number, number, number, number],
  metrics: ImageDisplayMetrics
): [number, number, number, number] {
  const topLeft = screenToImageCoords(screenCoords[0], screenCoords[1], metrics)
  const bottomRight = screenToImageCoords(screenCoords[2], screenCoords[3], metrics)
  
  return [
    Math.round(topLeft.x),
    Math.round(topLeft.y),
    Math.round(bottomRight.x),
    Math.round(bottomRight.y)
  ]
}

/**
 * 将多边形坐标（图片坐标系）转换为屏幕坐标系
 * 
 * @param polygon - 多边形顶点数组 [[x1, y1], [x2, y2], ...]
 * @param metrics - 图片显示指标
 * @returns 屏幕坐标系的多边形顶点数组
 */
export function polygonToScreen(
  polygon: number[][],
  metrics: ImageDisplayMetrics
): number[][] {
  return polygon.map(point => {
    const x = point[0] ?? 0
    const y = point[1] ?? 0
    const screenPoint = imageToScreenCoords(x, y, metrics)
    return [screenPoint.x, screenPoint.y]
  })
}

/**
 * 将屏幕坐标系的多边形转换为图片坐标系
 * 
 * @param screenPolygon - 屏幕坐标系的多边形顶点数组
 * @param metrics - 图片显示指标
 * @returns 图片坐标系的多边形顶点数组
 */
export function screenPolygonToImage(
  screenPolygon: number[][],
  metrics: ImageDisplayMetrics
): number[][] {
  return screenPolygon.map(point => {
    const x = point[0] ?? 0
    const y = point[1] ?? 0
    const imagePoint = screenToImageCoords(x, y, metrics)
    return [Math.round(imagePoint.x), Math.round(imagePoint.y)]
  })
}

/**
 * 计算缩放后的尺寸
 * 
 * @param width - 原始宽度
 * @param height - 原始高度
 * @param metrics - 图片显示指标
 * @returns 缩放后的尺寸 { width, height }
 */
export function scaleSize(
  width: number,
  height: number,
  metrics: ImageDisplayMetrics
): { width: number; height: number } {
  return {
    width: width * metrics.scaleX,
    height: height * metrics.scaleY
  }
}

/**
 * 检查点是否在图片可视区域内
 * 
 * @param screenX - 屏幕 X 坐标
 * @param screenY - 屏幕 Y 坐标
 * @param metrics - 图片显示指标
 * @returns 是否在可视区域内
 */
export function isPointInVisualContent(
  screenX: number,
  screenY: number,
  metrics: ImageDisplayMetrics
): boolean {
  const { visualContentOffsetX, visualContentOffsetY, visualContentWidth, visualContentHeight } = metrics
  
  return (
    screenX >= visualContentOffsetX &&
    screenX <= visualContentOffsetX + visualContentWidth &&
    screenY >= visualContentOffsetY &&
    screenY <= visualContentOffsetY + visualContentHeight
  )
}
