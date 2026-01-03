/**
 * 工具函数统一导出
 */

// 气泡状态工厂函数
export {
  createBubbleState,
  detectTextDirection,
  createBubbleStatesFromResponse,
  bubbleStatesToApiRequest,
  updateBubbleState,
  updateAllBubbleStates,
  cloneBubbleStates,
  cloneBubbleState,
  isValidBubbleState,
  getBubbleCenter,
  getBubbleSize,
  isPointInBubble,
  isPointInPolygon,
  isPointInBubbleArea
} from './bubbleFactory'

// RPM 限速器
export {
  createRateLimiter,
  createRateLimitedExecutor,
  executeBatchWithRateLimit,
  type RateLimiter
} from './rateLimiter'

// 图片显示指标计算
export {
  calculateImageDisplayMetrics,
  imageToScreenCoords,
  screenToImageCoords,
  bubbleCoordsToScreen,
  screenCoordsToBubble,
  polygonToScreen,
  screenPolygonToImage,
  scaleSize,
  isPointInVisualContent,
  type ImageDisplayMetrics
} from './imageMetrics'

// 颜色工具函数
export {
  rgbArrayToHex,
  hexToRgbArray,
  isValidHex,
  normalizeHex,
  isSameColor,
  isRgbEqualToHex,
  colorDifference,
  isDarkColor,
  getContrastColor,
  formatRgb,
  formatConfidence,
  type RgbArray
} from './colorUtils'
