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

export {
  naturalSortKey,
  naturalSortCompare,
  naturalSort
} from './naturalSort'

export {
  deepClone
} from './deepClone'

export {
  calculateDraggedCoords
} from './bubbleDrag'

export {
  triggerBlobDownload,
  triggerUrlDownload
} from './browserDownload'

export {
  copyTextToClipboard
} from './clipboard'

export {
  normalizeAppPath,
  isKnownFrontendRoute,
  classifyAppPath,
  buildApiPath,
  buildStaticPath,
  buildVueStaticAssetPath,
  type RouteClassification
} from './routePath'

export {
  applyFieldMappings,
  configFromApi,
  configToApi,
  toCamelCase,
  toSnakeCase
} from './insightConverters'
