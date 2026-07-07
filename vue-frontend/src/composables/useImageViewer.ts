import { computed, ref } from 'vue'

export interface TransformState {
  scale: number
  translateX: number
  translateY: number
}

export interface ImageViewerOptions {
  minScale?: number
  maxScale?: number
  zoomSpeed?: number
  onScaleChange?: (scale: number) => void
  onTransformChange?: (transform: TransformState) => void
}

const DEFAULT_OPTIONS: Required<Omit<ImageViewerOptions, 'onScaleChange' | 'onTransformChange'>> = {
  minScale: 0.1,
  maxScale: 5,
  zoomSpeed: 0.1,
}

function normalizePositiveOption(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function createViewerConfig(options: ImageViewerOptions) {
  const minScale = normalizePositiveOption(options.minScale, DEFAULT_OPTIONS.minScale)
  const requestedMaxScale = normalizePositiveOption(options.maxScale, DEFAULT_OPTIONS.maxScale)

  return {
    minScale,
    maxScale: Math.max(requestedMaxScale, minScale),
    zoomSpeed: normalizePositiveOption(options.zoomSpeed, DEFAULT_OPTIONS.zoomSpeed),
  }
}

export function useImageViewer(options: ImageViewerOptions = {}) {
  const config = createViewerConfig(options)
  const scale = ref(1)
  const translateX = ref(0)
  const translateY = ref(0)
  const isDragging = ref(false)
  const lastX = ref(0)
  const lastY = ref(0)

  const transformStyle = computed(() => ({
    transform: `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`,
  }))

  function clampScale(value: number): number {
    if (!Number.isFinite(value)) {
      return config.minScale
    }
    return Math.min(Math.max(value, config.minScale), config.maxScale)
  }

  function getTransform(): TransformState {
    return {
      scale: scale.value,
      translateX: translateX.value,
      translateY: translateY.value,
    }
  }

  function notifyTransform(): void {
    options.onTransformChange?.(getTransform())
  }

  function notifyScaleAndTransform(): void {
    options.onScaleChange?.(scale.value)
    notifyTransform()
  }

  function zoomAt(x: number, y: number, factor: number): void {
    const currentScale = clampScale(scale.value)
    scale.value = currentScale
    const newScale = clampScale(currentScale * factor)
    const scaleChange = newScale / currentScale

    translateX.value = x - (x - translateX.value) * scaleChange
    translateY.value = y - (y - translateY.value) * scaleChange
    scale.value = newScale

    notifyScaleAndTransform()
  }

  function zoom(factor: number, viewportWidth = 800, viewportHeight = 600): void {
    zoomAt(viewportWidth / 2, viewportHeight / 2, factor)
  }

  function zoomIn(): void {
    zoom(1.2)
  }

  function zoomOut(): void {
    zoom(0.8)
  }

  function setScale(newScale: number, viewportWidth = 800, viewportHeight = 600): void {
    scale.value = clampScale(scale.value)
    const factor = clampScale(newScale) / scale.value
    zoomAt(viewportWidth / 2, viewportHeight / 2, factor)
  }

  function startDrag(x: number, y: number): void {
    isDragging.value = true
    lastX.value = x
    lastY.value = y
  }

  function drag(x: number, y: number): void {
    if (!isDragging.value) return

    const dx = x - lastX.value
    const dy = y - lastY.value
    translateX.value += dx
    translateY.value += dy
    lastX.value = x
    lastY.value = y

    notifyTransform()
  }

  function endDrag(): void {
    isDragging.value = false
  }

  function pan(dx: number, dy: number): void {
    translateX.value += dx
    translateY.value += dy
    notifyTransform()
  }

  function reset(): void {
    scale.value = 1
    translateX.value = 0
    translateY.value = 0
    notifyScaleAndTransform()
  }

  function resetZoom(): void {
    reset()
  }

  function resetTransform(): void {
    scale.value = 1
    translateX.value = 0
    translateY.value = 0
  }

  function fitToScreen(
    imageWidth: number,
    imageHeight: number,
    viewportWidth: number,
    viewportHeight: number
  ): void {
    if (imageWidth <= 0 || imageHeight <= 0 || viewportWidth <= 0 || viewportHeight <= 0) return

    const scaleX = viewportWidth / imageWidth
    const scaleY = viewportHeight / imageHeight
    scale.value = Math.min(scaleX, scaleY) * 0.95
    translateX.value = (viewportWidth - imageWidth * scale.value) / 2
    translateY.value = (viewportHeight - imageHeight * scale.value) / 2

    notifyScaleAndTransform()
  }

  function setTransform(transform: Partial<TransformState>): void {
    if (transform.scale !== undefined) {
      scale.value = clampScale(transform.scale)
    }
    if (transform.translateX !== undefined) {
      translateX.value = transform.translateX
    }
    if (transform.translateY !== undefined) {
      translateY.value = transform.translateY
    }
    notifyTransform()
  }

  function syncWith(otherTransform: TransformState): void {
    scale.value = otherTransform.scale
    translateX.value = otherTransform.translateX
    translateY.value = otherTransform.translateY
  }

  function scrollToBubble(
    bubbleCoords: [number, number, number, number],
    viewportWidth: number,
    viewportHeight: number
  ): void {
    if (bubbleCoords.length < 4) return

    const [x1, y1, x2, y2] = bubbleCoords
    const bubbleCenterX = (x1 + x2) / 2
    const bubbleCenterY = (y1 + y2) / 2

    translateX.value = viewportWidth / 2 - bubbleCenterX * scale.value
    translateY.value = viewportHeight / 2 - bubbleCenterY * scale.value

    notifyTransform()
  }

  return {
    scale,
    translateX,
    translateY,
    isDragging,
    transformStyle,
    zoomAt,
    zoom,
    zoomIn,
    zoomOut,
    setScale,
    startDrag,
    drag,
    endDrag,
    pan,
    reset,
    resetZoom,
    resetTransform,
    fitToScreen,
    getTransform,
    setTransform,
    syncWith,
    scrollToBubble,
  }
}
