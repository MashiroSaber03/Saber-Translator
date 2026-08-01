import { ref } from 'vue'

export interface TransformState {
  scale: number
  translateX: number
  translateY: number
}

export interface ImageViewerOptions {
  minScale?: number
  maxScale?: number
}

const DEFAULT_OPTIONS: Required<ImageViewerOptions> = {
  minScale: 0.1,
  maxScale: 5,
}

function normalizePositiveOption(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}

function createViewerConfig(options: ImageViewerOptions) {
  const minScale = normalizePositiveOption(options.minScale, DEFAULT_OPTIONS.minScale)
  const requestedMaxScale = normalizePositiveOption(options.maxScale, DEFAULT_OPTIONS.maxScale)

  return {
    minScale,
    maxScale: Math.max(requestedMaxScale, minScale),
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

  function zoomAt(x: number, y: number, factor: number): void {
    const currentScale = clampScale(scale.value)
    scale.value = currentScale
    const newScale = clampScale(currentScale * factor)
    const scaleChange = newScale / currentScale

    translateX.value = x - (x - translateX.value) * scaleChange
    translateY.value = y - (y - translateY.value) * scaleChange
    scale.value = newScale

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

  }

  function endDrag(): void {
    isDragging.value = false
  }

  function reset(): void {
    scale.value = 1
    translateX.value = 0
    translateY.value = 0
  }

  function resetZoom(): void {
    reset()
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
  }

  return {
    scale,
    translateX,
    translateY,
    zoomAt,
    zoomIn,
    zoomOut,
    startDrag,
    drag,
    endDrag,
    resetZoom,
    getTransform,
    setTransform,
  }
}
