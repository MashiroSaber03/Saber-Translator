import { computed, onUnmounted, ref, watch } from 'vue'
import { getPageDocument } from '@/api/v2/content'
import {
  createMaskRepair,
  waitForOperation,
} from '@/api/v2/operations'
import {
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import {
  BRUSH_DEFAULT_SIZE,
  BRUSH_MAX_SIZE,
  BRUSH_MIN_SIZE,
} from '@/constants'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { InpaintMethod } from '@/types/bubble'
import { encodeBinaryMaskPng, encodeGrayscaleMaskPng, paintMaskPolygon } from '@/utils/binaryMaskPng'
import { showToast } from '@/utils/toast'

export type BrushMode = 'repair' | 'restore' | 'clone' | null
export type BrushShape = 'brush' | 'rectangle' | 'lasso'

export interface BrushPosition {
  x: number
  y: number
  scale: number
}

export interface BrushSurface {
  viewport: HTMLElement
  wrapper: HTMLElement
  image: HTMLImageElement
}

export interface BrushBounds {
  path: BrushPosition[]
  radius: number
  polygon?: BrushPosition[]
}

export interface CurrentRepairSettings {
  inpaintMethod: InpaintMethod
  fillColor: string
}

export interface BrushCallbacks {
  onBrushComplete?: () => void | Promise<unknown>
  getCurrentRepairSettings?: () => CurrentRepairSettings
  onMaskStroke?: (bounds: BrushBounds) => Promise<void> | undefined
  getBrushColor?: () => string | undefined
}

export function useBrush(callbacks?: BrushCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const brushMode = ref<BrushMode>(null)
  const cloneSource = ref<{ x: number; y: number } | null>(null)
  let cloneOffset: [number, number] | null = null
  watch(() => imageStore.currentImage?.id, () => {
    // A page change cancels its unfinished stroke; never submit old coordinates to the new page.
    isBrushPainting.value = false
    exitBrushMode()
    cloneSource.value = null
    cloneOffset = null
  }, { flush: 'sync' })
  const brushShape = ref<BrushShape>('brush')
  const brushSize = ref(BRUSH_DEFAULT_SIZE)
  const isBrushPainting = ref(false)
  const isBrushSubmitting = ref(false)
  const brushPath = ref<BrushPosition[]>([])
  const mouseX = ref(0)
  const mouseY = ref(0)
  const abortController = new AbortController()
  let activeSurface: BrushSurface | null = null
  let brushCanvas: HTMLCanvasElement | null = null
  let brushCtx: CanvasRenderingContext2D | null = null
  let brushCanvasScaleX = 1
  let brushCanvasScaleY = 1
  let isOwnerDisposed = false

  const isActive = computed(() => brushMode.value !== null)
  const brushColor = computed(() => {
    const override = callbacks?.getBrushColor?.()
    if (override) return override
    if (brushMode.value === 'repair' || brushMode.value === 'clone') {
      return 'rgba(76, 175, 80, 0.4)'
    }
    if (brushMode.value === 'restore') {
      return 'rgba(33, 150, 243, 0.4)'
    }
    return 'transparent'
  })

  function enterBrushMode(mode: Exclude<BrushMode, null>): void {
    if (brushMode.value === mode) return
    brushMode.value = mode
    if (mode === 'clone') brushShape.value = 'brush'
    brushPath.value = []
  }

  function exitBrushMode(): void {
    if (isBrushPainting.value && brushShape.value === 'brush') finishBrushPainting()
    brushMode.value = null
    isBrushPainting.value = false
    brushPath.value = []
    activeSurface = null
    removeBrushCanvas()
  }

  function toggleBrushMode(mode: Exclude<BrushMode, null>): void {
    if (brushMode.value === mode) exitBrushMode()
    else enterBrushMode(mode)
  }

  function setBrushSize(size: number): void {
    brushSize.value = Math.max(BRUSH_MIN_SIZE, Math.min(BRUSH_MAX_SIZE, size))
  }

  function adjustBrushSize(delta: number): void {
    if (isBrushPainting.value) return
    setBrushSize(brushSize.value + delta)
  }

  function startBrushPainting(event: MouseEvent, surface: BrushSurface): void {
    if (!isActive.value || isBrushSubmitting.value || event.button !== 0) return
    const position = getBrushPositionInImage(event, surface)
    if (!position) return
    event.preventDefault()
    event.stopPropagation()
    if (brushMode.value === 'clone') {
      if (event.altKey) { cloneSource.value = { x: Math.round(position.x), y: Math.round(position.y) }; cloneOffset = null; return }
      if (!cloneSource.value) { showToast('先按住 Alt 点击干净背景取样，再涂抹要修补的位置', 'info'); return }
      cloneOffset ??= [cloneSource.value.x - Math.round(position.x), cloneSource.value.y - Math.round(position.y)]
    }
    isBrushPainting.value = true
    activeSurface = surface
    brushPath.value = []
    brushPath.value.push(position)
    createBrushCanvas(surface)
    drawBrushStroke(position)
  }

  function continueBrushPainting(event: MouseEvent): void {
    mouseX.value = event.clientX
    mouseY.value = event.clientY
    if (!isBrushPainting.value || !isActive.value) return
    const position = getBrushPositionInImage(event, activeSurface, true)
    if (position) {
      const previous = brushPath.value[brushPath.value.length - 1]!
      if (brushShape.value !== 'brush') {
        if (brushShape.value === 'rectangle') brushPath.value.splice(1, 1, position)
        else if (Math.hypot(position.x - previous.x, position.y - previous.y) * position.scale >= 2) brushPath.value.push(position)
        drawSelection()
        return
      }
      const distance = Math.hypot(position.x - previous.x, position.y - previous.y)
      const steps = Math.ceil(distance / (brushSize.value / position.scale / 4))
      for (let step = 1; step <= steps; step++) {
        const sample = {
          x: previous.x + (position.x - previous.x) * step / steps,
          y: previous.y + (position.y - previous.y) * step / steps,
          scale: position.scale,
        }
        brushPath.value.push(sample)
        drawBrushStroke(sample)
      }
    }
  }

  function finishBrushPainting(): void {
    if (!isBrushPainting.value) return
    isBrushPainting.value = false
    const image = imageStore.currentImage
    const bounds = getBrushPathBounds()
    const mode = brushMode.value
    const width = Math.round(image?.width || activeSurface?.image.naturalWidth || 0)
    const height = Math.round(image?.height || activeSurface?.image.naturalHeight || 0)
    removeBrushCanvas()
    brushPath.value = []
    activeSurface = null
    if (
      !image
      || image.documentRevision === undefined
      || !bounds
      || !mode
      || width <= 0
      || height <= 0
    ) return

    isBrushSubmitting.value = true
    const submission = callbacks?.onMaskStroke?.(bounds) ?? submitMaskRepair(
      image.id,
      image.documentRevision,
      width,
      height,
      bounds,
      mode,
    )
    void submission
      .catch(error => {
        if (!isOwnerDisposed && imageStore.currentImage?.id === image.id) {
          showToast(error instanceof Error ? error.message : '画笔修复失败', 'error')
        }
      })
      .finally(() => {
        isBrushSubmitting.value = false
      })
  }

  async function submitMaskRepair(
    pageId: string,
    documentRevision: number,
    width: number,
    height: number,
    bounds: BrushBounds,
    mode: Exclude<BrushMode, null>,
  ): Promise<void> {
    await queuePageDocumentSave(
      pageId,
      documentRevision,
      bubbleStore.bubbles,
    )
    const current = imageStore.currentImage
    if (
      !current
      || current.id !== pageId
      || !current.chapterId
      || current.documentRevision === undefined
    ) {
      throw new Error('当前页面已切换')
    }
    const settings = callbacks?.getCurrentRepairSettings?.() || {
      inpaintMethod: TEXT_STYLE_DEFAULTS.inpaintMethod as InpaintMethod,
      fillColor: TEXT_STYLE_DEFAULTS.fillColor,
    }
    const method = mode === 'clone' ? 'clone' : mode === 'restore'
      ? 'restore_source'
      : settings.inpaintMethod
    const mask = await createBinaryMask(width, height, bounds)
    try {
      const accepted = await createMaskRepair(
        pageId,
        mask,
        method === 'clone'
          ? { baseRevision: current.documentRevision, method, cloneOffset: [...cloneOffset!] as [number, number] }
          : method === 'solid'
          ? {
              baseRevision: current.documentRevision,
              fillColor: settings.fillColor,
              method,
            }
          : {
              baseRevision: current.documentRevision,
              method,
            },
      )
      await waitForOperation(accepted.operationId, {
        signal: abortController.signal,
      })
    } finally {
      // Submission conflicts and failed accepted repairs both need a fresh revision.
      if (!isOwnerDisposed && imageStore.currentImage?.id === pageId) {
        const document = await getPageDocument(pageId, abortController.signal)
        if (!isOwnerDisposed && imageStore.currentImage?.id === pageId) {
          if (document.pageId !== pageId || document.chapterId !== current.chapterId) {
            // eslint-disable-next-line no-unsafe-finally -- Identity mismatch must prevent adoption even after a failed repair.
            throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
          }
          registerPageDocument(document)
        }
      }
    }
    if (isOwnerDisposed || imageStore.currentImage?.id !== pageId) return
    await callbacks?.onBrushComplete?.()
  }

  function getBrushPositionInImage(
    event: MouseEvent,
    surface: BrushSurface | null,
    allowOutsideImage = false,
  ): BrushPosition | null {
    if (!surface || !surface.image.naturalWidth || !surface.image.naturalHeight) return null
    const rect = surface.wrapper.getBoundingClientRect()
    const transform = window.getComputedStyle(surface.wrapper).transform
    let scale = 1
    if (transform && transform !== 'none') {
      scale = new DOMMatrix(transform).a
    }
    if (!Number.isFinite(scale) || scale <= 0) return null
    const x = (event.clientX - rect.left) / scale
    const y = (event.clientY - rect.top) / scale
    if (!allowOutsideImage && (
      x < 0
      || y < 0
      || x > surface.image.naturalWidth
      || y > surface.image.naturalHeight
    )) return null
    return { x, y, scale }
  }

  function getBrushPathBounds(): BrushBounds | null {
    if (brushPath.value.length === 0) return null
    const scale = brushPath.value[0]?.scale || 1
    const radius = brushSize.value / 2 / scale
    if (brushShape.value !== 'brush' && selectionPolygon().length < 3) return null
    return {
      path: [...brushPath.value],
      radius,
      ...(brushShape.value !== 'brush' ? { polygon: selectionPolygon() } : {}),
    }
  }

  function createBrushCanvas(surface: BrushSurface): void {
    removeBrushCanvas()
    const canvas = document.createElement('canvas')
    const naturalWidth = surface.image.naturalWidth
    const naturalHeight = surface.image.naturalHeight
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1)
    const previewScale = Math.min(
      1,
      (surface.viewport.clientWidth * pixelRatio) / naturalWidth,
      (surface.viewport.clientHeight * pixelRatio) / naturalHeight,
    )
    canvas.width = Math.max(1, Math.round(naturalWidth * previewScale))
    canvas.height = Math.max(1, Math.round(naturalHeight * previewScale))
    brushCanvasScaleX = canvas.width / naturalWidth
    brushCanvasScaleY = canvas.height / naturalHeight
    canvas.setAttribute('aria-hidden', 'true')
    Object.assign(canvas.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: 'var(--z-canvas)',
    })
    surface.wrapper.appendChild(canvas)
    brushCanvas = canvas
    brushCtx = canvas.getContext('2d')
  }

  function removeBrushCanvas(): void {
    brushCanvas?.remove()
    brushCanvas = null
    brushCtx = null
    brushCanvasScaleX = 1
    brushCanvasScaleY = 1
  }

  function drawBrushStroke(position: BrushPosition): void {
    if (brushShape.value !== 'brush') { drawSelection(); return }
    if (!brushCtx) return
    const radius = brushSize.value / 2 / (position.scale || 1)
    brushCtx.beginPath()
    brushCtx.ellipse(
      position.x * brushCanvasScaleX,
      position.y * brushCanvasScaleY,
      radius * brushCanvasScaleX,
      radius * brushCanvasScaleY,
      0,
      0,
      Math.PI * 2,
    )
    brushCtx.fillStyle = brushColor.value
    brushCtx.fill()
  }

  function selectionPolygon(): BrushPosition[] {
    const points = brushPath.value
    if (brushShape.value !== 'rectangle') return points.length >= 3 ? [...points] : []
    const a = points[0], b = points[1]
    return a && b && Math.abs(a.x - b.x) >= 1 && Math.abs(a.y - b.y) >= 1
      ? [a, { ...a, x: b.x }, b, { ...b, x: a.x }] : []
  }

  function drawSelection(): void {
    if (!brushCtx || !brushCanvas) return
    brushCtx.clearRect(0, 0, brushCanvas.width, brushCanvas.height)
    brushCtx.beginPath()
    const points = selectionPolygon()
    points.forEach((point, index) => brushCtx![index ? 'lineTo' : 'moveTo'](point.x * brushCanvasScaleX, point.y * brushCanvasScaleY))
    brushCtx.closePath()
    brushCtx.fillStyle = brushColor.value
    brushCtx.fill('evenodd')
    brushCtx.strokeStyle = brushColor.value
    brushCtx.stroke()
  }

  async function createBinaryMask(
    width: number,
    height: number,
    bounds: BrushBounds,
  ): Promise<Blob> {
    if (bounds.polygon) {
      const pixels = new Uint8Array(width * height)
      // The polygon was frozen on mouse-up before releasing the preview canvas.
      paintMaskPolygon(pixels, width, height, bounds.polygon, 255)
      if (!pixels.includes(255)) throw new Error('选区没有包含图片像素')
      return encodeGrayscaleMaskPng(width, height, pixels)
    }
    return encodeBinaryMaskPng(
      width,
      height,
      bounds.path.map(position => ({
        x: position.x,
        y: position.y,
        radius: bounds.radius,
      })),
    )
  }

  onUnmounted(() => {
    isOwnerDisposed = true
    abortController.abort()
    isBrushPainting.value = false
    brushMode.value = null
    isBrushSubmitting.value = false
    brushPath.value = []
    activeSurface = null
    removeBrushCanvas()
  })

  return {
    brushMode,
    cloneSource,
    brushShape,
    brushSize,
    isBrushSubmitting,
    isBrushPainting,
    mouseX,
    mouseY,
    exitBrushMode,
    toggleBrushMode,
    adjustBrushSize,
    startBrushPainting,
    continueBrushPainting,
    finishBrushPainting,
  }
}
