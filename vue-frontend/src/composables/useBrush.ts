import { computed, onUnmounted, ref } from 'vue'
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
import { encodeBinaryMaskPng } from '@/utils/binaryMaskPng'
import { showToast } from '@/utils/toast'

export type BrushMode = 'repair' | 'restore' | null

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
}

export interface CurrentRepairSettings {
  inpaintMethod: InpaintMethod
  fillColor: string
}

export interface BrushCallbacks {
  onBrushComplete?: () => void | Promise<unknown>
  getCurrentRepairSettings?: () => CurrentRepairSettings
}

export function useBrush(callbacks?: BrushCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const brushMode = ref<BrushMode>(null)
  const brushSize = ref(BRUSH_DEFAULT_SIZE)
  const isBrushKeyDown = ref(false)
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
    if (brushMode.value === 'repair') {
      return 'rgba(76, 175, 80, 0.4)'
    }
    if (brushMode.value === 'restore') {
      return 'rgba(33, 150, 243, 0.4)'
    }
    return 'transparent'
  })

  function enterBrushMode(mode: 'repair' | 'restore'): void {
    if (brushMode.value === mode) return
    brushMode.value = mode
    isBrushKeyDown.value = true
    brushPath.value = []
  }

  function exitBrushMode(): void {
    if (isBrushPainting.value) finishBrushPainting()
    brushMode.value = null
    isBrushKeyDown.value = false
    isBrushPainting.value = false
    brushPath.value = []
    activeSurface = null
    removeBrushCanvas()
  }

  function toggleBrushMode(mode: 'repair' | 'restore'): void {
    if (brushMode.value === mode) exitBrushMode()
    else enterBrushMode(mode)
  }

  function setBrushSize(size: number): void {
    brushSize.value = Math.max(BRUSH_MIN_SIZE, Math.min(BRUSH_MAX_SIZE, size))
  }

  function adjustBrushSize(delta: number): void {
    setBrushSize(brushSize.value + delta)
  }

  function startBrushPainting(event: MouseEvent, surface: BrushSurface): void {
    if (!isActive.value || isBrushSubmitting.value || event.button !== 0) return
    const position = getBrushPositionInImage(event, surface)
    if (!position) return
    event.preventDefault()
    event.stopPropagation()
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
    const position = getBrushPositionInImage(event, activeSurface)
    if (position) {
      brushPath.value.push(position)
      drawBrushStroke(position)
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
    const submission = submitMaskRepair(
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
    const method = mode === 'restore'
      ? 'restore_source'
      : settings.inpaintMethod
    const mask = await createBinaryMask(width, height, bounds)
    const accepted = await createMaskRepair(
      pageId,
      mask,
      method === 'solid'
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
    if (isOwnerDisposed || imageStore.currentImage?.id !== pageId) return
    const document = await getPageDocument(pageId, abortController.signal)
    if (isOwnerDisposed || imageStore.currentImage?.id !== pageId) return
    if (document.pageId !== pageId || document.chapterId !== current.chapterId) {
      throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
    }
    const bubbles = registerPageDocument(document)
    imageStore.updateCurrentImage({
      bubbleStates: bubbles,
      documentRevision: document.documentRevision,
      hasUnsavedChanges: false,
    })
    bubbleStore.setBubbles(bubbles, true)
    await callbacks?.onBrushComplete?.()
  }

  function getBrushPositionInImage(
    event: MouseEvent,
    surface: BrushSurface | null,
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
    if (
      x < 0
      || y < 0
      || x > surface.image.naturalWidth
      || y > surface.image.naturalHeight
    ) return null
    return { x, y, scale }
  }

  function getBrushPathBounds(): BrushBounds | null {
    if (brushPath.value.length === 0) return null
    const scale = brushPath.value[0]?.scale || 1
    const radius = brushSize.value / 2 / scale
    return {
      path: [...brushPath.value],
      radius,
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

  async function createBinaryMask(
    width: number,
    height: number,
    bounds: BrushBounds,
  ): Promise<Blob> {
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
    isBrushKeyDown.value = false
    isBrushSubmitting.value = false
    brushPath.value = []
    activeSurface = null
    removeBrushCanvas()
  })

  return {
    brushMode,
    brushSize,
    isBrushKeyDown,
    isBrushSubmitting,
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
