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
  x1: number
  y1: number
  x2: number
  y2: number
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
  const brushPath = ref<BrushPosition[]>([])
  const mouseX = ref(0)
  const mouseY = ref(0)
  const abortController = new AbortController()
  let activeSurface: BrushSurface | null = null
  let brushCanvas: HTMLCanvasElement | null = null
  let brushCtx: CanvasRenderingContext2D | null = null
  let isOwnerDisposed = false

  const isActive = computed(() => brushMode.value !== null)
  const brushColor = computed(() => {
    if (brushMode.value === 'repair') {
      return { fill: 'rgba(76, 175, 80, 0.4)', border: '#4CAF50' }
    }
    if (brushMode.value === 'restore') {
      return { fill: 'rgba(33, 150, 243, 0.4)', border: '#2196F3' }
    }
    return { fill: 'transparent', border: 'transparent' }
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
    if (!isActive.value || event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    isBrushPainting.value = true
    activeSurface = surface
    brushPath.value = []
    const position = getBrushPositionInImage(event, surface)
    if (position) {
      brushPath.value.push(position)
      createBrushCanvas(surface)
      drawBrushStroke(position)
    }
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
    const width = image?.width || activeSurface?.image.naturalWidth || 0
    const height = image?.height || activeSurface?.image.naturalHeight || 0
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

    void submitMaskRepair(
      image.id,
      image.documentRevision,
      width,
      height,
      bounds,
      mode,
    ).catch(error => {
      if (!isOwnerDisposed && imageStore.currentImage?.id === image.id) {
        showToast(error instanceof Error ? error.message : '画笔修复失败', 'error')
      }
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
    const accepted = await createMaskRepair(pageId, mask, {
      baseRevision: current.documentRevision,
      fillColor: mode === 'restore' ? undefined : settings.fillColor,
      method,
    })
    await waitForOperation(accepted.operationId, {
      signal: abortController.signal,
    })
    if (isOwnerDisposed || imageStore.currentImage?.id !== pageId) return
    const document = await getPageDocument(pageId, abortController.signal)
    if (isOwnerDisposed || imageStore.currentImage?.id !== pageId) return
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
    if (!surface || !surface.image.naturalWidth) return null
    const rect = surface.wrapper.getBoundingClientRect()
    const transform = window.getComputedStyle(surface.wrapper).transform
    let scale = 1
    if (transform && transform !== 'none') {
      scale = new DOMMatrix(transform).a
    }
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
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const position of brushPath.value) {
      minX = Math.min(minX, position.x - radius)
      minY = Math.min(minY, position.y - radius)
      maxX = Math.max(maxX, position.x + radius)
      maxY = Math.max(maxY, position.y + radius)
    }
    return {
      x1: Math.max(0, Math.floor(minX)),
      y1: Math.max(0, Math.floor(minY)),
      x2: Math.ceil(maxX),
      y2: Math.ceil(maxY),
      path: [...brushPath.value],
      radius,
    }
  }

  function createBrushCanvas(surface: BrushSurface): void {
    removeBrushCanvas()
    const canvas = document.createElement('canvas')
    canvas.width = surface.image.naturalWidth
    canvas.height = surface.image.naturalHeight
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
  }

  function drawBrushStroke(position: BrushPosition): void {
    if (!brushCtx) return
    const radius = brushSize.value / 2 / (position.scale || 1)
    brushCtx.beginPath()
    brushCtx.arc(position.x, position.y, radius, 0, Math.PI * 2)
    brushCtx.fillStyle = brushColor.value.fill
    brushCtx.fill()
  }

  async function createBinaryMask(
    width: number,
    height: number,
    bounds: BrushBounds,
  ): Promise<Blob> {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) throw new Error('无法创建掩膜画布')
    context.fillStyle = 'black'
    context.fillRect(0, 0, width, height)
    context.fillStyle = 'white'
    for (const position of bounds.path) {
      context.beginPath()
      context.arc(position.x, position.y, bounds.radius, 0, Math.PI * 2)
      context.fill()
    }
    return new Promise((resolve, reject) => {
      canvas.toBlob(blob => {
        if (blob) resolve(blob)
        else reject(new Error('生成修复掩膜失败'))
      }, 'image/png')
    })
  }

  onUnmounted(() => {
    isOwnerDisposed = true
    abortController.abort()
    isBrushPainting.value = false
    brushMode.value = null
    isBrushKeyDown.value = false
    brushPath.value = []
    activeSurface = null
    removeBrushCanvas()
  })

  return {
    brushMode,
    brushSize,
    isBrushKeyDown,
    isBrushPainting,
    mouseX,
    mouseY,
    isActive,
    brushColor,
    BRUSH_MIN_SIZE,
    BRUSH_MAX_SIZE,
    enterBrushMode,
    exitBrushMode,
    toggleBrushMode,
    setBrushSize,
    adjustBrushSize,
    startBrushPainting,
    continueBrushPainting,
    finishBrushPainting,
  }
}
