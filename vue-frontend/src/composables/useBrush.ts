import { ref, computed, onUnmounted } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { BRUSH_MIN_SIZE, BRUSH_MAX_SIZE, BRUSH_DEFAULT_SIZE } from '@/constants'
import { inpaintSingleBubble } from '@/api/translate'
import { showToast } from '@/utils/toast'
import type { BubbleCoords, InpaintMethod } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { addErasureToUserMask, addRestorationToUserMask } from '@/utils/maskMerger'

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
  onBrushComplete?: () => void
  getCurrentRepairSettings?: () => CurrentRepairSettings
}

export function useBrush(callbacks?: BrushCallbacks) {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  const brushMode = ref<BrushMode>(null)
  const brushSize = ref(BRUSH_DEFAULT_SIZE)
  const isBrushKeyDown = ref(false)
  const isBrushPainting = ref(false)
  const brushPath = ref<BrushPosition[]>([])
  const mouseX = ref(0)
  const mouseY = ref(0)
  let activeSurface: BrushSurface | null = null
  let brushCanvas: HTMLCanvasElement | null = null
  let brushCtx: CanvasRenderingContext2D | null = null
  let isOwnerDisposed = false

  function isSameCurrentImage(expectedImageId: string): boolean {
    return imageStore.currentImage?.id === expectedImageId
  }

  function isOwnerActiveForImage(expectedImageId: string): boolean {
    return !isOwnerDisposed && isSameCurrentImage(expectedImageId)
  }

  function updateCurrentImageIfStillCurrent(
    expectedImageId: string,
    updates: Parameters<typeof imageStore.updateCurrentImage>[0]
  ): boolean {
    if (!isOwnerActiveForImage(expectedImageId)) {
      return false
    }
    imageStore.updateCurrentImage(updates)
    return true
  }

  const isActive = computed(() => brushMode.value !== null)

  const brushColor = computed(() => {
    if (brushMode.value === 'repair') {
      return {
        fill: 'rgba(76, 175, 80, 0.4)',
        border: '#4CAF50'
      }
    } else if (brushMode.value === 'restore') {
      return {
        fill: 'rgba(33, 150, 243, 0.4)',
        border: '#2196F3'
      }
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
    if (isBrushPainting.value) {
      finishBrushPainting()
    }

    brushMode.value = null
    isBrushKeyDown.value = false
    isBrushPainting.value = false
    brushPath.value = []
    activeSurface = null

    removeBrushCanvas()
  }

  function toggleBrushMode(mode: 'repair' | 'restore'): void {
    if (brushMode.value === mode) {
      exitBrushMode()
    } else {
      enterBrushMode(mode)
    }
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

    const pos = getBrushPositionInImage(event, surface)
    if (pos) {
      brushPath.value.push(pos)
      createBrushCanvas(surface)
      drawBrushStroke(pos)
    }
  }

  function continueBrushPainting(event: MouseEvent): void {
    mouseX.value = event.clientX
    mouseY.value = event.clientY

    if (!isBrushPainting.value || !isActive.value) return

    const pos = getBrushPositionInImage(event, activeSurface)
    if (pos) {
      brushPath.value.push(pos)
      drawBrushStroke(pos)
    }
  }

  function finishBrushPainting(): void {
    if (!isBrushPainting.value) return

    isBrushPainting.value = false

    const currentImage = imageStore.currentImage
    if (!currentImage || brushPath.value.length === 0) {
      removeBrushCanvas()
      brushPath.value = []
      activeSurface = null
      return
    }
    const expectedImageId = currentImage.id

    const bounds = getBrushPathBounds()
    if (!bounds) {
      removeBrushCanvas()
      brushPath.value = []
      activeSurface = null
      return
    }

    const mode = brushMode.value

    const executeAndRender = async () => {
      if (mode === 'restore') {
        await restoreBrushArea(currentImage, bounds, expectedImageId)
      } else if (mode === 'repair') {
        await repairBrushArea(currentImage, bounds, expectedImageId)
      }

      if (isOwnerActiveForImage(expectedImageId)) {
        callbacks?.onBrushComplete?.()
      }
    }

    void executeAndRender().catch(() => {
      if (isOwnerActiveForImage(expectedImageId)) {
        showToast('画笔修复失败', 'error')
      }
    })

    removeBrushCanvas()
    brushPath.value = []
    activeSurface = null
  }

  function getBrushPositionInImage(event: MouseEvent, surface: BrushSurface | null): BrushPosition | null {
    if (!surface || !surface.image.naturalWidth) return null

    const rect = surface.wrapper.getBoundingClientRect()
    const transform = window.getComputedStyle(surface.wrapper).transform
    let scale = 1

    if (transform && transform !== 'none') {
      const matrix = new DOMMatrix(transform)
      scale = matrix.a
    }

    const imgX = (event.clientX - rect.left) / scale
    const imgY = (event.clientY - rect.top) / scale

    const imgWidth = surface.image.naturalWidth
    const imgHeight = surface.image.naturalHeight

    if (imgX < 0 || imgY < 0 || imgX > imgWidth || imgY > imgHeight) {
      return null
    }

    return { x: imgX, y: imgY, scale }
  }

  function getBrushPathBounds(): BrushBounds | null {
    if (brushPath.value.length === 0) return null

    const firstPoint = brushPath.value[0]
    const scale = firstPoint?.scale || 1
    const radius = brushSize.value / 2 / scale

    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity

    for (const pos of brushPath.value) {
      minX = Math.min(minX, pos.x - radius)
      minY = Math.min(minY, pos.y - radius)
      maxX = Math.max(maxX, pos.x + radius)
      maxY = Math.max(maxY, pos.y + radius)
    }

    return {
      x1: Math.max(0, Math.floor(minX)),
      y1: Math.max(0, Math.floor(minY)),
      x2: Math.ceil(maxX),
      y2: Math.ceil(maxY),
      path: [...brushPath.value],
      radius
    }
  }

  function createBrushCanvas(surface: BrushSurface): void {
    removeBrushCanvas()

    const canvas = document.createElement('canvas')
    canvas.width = surface.image.naturalWidth
    canvas.height = surface.image.naturalHeight
    applyBrushCanvasContract(canvas)

    surface.wrapper.appendChild(canvas)
    brushCanvas = canvas
    brushCtx = canvas.getContext('2d')
  }

  function applyBrushCanvasContract(canvas: HTMLCanvasElement): void {
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
  }

  function removeBrushCanvas(): void {
    if (brushCanvas && brushCanvas.parentNode) {
      brushCanvas.parentNode.removeChild(brushCanvas)
    }
    brushCanvas = null
    brushCtx = null
  }

  function drawBrushStroke(pos: BrushPosition): void {
    if (!brushCtx || !pos) return

    const color = brushColor.value.fill
    const radius = brushSize.value / 2 / (pos.scale || 1)

    brushCtx.beginPath()
    brushCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2)
    brushCtx.fillStyle = color
    brushCtx.fill()
  }

  async function restoreBrushArea(currentImage: ImageData, bounds: BrushBounds, expectedImageId: string): Promise<void> {
    if (!currentImage.originalDataURL) return

    let cleanSrc: string
    if (currentImage.cleanImageData) {
      cleanSrc = 'data:image/png;base64,' + currentImage.cleanImageData
    } else {
      cleanSrc = currentImage.originalDataURL
    }

    return new Promise((resolve, reject) => {
      const cleanImg = new Image()
      const originalImg = new Image()
      let loadedCount = 0

      const onLoad = async () => {
        loadedCount++
        if (loadedCount < 2) return

        try {
          const canvas = document.createElement('canvas')
          canvas.width = cleanImg.naturalWidth
          canvas.height = cleanImg.naturalHeight
          const ctx = canvas.getContext('2d')
          if (!ctx) {
            resolve()
            return
          }

          ctx.drawImage(cleanImg, 0, 0)

          const maskCanvas = document.createElement('canvas')
          maskCanvas.width = canvas.width
          maskCanvas.height = canvas.height
          const maskCtx = maskCanvas.getContext('2d')
          if (!maskCtx) {
            resolve()
            return
          }

          maskCtx.fillStyle = 'white'
          for (const pos of bounds.path) {
            maskCtx.beginPath()
            maskCtx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2)
            maskCtx.fill()
          }

          ctx.globalCompositeOperation = 'destination-out'
          ctx.drawImage(maskCanvas, 0, 0)
          ctx.globalCompositeOperation = 'destination-over'
          ctx.drawImage(originalImg, 0, 0)
          ctx.globalCompositeOperation = 'source-over'

          const newUserMask = await addRestorationToUserMask(
            currentImage.userMask,
            canvas.width,
            canvas.height,
            bounds.path,
            bounds.radius
          )

          const newCleanImageData = canvas.toDataURL('image/png').split(',')[1]
          updateCurrentImageIfStillCurrent(expectedImageId, {
            cleanImageData: newCleanImageData,
            userMask: newUserMask
          })
          resolve()
        } catch (error) {
          reject(error)
        }
      }

      cleanImg.onload = onLoad
      cleanImg.onerror = () => resolve()
      originalImg.onload = onLoad
      originalImg.onerror = () => resolve()

      cleanImg.src = cleanSrc
      originalImg.src = currentImage.originalDataURL
    })
  }

  async function repairBrushArea(currentImage: ImageData, bounds: BrushBounds, expectedImageId: string): Promise<void> {
    const settings = callbacks?.getCurrentRepairSettings?.() || {
      inpaintMethod: TEXT_STYLE_DEFAULTS.inpaintMethod as InpaintMethod,
      fillColor: TEXT_STYLE_DEFAULTS.fillColor
    }
    const inpaintMethod = settings.inpaintMethod

    const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama'
    if (isLamaMethod) {
      await repairBrushAreaWithLama(currentImage, bounds, expectedImageId, inpaintMethod)
    } else {
      await repairBrushAreaWithColor(currentImage, bounds, expectedImageId, settings.fillColor)
    }
  }

  async function repairBrushAreaWithLama(
    currentImage: ImageData,
    bounds: BrushBounds,
    expectedImageId: string,
    inpaintMethod: 'lama_mpe' | 'litelama' = 'lama_mpe'
  ): Promise<void> {
    let baseImageData: string
    let baseImageSrc: string
    if (currentImage.cleanImageData) {
      baseImageData = currentImage.cleanImageData
      baseImageSrc = 'data:image/png;base64,' + currentImage.cleanImageData
    } else if (currentImage.originalDataURL) {
      baseImageData = currentImage.originalDataURL.split(',')[1]
      baseImageSrc = currentImage.originalDataURL
    } else {
      showToast('无法获取基础图像用于 LAMA 修复', 'error')
      return
    }

    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = async () => {
        if (!isOwnerActiveForImage(expectedImageId)) {
          resolve()
          return
        }

        const imgWidth = img.naturalWidth
        const imgHeight = img.naturalHeight

        const maskCanvas = document.createElement('canvas')
        maskCanvas.width = imgWidth
        maskCanvas.height = imgHeight
        const maskCtx = maskCanvas.getContext('2d')

        if (!maskCtx) {
          showToast('无法创建掩膜画布上下文', 'error')
          resolve()
          return
        }

        maskCtx.fillStyle = 'black'
        maskCtx.fillRect(0, 0, imgWidth, imgHeight)

        maskCtx.fillStyle = 'white'
        for (const pos of bounds.path) {
          maskCtx.beginPath()
          maskCtx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2)
          maskCtx.fill()
        }

        const maskDataUrl = maskCanvas.toDataURL('image/png')
        const maskBase64 = maskDataUrl.split(',')[1]

        const coords: BubbleCoords = [bounds.x1, bounds.y1, bounds.x2, bounds.y2]

        const lamaModel = inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe'

        try {
          showToast('LAMA 修复中...', 'info')

          const response = await inpaintSingleBubble(baseImageData, coords, {
            method: 'lama',
            lamaModel: lamaModel,
            maskData: maskBase64
          })

          if (!isOwnerActiveForImage(expectedImageId)) {
            resolve()
            return
          }

          if (response.success && response.inpainted_image) {
            const newUserMask = await addErasureToUserMask(
              currentImage.userMask,
              imgWidth,
              imgHeight,
              bounds.path,
              bounds.radius
            )

            if (!updateCurrentImageIfStillCurrent(expectedImageId, {
              cleanImageData: response.inpainted_image,
              userMask: newUserMask
            })) {
              resolve()
              return
            }
            showToast('LAMA 修复完成', 'success')
          } else {
            throw new Error(response.error || 'LAMA 修复返回无效数据')
          }
        } catch (error) {
          if (!isOwnerActiveForImage(expectedImageId)) {
            resolve()
            return
          }
          showToast('LAMA 修复失败，使用纯色填充', 'warning')
          const fallbackSettings = callbacks?.getCurrentRepairSettings?.()
          try {
            await repairBrushAreaWithColor(currentImage, bounds, expectedImageId, fallbackSettings?.fillColor)
          } catch (fallbackError) {
            reject(fallbackError)
            return
          }
        }
        resolve()
      }
      img.onerror = () => {
        if (isOwnerActiveForImage(expectedImageId)) {
          showToast('加载图像失败，无法进行 LAMA 修复', 'error')
        }
        resolve()
      }
      img.src = baseImageSrc
    })
  }

  async function repairBrushAreaWithColor(
    currentImage: ImageData,
    bounds: BrushBounds,
    expectedImageId: string,
    fillColor?: string
  ): Promise<void> {
    const color = fillColor || settingsStore.settings.textStyle.fillColor || '#FFFFFF'

    let cleanSrc: string
    if (currentImage.cleanImageData) {
      cleanSrc = 'data:image/png;base64,' + currentImage.cleanImageData
    } else if (currentImage.originalDataURL) {
      cleanSrc = currentImage.originalDataURL
    } else {
      return
    }

    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = async () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          const ctx = canvas.getContext('2d')
          if (!ctx) {
            resolve()
            return
          }

          ctx.drawImage(img, 0, 0)

          ctx.fillStyle = color
          for (const pos of bounds.path) {
            ctx.beginPath()
            ctx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2)
            ctx.fill()
          }

          const newUserMask = await addErasureToUserMask(
            currentImage.userMask,
            img.naturalWidth,
            img.naturalHeight,
            bounds.path,
            bounds.radius
          )

          const newCleanImageData = canvas.toDataURL('image/png').split(',')[1]
          updateCurrentImageIfStillCurrent(expectedImageId, {
            cleanImageData: newCleanImageData,
            userMask: newUserMask
          })
          resolve()
        } catch (error) {
          reject(error)
        }
      }
      img.onerror = () => resolve()
      img.src = cleanSrc
    })
  }

  onUnmounted(() => {
    isOwnerDisposed = true
    exitBrushMode()
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

    getBrushPositionInImage,
    getBrushPathBounds
  }
}
