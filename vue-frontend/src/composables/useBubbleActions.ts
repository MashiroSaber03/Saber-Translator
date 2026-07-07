import { onUnmounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { ocrSingleBubble as ocrSingleBubbleApi, inpaintSingleBubble as inpaintSingleBubbleApi } from '@/api/translate'
import { showToast } from '@/utils/toast'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import { buildSingleBubbleOcrRequest } from '@/composables/edit/singleBubbleOcrRequest'

export interface BubbleActionCallbacks {
  onReRender?: () => void | Promise<unknown>
  onDelayedPreview?: () => void | Promise<unknown>
}

// Backend OpenCV endpoints require integer coordinates.
function normalizeCoords(coords: BubbleCoords): BubbleCoords {
  return [
    Math.round(coords[0]),
    Math.round(coords[1]),
    Math.round(coords[2]),
    Math.round(coords[3])
  ]
}

export function useBubbleActions(callbacks?: BubbleActionCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  const {
    bubbles,
    selectedIndex: selectedBubbleIndex,
    hasSelection
  } = storeToRefs(bubbleStore)

  const { currentImage } = storeToRefs(imageStore)

  const isDrawingMode = ref(false)
  const isDrawingBox = ref(false)
  const currentDrawingRect = ref<BubbleCoords | null>(null)
  const isMiddleButtonDown = ref(false)

  function handleBubbleSelect(index: number): void {
    bubbleStore.selectBubble(index)
  }

  function handleBubbleMultiSelect(index: number): void {
    bubbleStore.toggleMultiSelect(index)
  }

  function handleClearMultiSelect(): void {
    bubbleStore.clearMultiSelect()
  }

  function handleBubbleDragStart(index: number, _event: MouseEvent): void {
    void index
  }

  function handleBubbleDragEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    triggerDelayedPreview()
  }

  function handleBubbleResizeStart(index: number, handle: string, _event: MouseEvent): void {
    void index
    void handle
  }

  function handleBubbleResizeEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    triggerDelayedPreview()
  }

  function handleBubbleRotateStart(index: number, _event: MouseEvent): void {
    void index
  }

  function handleBubbleRotateEnd(index: number, angle: number): void {
    bubbleStore.updateBubble(index, { rotationAngle: angle })
    triggerDelayedPreview()
  }

  function toggleDrawingMode(): void {
    isDrawingMode.value = !isDrawingMode.value
  }

  function handleDrawBubble(coords: BubbleCoords): void {
    bubbleStore.addBubble(coords)
    bubbleStore.selectBubble(bubbleStore.bubbleCount - 1)
    callbacks?.onReRender?.()
  }

  function getDrawingRectStyle(): Record<string, string> {
    if (!currentDrawingRect.value) return {}
    const [x1, y1, x2, y2] = currentDrawingRect.value
    return {
      position: 'absolute',
      left: `${Math.min(x1, x2)}px`,
      top: `${Math.min(y1, y2)}px`,
      width: `${Math.abs(x2 - x1)}px`,
      height: `${Math.abs(y2 - y1)}px`
    }
  }

  let previewTimer: ReturnType<typeof setTimeout> | null = null
  let isRenderingPreview = false
  let previewRequestedWhileRendering = false
  const PREVIEW_DELAY = 150

  // Coalesces rapid geometry/style edits and queues one more preview if a render is already running.
  function triggerDelayedPreview(): void {
    if (previewTimer) {
      clearTimeout(previewTimer)
    }
    previewTimer = setTimeout(async () => {
      previewTimer = null
      if (isRenderingPreview) {
        previewRequestedWhileRendering = true
        return
      }
      isRenderingPreview = true

      try {
        if (callbacks?.onDelayedPreview) {
          await callbacks.onDelayedPreview()
        } else if (callbacks?.onReRender) {
          await callbacks.onReRender()
        }
      } catch {
        showToast('预览渲染失败', 'error')
      } finally {
        isRenderingPreview = false
        if (previewRequestedWhileRendering) {
          previewRequestedWhileRendering = false
          triggerDelayedPreview()
        }
      }
    }, PREVIEW_DELAY)
  }

  function isSameCurrentImage(expectedImageId: string): boolean {
    return currentImage.value?.id === expectedImageId
  }

  function isSameBubbleTarget(expectedImageId: string, index: number, expectedBubble: BubbleState): boolean {
    return isSameCurrentImage(expectedImageId) && bubbles.value[index] === expectedBubble
  }

  function updateCurrentImageIfStillCurrent(
    expectedImageId: string,
    updates: Parameters<typeof imageStore.updateCurrentImage>[0]
  ): boolean {
    if (!isSameCurrentImage(expectedImageId)) {
      return false
    }
    imageStore.updateCurrentImage(updates)
    return true
  }

  onUnmounted(() => {
    if (previewTimer) {
      clearTimeout(previewTimer)
      previewTimer = null
    }
    isRenderingPreview = false
    previewRequestedWhileRendering = false
  })

  function handleBubbleUpdate(updates: Partial<BubbleState>): void {
    bubbleStore.updateSelectedBubble(updates)
    triggerDelayedPreview()
  }

  function deleteSelectedBubbles(): void {
    if (hasSelection.value) {
      bubbleStore.deleteSelected()
      callbacks?.onReRender?.()
    }
  }

  async function repairSelectedBubble(): Promise<void> {
    const index = selectedBubbleIndex.value
    if (index < 0) {
      showToast('请先选中要修复的气泡框', 'warning')
      return
    }

    const bubble = bubbles.value[index]
    const image = currentImage.value
    if (!bubble || !image?.originalDataURL) {
      showToast('无法修复背景：缺少气泡或图片数据', 'warning')
      return
    }
    const expectedImageId = image.id

    const inpaintMethod = bubble.inpaintMethod || 'solid'
    const fillColor = bubble.fillColor || '#FFFFFF'
    const rotationAngle = bubble.rotationAngle || 0

    try {
      let baseImageData: string
      if (image.cleanImageData) {
        baseImageData = image.cleanImageData
      } else {
        const match = image.originalDataURL.match(/^data:image\/[^;]+;base64,(.+)$/)
        baseImageData = match && match[1] ? match[1] : ''
        if (!baseImageData) {
          showToast('无法解析图像数据', 'error')
          return
        }
      }

      const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama'

      if (isLamaMethod) {
        const lamaModel = inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe'
        const coords = normalizeCoords(bubble.coords)

        const response = await inpaintSingleBubbleApi(
          baseImageData,
          coords,
          {
            bubbleAngle: rotationAngle,
            method: 'lama',
            lamaModel: lamaModel
          }
        )

        if (response.success && response.inpainted_image) {
          if (!updateCurrentImageIfStillCurrent(expectedImageId, { cleanImageData: response.inpainted_image })) {
            return
          }
          triggerDelayedPreview()
        } else {
          showToast('LAMA 修复失败，已使用纯色填充', 'warning')
          const applied = await fillBubbleWithColor(bubble.coords, fillColor, rotationAngle, expectedImageId)
          if (applied) {
            triggerDelayedPreview()
          }
        }
      } else {
        const applied = await fillBubbleWithColor(bubble.coords, fillColor, rotationAngle, expectedImageId)
        if (applied) {
          triggerDelayedPreview()
        }
      }
    } catch (error) {
      if (!isSameCurrentImage(expectedImageId)) {
        return
      }
      const errorMessage = error instanceof Error ? error.message : '背景修复失败'
      showToast(errorMessage, 'error')
    }
  }

  async function fillBubbleWithColor(
    coords: [number, number, number, number],
    fillColor: string,
    rotationAngle: number = 0,
    expectedImageId?: string
  ): Promise<boolean> {
    const image = currentImage.value
    if (!image) return false

    const [x1, y1, x2, y2] = coords

    let baseSrc: string
    if (image.cleanImageData) {
      baseSrc = 'data:image/png;base64,' + image.cleanImageData
    } else if (image.originalDataURL) {
      baseSrc = image.originalDataURL
    } else {
      showToast('无法找到基础图像用于填充', 'error')
      return false
    }

    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          const ctx = canvas.getContext('2d')
          if (!ctx) {
            resolve(false)
            return
          }

          ctx.drawImage(img, 0, 0)
          ctx.fillStyle = fillColor

          if (Math.abs(rotationAngle) < 0.1) {
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
          } else {
            const cx = (x1 + x2) / 2
            const cy = (y1 + y2) / 2
            const hw = (x2 - x1) / 2
            const hh = (y2 - y1) / 2
            const rad = rotationAngle * Math.PI / 180
            const cos_a = Math.cos(rad)
            const sin_a = Math.sin(rad)

            const corners: [number, number][] = [
              [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
            ].map(([dx, dy]): [number, number] => [
              cx + (dx as number) * cos_a - (dy as number) * sin_a,
              cy + (dx as number) * sin_a + (dy as number) * cos_a
            ])

            ctx.beginPath()
            const firstCorner = corners[0]
            if (firstCorner) {
              ctx.moveTo(firstCorner[0], firstCorner[1])
              for (let i = 1; i < corners.length; i++) {
                const corner = corners[i]
                if (corner) {
                  ctx.lineTo(corner[0], corner[1])
                }
              }
            }
            ctx.closePath()
            ctx.fill()
          }

          const newCleanData = canvas.toDataURL('image/png').split(',')[1]
          if (expectedImageId) {
            if (!updateCurrentImageIfStillCurrent(expectedImageId, { cleanImageData: newCleanData })) {
              resolve(false)
              return
            }
          } else {
            imageStore.updateCurrentImage({ cleanImageData: newCleanData })
          }

          resolve(true)
        } catch (error) {
          reject(error)
        }
      }
      img.onerror = () => {
        showToast('加载基础图像失败', 'error')
        resolve(false)
      }
      img.src = baseSrc
    })
  }

  async function handleOcrRecognize(index: number): Promise<void> {
    const bubble = bubbles.value[index]
    const image = currentImage.value
    if (!bubble || !image?.originalDataURL) {
      showToast('无法进行 OCR 识别：缺少气泡或图片数据', 'warning')
      return
    }
    const expectedImageId = image.id
    const expectedBubble = bubble

    try {
      const ocrRequest = buildSingleBubbleOcrRequest({
        image,
        bubble,
        bubbleIndex: index,
        settings: settingsStore.settings,
      })
      const response = await ocrSingleBubbleApi(
        ocrRequest.imageData,
        ocrRequest.bubbleCoords,
        ocrRequest.ocrEngine,
        ocrRequest.options,
      )

      if (response.success && response.text !== undefined) {
        if (!isSameBubbleTarget(expectedImageId, index, expectedBubble)) {
          return
        }
        bubbleStore.updateBubble(index, {
          originalText: response.text,
          textlines: response.textlines || ocrRequest.bubbleTextlines,
          ocrResult: response.ocr_result || null
        })
      } else {
        if (!isSameCurrentImage(expectedImageId)) {
          return
        }
        const errorMsg = response.error || '识别失败'
        showToast(errorMsg, 'error')
      }
    } catch (error) {
      if (!isSameCurrentImage(expectedImageId)) {
        return
      }
      const errorMessage = error instanceof Error ? error.message : 'OCR 识别出错'
      showToast(errorMessage, 'error')
    }
  }

  return {
    isDrawingMode,
    isDrawingBox,
    currentDrawingRect,
    isMiddleButtonDown,
    handleBubbleSelect,
    handleBubbleMultiSelect,
    handleClearMultiSelect,
    handleBubbleDragStart,
    handleBubbleDragEnd,
    handleBubbleResizeStart,
    handleBubbleResizeEnd,
    handleBubbleRotateStart,
    handleBubbleRotateEnd,
    toggleDrawingMode,
    handleDrawBubble,
    getDrawingRectStyle,
    handleBubbleUpdate,
    deleteSelectedBubbles,
    repairSelectedBubble,
    triggerDelayedPreview,
    handleOcrRecognize
  }
}
