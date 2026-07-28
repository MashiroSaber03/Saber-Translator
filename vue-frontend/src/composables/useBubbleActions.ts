import { getCurrentInstance, onUnmounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { getPageDocument } from '@/api/v2/content'
import {
  runBubbleRepair,
  runPageOperation,
} from '@/api/v2/operations'
import {
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { showToast } from '@/utils/toast'
import type { BubbleState, BubbleCoords } from '@/types/bubble'

export interface BubbleActionCallbacks {
  onReRender?: () => void | Promise<unknown>
  onDelayedPreview?: () => void | Promise<unknown>
}

export function useBubbleActions(callbacks?: BubbleActionCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()

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

  if (getCurrentInstance()) {
    onUnmounted(() => {
      if (previewTimer) {
        clearTimeout(previewTimer)
        previewTimer = null
      }
      isRenderingPreview = false
      previewRequestedWhileRendering = false
    })
  }

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
    if (!bubble || !image || image.documentRevision === undefined) {
      showToast('无法修复背景：页面尚未完成后端初始化', 'warning')
      return
    }

    try {
      await queuePageDocumentSave(
        image.id,
        image.documentRevision,
        bubbles.value,
      )
      const current = currentImage.value
      const bubbleId = bubble.backendBubbleId
      if (
        !current
        || current.id !== image.id
        || current.documentRevision === undefined
        || !bubbleId
      ) {
        throw new Error('气泡尚未完成后端持久化')
      }
      await runBubbleRepair(
        current.id,
        bubbleId,
        current.documentRevision,
      )
      await reloadPageDocument(current.id, bubbleId)
      await callbacks?.onReRender?.()
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '背景修复失败'
      showToast(errorMessage, 'error')
    }
  }

  async function handleOcrRecognize(index: number): Promise<void> {
    const bubble = bubbles.value[index]
    const image = currentImage.value
    if (!bubble || !image || image.documentRevision === undefined) {
      showToast('无法进行 OCR 识别：页面尚未完成后端初始化', 'warning')
      return
    }

    try {
      await queuePageDocumentSave(
        image.id,
        image.documentRevision,
        bubbles.value,
      )
      const current = currentImage.value
      const bubbleId = bubble.backendBubbleId
      if (
        !current
        || current.id !== image.id
        || current.documentRevision === undefined
        || !bubbleId
      ) {
        throw new Error('气泡尚未完成后端持久化')
      }
      await runPageOperation(current.id, {
        baseRevision: current.documentRevision,
        bubbleId,
        kind: 'bubble_ocr',
      })
      await reloadPageDocument(current.id, bubbleId)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'OCR 识别出错'
      showToast(errorMessage, 'error')
    }
  }

  async function reloadPageDocument(
    pageId: string,
    bubbleId?: string,
  ): Promise<void> {
    const document = await getPageDocument(pageId)
    if (currentImage.value?.id !== pageId) return
    const updated = registerPageDocument(document)
    imageStore.updateCurrentImage({
      bubbleStates: updated,
      documentRevision: document.documentRevision,
      hasUnsavedChanges: false,
    })
    bubbleStore.setBubbles(updated, true)
    const index = bubbleId
      ? updated.findIndex(item => item.backendBubbleId === bubbleId)
      : -1
    if (index >= 0) bubbleStore.selectBubble(index)
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
