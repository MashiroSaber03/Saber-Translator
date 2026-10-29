import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { getPageDocument } from '@/api/v2/content'
import {
  runBubbleRepair,
  runPageOperation,
} from '@/api/v2/operations'
import {
  queuePageDocumentSave,
  queuePageDocumentMutation,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { showToast } from '@/utils/toast'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import { buildDrawingRectStyle } from '@/components/edit/bubbleOverlayGeometry'
import { deepClone } from '@/utils/deepClone'

export interface BubbleActionCallbacks {
  onReRender?: () => void | Promise<unknown>
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

  function handleBubbleDragEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    void requestPreview()
  }

  function handleBubbleResizeEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    void requestPreview()
  }

  function handleBubbleRotateEnd(index: number, angle: number): void {
    bubbleStore.updateBubble(index, { rotationAngle: angle })
    void requestPreview()
  }

  function toggleDrawingMode(): void {
    isDrawingMode.value = !isDrawingMode.value
  }

  function handleDrawBubble(coords: BubbleCoords): void {
    bubbleStore.addBubble(coords)
    bubbleStore.selectBubble(bubbleStore.bubbleCount - 1)
    void requestPreview()
  }

  function getDrawingRectStyle(): Record<string, string> {
    return buildDrawingRectStyle(currentDrawingRect.value)
  }

  async function requestPreview(): Promise<void> {
    // Saving coalesces edits; rendering cancels obsolete observers. No second timer.
    try { await callbacks?.onReRender?.() }
    catch { showToast('预览渲染失败', 'error') }
  }

  function updateBubbles(patches: [number, Partial<BubbleState>][]): void {
    const fields: Record<string, string[]> = {}
    for (const [index, updates] of patches) {
      if (!bubbles.value[index]) continue
      bubbleStore.updateBubble(index, updates)
      const id = bubbles.value[index]?.backendBubbleId
      if (id) fields[id] = Object.keys(updates)
    }
    const image = currentImage.value
    if (image?.documentRevision !== undefined && Object.keys(fields).length) {
      void queuePageDocumentMutation(image.id, image.documentRevision, bubbles.value, {
        bubbleFields: fields,
      }).catch(() => { /* The queued preview reports the same save failure. */ })
    }
    void requestPreview()
  }

  function handleBubbleUpdate(updates: Partial<BubbleState>): void {
    updateBubbles([[selectedBubbleIndex.value, updates]])
  }

  function deleteSelectedBubbles(): void {
    if (hasSelection.value) {
      bubbleStore.deleteSelected()
      void requestPreview()
    }
  }

  function duplicateSelectedBubbles(): void {
    if (!currentImage.value) return
    const indices = bubbleStore.selectedIndices.length ? [...bubbleStore.selectedIndices].sort((a, b) => a - b) : [selectedBubbleIndex.value]
    const copies = indices.flatMap(index => {
      const bubble = bubbles.value[index]
      return bubble ? [deepClone(bubble)] : []
    })
    if (!copies.length) return
    const first = bubbles.value.length
    for (const copy of copies) {
      bubbleStore.addBubble(copy.coords.map(value => value + 20) as BubbleCoords, {
        ...copy, backendBubbleId: undefined, clientMutationId: undefined,
        duplicateSourceId: copy.backendBubbleId ?? copy.duplicateSourceId,
        originalText: '', textboxText: '', polygon: [], textlines: [], ocrResult: null,
        autoFgColor: null, autoBgColor: null, colorConfidence: 0,
      })
    }
    bubbleStore.selectBubble(first)
    for (let index = first + 1; index < bubbles.value.length; index++) bubbleStore.toggleMultiSelect(index)
    void requestPreview()
  }

  function nudgeSelectedText(dx: number, dy: number): void {
    const indices = bubbleStore.selectedIndices.length ? bubbleStore.selectedIndices : [selectedBubbleIndex.value]
    const fields: Record<string, string[]> = {}
    for (const index of indices) {
      const bubble = bubbles.value[index]
      if (!bubble) continue
      bubbleStore.updateBubble(index, { coords: bubble.coords.map((value, i) => value + (i % 2 ? dy : dx)) as BubbleCoords })
      if (bubble.backendBubbleId) fields[bubble.backendBubbleId] = ['coords']
    }
    const image = currentImage.value
    if (image?.documentRevision !== undefined && Object.keys(fields).length) {
      void queuePageDocumentMutation(image.id, image.documentRevision, bubbles.value, {
        bubbleFields: fields,
      }).catch(() => { /* The queued preview reports the same save failure. */ })
    }
    void requestPreview()
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
      try {
        await runBubbleRepair(
          current.id,
          bubbleId,
          current.documentRevision,
        )
      } finally {
        await reloadPageDocument(current.id, bubbleId)
      }
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
      if (imageStore.currentImage?.id === current.id) await callbacks?.onReRender?.()
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'OCR 识别出错'
      showToast(errorMessage, 'error')
    }
  }

  async function reloadPageDocument(
    pageId: string,
    bubbleId?: string,
  ): Promise<void> {
    const requested = currentImage.value
    if (!requested || requested.id !== pageId || !requested.chapterId) return
    const document = await getPageDocument(pageId)
    if (currentImage.value?.id !== pageId) return
    if (document.pageId !== pageId || document.chapterId !== requested.chapterId) {
      throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
    }
    const updated = registerPageDocument(document)
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
    handleBubbleDragEnd,
    handleBubbleResizeEnd,
    handleBubbleRotateEnd,
    toggleDrawingMode,
    handleDrawBubble,
    getDrawingRectStyle,
    handleBubbleUpdate,
    updateBubbles,
    deleteSelectedBubbles,
    duplicateSelectedBubbles,
    nudgeSelectedText,
    repairSelectedBubble,
    handleOcrRecognize
  }
}
