import { ref, computed, watch, onMounted, onUnmounted, onErrorCaptured, nextTick, type Ref } from 'vue'
import { storeToRefs } from 'pinia'
import { calculatePageTextFit, getPageDocument } from '@/api/v2/content'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageViewer } from '@/composables/useImageViewer'
import { useBrush, type BrushSurface } from '@/composables/useBrush'
import { useMaskEditor, type MaskTool } from '@/composables/edit/useMaskEditor'
import { usePageHistory } from '@/composables/edit/usePageHistory'
import { useBubbleActions } from '@/composables/useBubbleActions'
import { useEditRender } from '@/composables/useEditRender'
import { useEditWorkspaceKeyboardShortcuts } from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'
import { useImageColorPicker } from '@/composables/edit/useImageColorPicker'
import { useEditWorkspaceProcessingActions } from '@/composables/edit/useEditWorkspaceProcessingActions'
import { useEditWorkspaceResizeActions } from '@/composables/edit/useEditWorkspaceResizeActions'
import {
  flushPageDocument,
  queuePageDocumentMutation,
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { deepClone } from '@/utils/deepClone'
import { applyTextStyle, copiedTextStyle, type TextStyleTemplate } from './textStyleClipboard'
import { plainRun, formatBubbleText, type RunStyle } from './richTextDocument'
import { showToast } from '@/utils/toast'
import { isRequestCanceled } from '@/api/client'
import type EditImageComparison from './EditImageComparison.vue'
import { LAYOUT_MODE_KEY } from '@/constants'
import type { BubbleState, BubbleColorField, InpaintMethod } from '@/types/bubble'
import {
  TEXT_STYLE_DEFAULTS,
} from '@/defaults/textStyleDefaults'

export type EditWorkspaceEmit = {
  (e: 'exit'): void
}

export function useEditWorkspace(emit: EditWorkspaceEmit, ordinaryActive: Readonly<Ref<boolean>> = ref(true)) {
  const isTextFitting = ref(false)
  const isSearchReplacing = ref(false)
  const searchTarget = ref<{ pageId: string; bubbleId: string } | null>(null)
  const focusAfterNavigate = ref(false)
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()

  const {
    reRenderFullImage,
    isRendering: isPreviewRendering,
    renderError,
  } = useEditRender({
    onRenderError: message => showToast(message, 'error')
  })

  const {
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
    duplicateSelectedBubbles: duplicateBubbles,
    nudgeSelectedText: nudgeText,
    repairSelectedBubble: bubbleRepairSelectedBubble,
    handleOcrRecognize: bubbleOcrRecognize
  } = useBubbleActions({
    onReRender: () => isNavigationPending.value ? undefined : reRenderFullImage(),
  })

  const drawStartX = ref(0)
  const drawStartY = ref(0)

  const {
    brushMode,
    cloneSource,
    brushShape,
    brushSize,
    isBrushSubmitting,
    isBrushPainting,
    mouseX,
    mouseY,
    toggleBrushMode,
    exitBrushMode,
    startBrushPainting,
    continueBrushPainting,
    finishBrushPainting,
    adjustBrushSize
  } = useBrush({
    onBrushComplete: () => reRenderFullImage(),
    onMaskStroke: bounds => maskEditor.active.value ? maskEditor.stroke(bounds) : undefined,
    getBrushColor: () => maskEditor.active.value && maskEditor.tool.value === 'clear' ? 'rgba(128, 128, 128, 0.4)' : undefined,
    // 提供当前编辑面板的修复设置，不依赖气泡选中状态
    getCurrentRepairSettings: () => ({
      inpaintMethod: currentInpaintMethod.value,
      fillColor: currentFillColor.value
    })
  })

  const {
    images,
    currentImageIndex,
    currentImage,
    imageCount,
    canGoPrevious,
    canGoNext
  } = storeToRefs(imageStore)

  const {
    bubbles,
    selectedIndex: selectedBubbleIndex,
    selectedIndices,
    selectedBubble,
    bubbleCount,
    hasBubbles,
    hasSelection
  } = storeToRefs(bubbleStore)

  const currentImageWidth = computed(() => currentImage.value?.width || 0)

  const currentImageHeight = computed(() => currentImage.value?.height || 0)

  function updateImageDimensions(): void {
    const img = originalImageRef.value
    if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
      imageStore.updateCurrentImageDimensions(img.naturalWidth, img.naturalHeight)
    }
  }


  type EditImageComparisonExposed = InstanceType<typeof EditImageComparison> & {
    originalViewportRef: HTMLElement | null
    originalWrapperRef: HTMLElement | null
    originalImageRef: HTMLImageElement | null
    originalPanelRef: HTMLElement | null
    translatedViewportRef: HTMLElement | null
    translatedWrapperRef: HTMLElement | null
    translatedImageRef: HTMLImageElement | null
    translatedPanelRef: HTMLElement | null
    editPanelRef: HTMLElement | null
  }

  const workspaceRef = ref<HTMLElement | null>(null)
  const imageComparisonRef = ref<EditImageComparisonExposed | null>(null)

  function focusWorkspaceAfterPointer(): void {
    requestAnimationFrame(() => {
      workspaceRef.value?.focus({ preventScroll: true })
    })
  }
  const originalViewportRef = computed(() => imageComparisonRef.value?.originalViewportRef ?? null)
  const originalWrapperRef = computed(() => imageComparisonRef.value?.originalWrapperRef ?? null)
  const originalImageRef = computed(() => imageComparisonRef.value?.originalImageRef ?? null)
  const originalPanelRef = computed(() => imageComparisonRef.value?.originalPanelRef ?? null)
  const translatedViewportRef = computed(() => imageComparisonRef.value?.translatedViewportRef ?? null)
  const translatedWrapperRef = computed(() => imageComparisonRef.value?.translatedWrapperRef ?? null)
  const translatedImageRef = computed(() => imageComparisonRef.value?.translatedImageRef ?? null)
  const translatedPanelRef = computed(() => imageComparisonRef.value?.translatedPanelRef ?? null)
  const editPanelRef = computed(() => imageComparisonRef.value?.editPanelRef ?? null)


  const viewMode = ref<'dual' | 'original' | 'translated'>('dual')

  const layoutMode = ref<'horizontal' | 'vertical'>('horizontal')

  const showThumbnails = ref(false)

  const syncEnabled = ref(true)

  const {
    startDividerDrag,
    startPanelResize,
  } = useEditWorkspaceResizeActions({
    layoutMode,
    originalPanelRef,
    translatedPanelRef,
    editPanelRef,
  })

  // 独立修复设置由编辑工作区持有，未选中气泡时仍能作为新气泡默认值。

  const currentInpaintMethod = ref<InpaintMethod>('solid')

  const currentFillColor = ref(TEXT_STYLE_DEFAULTS.fillColor)

  const isOcrLoading = ref(false)

  const isRepairLoading = ref(false)
  const isNavigationPending = ref(false)
  const isPageDocumentReady = ref(false)

  let layoutFitTimeout: ReturnType<typeof setTimeout> | null = null
  let initialFitTimeout: ReturnType<typeof setTimeout> | null = null
  let imageLoadFitTimeout: ReturnType<typeof setTimeout> | null = null
  let isOwnerActive = false
  let pendingScreenFit = false

  function clearDelayedFitTimers(): void {
    if (layoutFitTimeout) {
      clearTimeout(layoutFitTimeout)
      layoutFitTimeout = null
    }
    if (initialFitTimeout) {
      clearTimeout(initialFitTimeout)
      initialFitTimeout = null
    }
    if (imageLoadFitTimeout) {
      clearTimeout(imageLoadFitTimeout)
      imageLoadFitTimeout = null
    }
  }

  const originalViewer = useImageViewer({ minScale: 0.01 })
  const translatedViewer = useImageViewer({ minScale: 0.01 })
  const useOriginalViewport = computed(() => viewMode.value === 'original'
    || (viewMode.value === 'dual' && imageComparisonRef.value?.translatedPanelCollapsed))
  const primaryViewer = computed(() => useOriginalViewport.value ? originalViewer : translatedViewer)
  const toolbarScale = computed(() => primaryViewer.value.scale.value)

  const scale = computed(() => translatedViewer.scale.value)

  const originalScale = computed(() => originalViewer.scale.value)

  const activeViewport = ref<'original' | 'translated' | null>(null)

  const originalTransformStyle = computed(() => ({
    transform: `translate(${originalViewer.translateX.value}px, ${originalViewer.translateY.value}px) scale(${originalViewer.scale.value})`
  }))

  const translatedTransformStyle = computed(() => ({
    transform: `translate(${translatedViewer.translateX.value}px, ${translatedViewer.translateY.value}px) scale(${translatedViewer.scale.value})`
  }))

  function currentViewportCenter(): { x: number; y: number } {
    const viewport = useOriginalViewport.value ? originalViewportRef.value : translatedViewportRef.value
    return {
      x: (viewport?.clientWidth ?? 0) / 2,
      y: (viewport?.clientHeight ?? 0) / 2,
    }
  }

  function syncViewerTransform(viewer: ReturnType<typeof useImageViewer>): void {
    if (syncEnabled.value) {
      const otherViewer = viewer === originalViewer ? translatedViewer : originalViewer
      otherViewer.setTransform(viewer.getTransform())
    }
  }

  function zoomIn(): void {
    const center = currentViewportCenter()
    primaryViewer.value.zoomIn(center.x, center.y)
    syncViewerTransform(primaryViewer.value)
  }

  function zoomOut(): void {
    const center = currentViewportCenter()
    primaryViewer.value.zoomOut(center.x, center.y)
    syncViewerTransform(primaryViewer.value)
  }

  function resetZoom(): void {
    primaryViewer.value.resetZoom()
    syncViewerTransform(primaryViewer.value)
  }

  async function persistCurrentDocument(): Promise<void> {
    const image = currentImage.value
    if (!image || image.documentRevision === undefined) return
    await Promise.all([
      queuePageDocumentSave(image.id, image.documentRevision, bubbles.value),
      flushPageDocument(image.id),
    ])
  }

  async function navigateAfterPersist(navigate: () => unknown): Promise<void> {
    if (isBusy.value || brushMode.value) return
    if (maskEditor.dirty.value) {
      showToast('请先保存当前修复范围，或放弃未保存的修改', 'warning')
      return
    }
    isNavigationPending.value = true
    try {
      await persistCurrentDocument()
      await navigate()
    } catch (error) {
      showToast(
        error instanceof Error ? error.message : '保存当前页失败，操作未完成',
        'error',
      )
    } finally {
      isNavigationPending.value = false
    }
  }

  function selectFirstBubbleIfExists(): void {
    if (!bubbleStore.selectedBubble && bubbleStore.bubbles.length > 0) {
      bubbleStore.selectBubble(0)
    }
  }

  async function goToPreviousImage(): Promise<void> {
    if (canGoPrevious.value) {
      await navigateAfterPersist(() => imageStore.goToPrevious())
    }
  }

  async function goToNextImage(): Promise<void> {
    if (canGoNext.value) {
      await navigateAfterPersist(() => imageStore.goToNext())
    }
  }

  async function switchToImage(index: number): Promise<void> {
    if (index !== currentImageIndex.value && index >= 0 && index < imageCount.value) {
      await navigateAfterPersist(() => imageStore.setCurrentImageIndex(index))
    }
  }

  let pageDocumentRequest = 0
  let pageDocumentAbortController: AbortController | null = null

  function loadBubbleStatesFromImage(): void {
    pageDocumentRequest += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    isPageDocumentReady.value = false
    bubbleStore.clearSelection()
    selectFirstBubbleIfExists()
    const pageId = currentImage.value?.id
    const chapterId = currentImage.value?.chapterId
    if (!pageId || !chapterId) return
    pageDocumentAbortController = new AbortController()
    const request = pageDocumentRequest
    void getPageDocument(pageId, pageDocumentAbortController.signal)
      .then(document => {
        if (
          request !== pageDocumentRequest
          || currentImage.value?.id !== pageId
        ) return
        if (document.pageId !== pageId || document.chapterId !== chapterId) {
          throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
        }
        registerPageDocument(document)
        if (bubbles.value.length === 0 && currentImage.value) {
          currentInpaintMethod.value = currentImage.value.inpaintMethod
          currentFillColor.value = currentImage.value.fillColor
        }
        handleDocumentLoaded(pageId)
      })
      .catch(error => {
        if (isRequestCanceled(error)) return
        if (request === pageDocumentRequest) {
          showToast(
            error instanceof Error ? error.message : '加载当前页编辑数据失败',
            'error',
          )
        }
      })
  }

  function handleDocumentLoaded(pageId: string): void {
    if (currentImage.value?.id !== pageId) return
    selectFirstBubbleIfExists()
    isPageDocumentReady.value = true
  }

  const {
    isProcessing,
    isTranslateLoading,
    handleReTranslateBubble,
    autoDetectBubbles,
    detectAllImages,
    translateWithCurrentBubbles,
  } = useEditWorkspaceProcessingActions({
    images,
    currentImage,
    currentImageIndex,
    bubbles,
    selectFirstBubbleIfExists,
    onReRender: reRenderFullImage,
  })

  const maskEditor = useMaskEditor({
    getRepairSettings: () => ({ inpaintMethod: currentInpaintMethod.value, fillColor: currentFillColor.value }),
    onApplied: () => reRenderFullImage(),
  })

  const pageHistory = usePageHistory({
    pageId: computed(() => currentImage.value?.id),
    revision: computed(() => currentImage.value?.documentRevision),
    enabled: ordinaryActive,
    ready: isPageDocumentReady,
    dirty: computed(() => !!currentImage.value?.hasUnsavedChanges),
    onRestore: async document => {
      registerPageDocument(document)
      selectFirstBubbleIfExists()
      await reRenderFullImage()
    },
  })

  function undoPage(): void { if (!isBusy.value && !isBrushPainting.value && !maskEditor.dirty.value) void pageHistory.undo() }
  function redoPage(): void { if (!isBusy.value && !isBrushPainting.value && !maskEditor.dirty.value) void pageHistory.redo() }

  async function toggleMaskMode(): Promise<void> {
    if (isBusy.value || isBrushPainting.value) return
    if (maskEditor.active.value) {
      if (maskEditor.close()) {
        cancelBrushShortcut()
        exitBrushMode()
      }
      return
    }
    cancelColorPick()
    cancelBrushShortcut()
    exitBrushMode()
    isDrawingMode.value = false
    isNavigationPending.value = true
    try { await persistCurrentDocument() }
    catch (error) { showToast(error instanceof Error ? error.message : '页面保存失败', 'error'); return }
    finally { isNavigationPending.value = false }
    await maskEditor.open()
  }

  function setMaskTool(tool: MaskTool | null): void {
    if (!maskEditor.active.value || isBusy.value || isBrushPainting.value) return
    const wasActive = brushMode.value !== null && maskEditor.tool.value === tool
    cancelBrushShortcut()
    exitBrushMode()
    if (!tool || wasActive) return
    brushShape.value = 'brush'
    maskEditor.tool.value = tool
    toggleBrushMode(tool === 'preserve' ? 'restore' : 'repair')
  }

  const isBusy = computed(() => (
    !isPageDocumentReady.value
    || isProcessing.value
    || isOcrLoading.value
    || isRepairLoading.value
    || isTextFitting.value
    || isSearchReplacing.value
    || isBrushSubmitting.value
    || isNavigationPending.value
    || maskEditor.busy.value
    || pageHistory.busy.value
    || pageHistory.initializing.value
  ))

  const { colorPickField, isPickingColor, startColorPick, cancelColorPick, pickImageColor } = useImageColorPicker({
    pageId: computed(() => currentImage.value?.id),
    bubble: selectedBubble,
    bubbleIndex: selectedBubbleIndex,
    disabled: isBusy,
    onPick: (field, color) => {
      if (!imageComparisonRef.value?.applyPickedColor(field, color)) handleBubbleUpdateWithSync({ [field]: color })
    },
    onError: message => showToast(message, 'error'),
  })

  function startImageColorPick(field: BubbleColorField): void {
    if (isDrawingBox.value || !startColorPick(field)) return
    cancelBrushShortcut()
    exitBrushMode()
    isDrawingMode.value = false
    handleDragEnd()
    nextTick(() => {
      if (isOwnerActive && isPickingColor.value) workspaceRef.value?.focus({ preventScroll: true })
    })
  }

  function handleWorkspacePointerDown(event: PointerEvent): void {
    if (!(event.target instanceof Node)) return
    const inOriginal = originalViewportRef.value?.contains(event.target)
    const inTranslated = translatedViewportRef.value?.contains(event.target)
    if (!isPickingColor.value) {
      if (inOriginal || inTranslated) focusWorkspaceAfterPointer()
      return
    }
    // Let the cancel button complete its click before removing its DOM target.
    if (event.target instanceof Element && event.target.closest('.edit-workspace__color-pick-status')) return
    if (!inOriginal && !inTranslated) {
      cancelColorPick()
      return
    }
    // PointerEvent preserves fractional CSS coordinates at small zoom levels.
    // Cancelling it also suppresses the compatibility mousedown that would drag the image.
    event.preventDefault()
    event.stopPropagation()
    if (event.button === 0) {
      pickImageColor(inOriginal ? originalImageRef.value : translatedImageRef.value, event)
    }
  }

  const { handleKeyDown, handleShortcutCapture, handleKeyUp, cancelBrushShortcut } = useEditWorkspaceKeyboardShortcuts({
    enabled: ordinaryActive,
    isPickingColor,
    cancelColorPick,
    brushMode,
    hasSelection,
    exitEditMode: handleExitToolbarAction,
    deleteSelectedBubbles: deleteSelectedBubblesWhenIdle,
    duplicateSelectedBubbles,
    selectAllBubbles: () => { if (!isBusy.value && !isDrawingBox.value && !isDrawingMode.value) bubbleStore.selectAll() },
    nudgeSelectedText: moveSelectedText,
    goToPreviousImage,
    goToNextImage,
    applyAndNext,
    save: () => maskEditor.active.value ? maskEditor.retry() : reRenderFullImage(),
    selectBubble: (delta, focusText) => selectAdjacentBubble(delta, focusText),
    toggleBrushMode: activateBrush,
    exitBrushMode,
    zoomIn,
    zoomOut,
    resetZoom,
    maskActive: maskEditor.active,
    exitMaskMode: () => { void toggleMaskMode() },
    applyMask: maskEditor.apply,
    undo: undoPage,
    redo: redoPage,
  })

  watch(ordinaryActive, active => {
    if (active) {
      void nextTick(() => {
        // Failed image requests are not automatically retried when the network returns.
        for (const image of [originalImageRef.value, translatedImageRef.value]) {
          if (image?.complete && image.naturalWidth === 0) image.setAttribute('src', image.src)
        }
        if (pendingScreenFit) fitToScreen()
      })
      return
    }
    cancelBrushShortcut()
    exitBrushMode()
    cancelColorPick()
    isDrawingMode.value = false
  })


  function selectAdjacentBubble(delta: number, focusText = false): void {
    if (isBusy.value) return
    const index = selectedBubbleIndex.value + delta
    if (index < 0 || index >= bubbles.value.length) return
    bubbleStore.selectBubble(index)
    const bubble = selectedBubble.value
    const viewport = useOriginalViewport.value ? originalViewportRef.value : translatedViewportRef.value
    if (bubble && viewport) {
      const { width, height } = viewport.getBoundingClientRect()
      const [x1, y1, x2, y2] = bubble.coords
      const cx = (x1 + x2) / 2 + bubble.position.x, cy = (y1 + y2) / 2 + bubble.position.y
      const angle = bubble.rotationAngle * Math.PI / 180
      const rx = (Math.abs(Math.cos(angle)) * (x2 - x1) + Math.abs(Math.sin(angle)) * (y2 - y1)) / 2
      const ry = (Math.abs(Math.sin(angle)) * (x2 - x1) + Math.abs(Math.cos(angle)) * (y2 - y1)) / 2
      const viewer = primaryViewer.value, transform = viewer.getTransform(), zoom = transform.scale
      if (width > 0 && height > 0 && (transform.translateX + (cx - rx) * zoom < 16
          || transform.translateX + (cx + rx) * zoom > width - 16
          || transform.translateY + (cy - ry) * zoom < 16
          || transform.translateY + (cy + ry) * zoom > height - 16)) {
        viewer.setTransform({ ...transform, translateX: width / 2 - cx * zoom, translateY: height / 2 - cy * zoom })
        syncViewerTransform(viewer)
      }
    }
    if (focusText) nextTick(() => imageComparisonRef.value?.focusText())
  }

  const selectPreviousBubble = () => selectAdjacentBubble(-1)
  const selectNextBubble = () => selectAdjacentBubble(1)

  async function locateSearchResult(pageId: string, bubbleId: string): Promise<void> {
    if (isBusy.value || brushMode.value) return
    const index = images.value.findIndex(image => image.id === pageId)
    if (index < 0) return
    await switchToImage(index)
    if (currentImage.value?.id === pageId) searchTarget.value = { pageId, bubbleId }
  }
  watch([searchTarget, isBusy, focusAfterNavigate], () => {
    if (!isBusy.value && focusAfterNavigate.value) {
      focusAfterNavigate.value = false
      nextTick(() => imageComparisonRef.value?.focusText())
    }
    if (!searchTarget.value || isBusy.value || currentImage.value?.id !== searchTarget.value.pageId) return
    const index = bubbles.value.findIndex(bubble => bubble.backendBubbleId === searchTarget.value!.bubbleId)
    searchTarget.value = null
    if (index >= 0) selectAdjacentBubble(index - selectedBubbleIndex.value)
  }, { flush: 'post' })

  const saveStatus = computed(() => !isPageDocumentReady.value ? '正在加载…'
    : isOcrLoading.value ? '识别中…'
    : isTranslateLoading.value ? '翻译中…'
    : isProcessing.value ? '处理中…'
    : maskEditor.busy.value ? '处理范围…'
    : isBrushSubmitting.value || isRepairLoading.value ? '正在修图…'
    : isTextFitting.value ? '计算字号…'
    : isSearchReplacing.value ? '替换中…'
    : renderError.value ? (currentImage.value?.hasUnsavedChanges ? '保存失败' : '出图失败')
      : currentImage.value?.hasUnsavedChanges ? '正在保存…'
        : isPreviewRendering.value || (bubbles.value.some(bubble => bubble.translatedText.trim())
          && currentImage.value?.renderedRevision !== currentImage.value?.documentRevision) ? '出图中…' : '已自动保存')


  function toggleThumbnails(): void {
    showThumbnails.value = !showThumbnails.value
  }

  function toggleLayout(): void {
    layoutMode.value = layoutMode.value === 'horizontal' ? 'vertical' : 'horizontal'
    try {
      localStorage.setItem(LAYOUT_MODE_KEY, layoutMode.value)
    } catch {
      // 布局偏好不可写时继续使用当前会话内状态。
    }
    // 切换布局后等待过渡完成再适应屏幕。
    if (layoutFitTimeout) {
      clearTimeout(layoutFitTimeout)
    }
    layoutFitTimeout = setTimeout(() => {
      layoutFitTimeout = null
      fitToScreen()
    }, 300)
  }

  function toggleViewMode(): void {
    const modes: Array<'dual' | 'original' | 'translated'> = ['dual', 'original', 'translated']
    const currentIndex = modes.indexOf(viewMode.value)
    const nextMode = modes[(currentIndex + 1) % modes.length]
    if (nextMode) {
      viewMode.value = nextMode
    }
  }

  function toggleSync(): void {
    syncEnabled.value = !syncEnabled.value
    // 开启同步时，立即同步两个视口的变换状态
    syncViewerTransform(primaryViewer.value)
  }

  function fitToScreen(): void {
    pendingScreenFit = true
    const viewport = useOriginalViewport.value ? originalViewportRef.value : translatedViewportRef.value
    if (!viewport) return

    const img = useOriginalViewport.value ? originalImageRef.value : translatedImageRef.value
    if (!img || !img.naturalWidth || !img.naturalHeight) return

    const viewportRect = viewport.getBoundingClientRect()
    if (viewportRect.width <= 0 || viewportRect.height <= 0) return
    const scaleX = viewportRect.width / img.naturalWidth
    const scaleY = viewportRect.height / img.naturalHeight
    const fitPadding = 0.95
    const newScale = Math.min(scaleX, scaleY) * fitPadding
    if (!Number.isFinite(newScale) || newScale <= 0) return

    const newTranslateX = (viewportRect.width - img.naturalWidth * newScale) / 2
    const newTranslateY = (viewportRect.height - img.naturalHeight * newScale) / 2

    // 切换图片时两个视口都需要适应屏幕，无论 syncEnabled 状态。
    const transform = { scale: newScale, translateX: newTranslateX, translateY: newTranslateY }
    translatedViewer.setTransform(transform)
    originalViewer.setTransform(transform)
    pendingScreenFit = false
  }


  function handleWheel(event: WheelEvent, viewport: 'original' | 'translated'): void {
    if (isPickingColor.value) return
    if (isBrushPainting.value) return
    if (brushMode.value && brushShape.value === 'brush') {
      const delta = event.deltaY > 0 ? -5 : 5
      adjustBrushSize(delta)
      return
    }

    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
    const mouseX = event.clientX - rect.left
    const mouseY = event.clientY - rect.top

    const factor = event.deltaY > 0 ? 0.9 : 1.1

    const viewer = viewport === 'original' ? originalViewer : translatedViewer
    viewer.zoomAt(mouseX, mouseY, factor)

    syncViewerTransform(viewer)
  }

  function handleMouseDown(event: MouseEvent, viewport: 'original' | 'translated'): void {
    if (isPickingColor.value) return
    if (brushMode.value) {
      if (isBusy.value || (maskEditor.active.value && !maskEditor.overlay.value)) return
      const surface = getBrushSurface(viewport)
      if (surface) {
        startBrushPainting(event, surface)
      }
      return
    }

    if (event.button === 1) {
      if (!maskEditor.active.value && !isBusy.value && startDrawing(event, viewport)) {
        isMiddleButtonDown.value = true
        document.body.classList.add('middle-button-drawing')
      }
      event.preventDefault()
      return
    }

    if (isDrawingMode.value && !isBusy.value && event.button === 0) {
      startDrawing(event, viewport)
      event.preventDefault()
      return
    }

    if (event.button === 0) {
      if (
        event.target instanceof Element
        && event.target.closest('.bubble-overlay__highlight-box')
      ) {
        return
      }

      if (!event.shiftKey) {
        handleClearMultiSelect()
      }

      activeViewport.value = viewport
      const viewer = viewport === 'original' ? originalViewer : translatedViewer
      viewer.startDrag(event.clientX, event.clientY)

      document.addEventListener('mousemove', handleDragMove)
      document.addEventListener('mouseup', handleDragEnd)
      event.preventDefault()
    }
  }

  function getBrushSurface(viewport: 'original' | 'translated'): BrushSurface | null {
    const viewportEl = viewport === 'original' ? originalViewportRef.value : translatedViewportRef.value
    const wrapper = viewport === 'original' ? originalWrapperRef.value : translatedWrapperRef.value
    const image = viewport === 'original' ? originalImageRef.value : translatedImageRef.value

    if (!viewportEl || !wrapper || !image) {
      return null
    }

    return {
      viewport: viewportEl,
      wrapper,
      image,
    }
  }

  function handleDragMove(event: MouseEvent): void {
    if (!activeViewport.value) return

    const viewer = activeViewport.value === 'original' ? originalViewer : translatedViewer
    viewer.drag(event.clientX, event.clientY)

    syncViewerTransform(viewer)
  }

  function handleDragEnd(): void {
    if (activeViewport.value) {
      const viewer = activeViewport.value === 'original' ? originalViewer : translatedViewer
      viewer.endDrag()
    }
    activeViewport.value = null
    document.removeEventListener('mousemove', handleDragMove)
    document.removeEventListener('mouseup', handleDragEnd)
  }


  let drawingViewport: 'original' | 'translated' = 'translated'

  function drawingPoint(
    event: MouseEvent,
    viewport: 'original' | 'translated',
    clampToImage: boolean,
  ): { x: number; y: number } | null {
    const wrapper = viewport === 'original' ? originalWrapperRef.value : translatedWrapperRef.value
    const viewer = viewport === 'original' ? originalViewer : translatedViewer
    const width = currentImageWidth.value
    const height = currentImageHeight.value
    if (
      !wrapper
      || !Number.isFinite(width)
      || !Number.isFinite(height)
      || width <= 0
      || height <= 0
      || !Number.isFinite(viewer.scale.value)
      || viewer.scale.value <= 0
    ) return null

    const wrapperRect = wrapper.getBoundingClientRect()
    const rawX = (event.clientX - wrapperRect.left) / viewer.scale.value
    const rawY = (event.clientY - wrapperRect.top) / viewer.scale.value
    if (!clampToImage && (rawX < 0 || rawX > width || rawY < 0 || rawY > height)) {
      return null
    }
    return {
      x: Math.max(0, Math.min(width, rawX)),
      y: Math.max(0, Math.min(height, rawY)),
    }
  }

  function startDrawing(
    event: MouseEvent,
    viewport: 'original' | 'translated' = 'translated',
  ): boolean {
    const point = drawingPoint(event, viewport, false)
    if (!point) return false
    drawingViewport = viewport

    drawStartX.value = point.x
    drawStartY.value = point.y
    isDrawingBox.value = true
    currentDrawingRect.value = [point.x, point.y, point.x, point.y]

    document.addEventListener('mousemove', handleDrawingMove)
    document.addEventListener('mouseup', handleDrawingEnd)
    return true
  }

  function handleDrawingMove(event: MouseEvent): void {
    if (!isDrawingBox.value) return
    const point = drawingPoint(event, drawingViewport, true)
    if (!point) return

    currentDrawingRect.value = [
      Math.min(drawStartX.value, point.x),
      Math.min(drawStartY.value, point.y),
      Math.max(drawStartX.value, point.x),
      Math.max(drawStartY.value, point.y)
    ]
  }

  function handleDrawingEnd(): void {
    document.removeEventListener('mousemove', handleDrawingMove)
    document.removeEventListener('mouseup', handleDrawingEnd)

    const wasMiddleButton = isMiddleButtonDown.value
    document.body.classList.remove('middle-button-drawing')

    if (!isDrawingBox.value || !currentDrawingRect.value) {
      isDrawingBox.value = false
      currentDrawingRect.value = null
      isMiddleButtonDown.value = false
      return
    }

    const [x1, y1, x2, y2] = currentDrawingRect.value
    const width = x2 - x1
    const height = y2 - y1

    if (width > 10 && height > 10 && isPageDocumentReady.value) {
      handleDrawBubble(currentDrawingRect.value)
    }

    isDrawingBox.value = false
    currentDrawingRect.value = null
    isMiddleButtonDown.value = false

    if (!wasMiddleButton && isDrawingMode.value) {
      isDrawingMode.value = false
    }
  }

  function handleImageLoad(viewport: 'original' | 'translated'): void {
    if (viewport === 'original') {
      updateImageDimensions()
      nextTick(() => {
        if (!isOwnerActive) return
        if (imageLoadFitTimeout) {
          clearTimeout(imageLoadFitTimeout)
        }
        imageLoadFitTimeout = setTimeout(() => {
          imageLoadFitTimeout = null
          fitToScreen()
        }, 50)
      })
    }
  }

  async function handleExitToolbarAction(): Promise<void> {
    await navigateAfterPersist(async () => {
      // Unmounting cancels editor polling, so refresh the result before leaving.
      await reRenderFullImage()
      emit('exit')
    })
  }

  function handleBubbleUpdateWithSync(updates: Partial<BubbleState>, pageOnly = false): void {
    if (isBusy.value) return
    if (updates.inpaintMethod !== undefined) {
      currentInpaintMethod.value = updates.inpaintMethod
    }
    if (updates.fillColor !== undefined) {
      currentFillColor.value = updates.fillColor
    }

    if (selectedBubbleIndex.value >= 0 && !pageOnly) {
      const batchFields = ['textDirection', 'lineSpacing', 'inlineAlign', 'blockAlign', 'strokeEnabled', 'strokeColor', 'strokeWidth', 'rotationAngle', 'inpaintMethod', 'fillColor']
      if (selectedIndices.value.length > 1 && Object.keys(updates).every(field => batchFields.includes(field)))
        updateBubbles(selectedIndices.value.map(index => [index, updates]))
      else handleBubbleUpdate(updates)
    } else if (updates.inpaintMethod !== undefined || updates.fillColor !== undefined) {
      const image = currentImage.value
      if (!image || image.documentRevision === undefined) return
      const repairPatch = {
        ...(updates.inpaintMethod !== undefined ? { inpaintMethod: updates.inpaintMethod } : {}),
        ...(updates.fillColor !== undefined ? { fillColor: updates.fillColor } : {}),
      }
      imageStore.updateCurrentImage(repairPatch)
      void queuePageDocumentMutation(image.id, image.documentRevision, bubbles.value, {
        pageStyleDefaultsPatch: repairPatch,
      }).catch(error => showToast(error instanceof Error ? error.message : '修复设置保存失败', 'error'))
    }
  }

  function handleApplyStyleToAllBubbles(scope: 'selected' | 'page', template = copiedTextStyle.value as TextStyleTemplate | null): void {
    if (isBusy.value || !template) return
    const indices = scope === 'page' ? bubbles.value.map((_, i) => i)
      : selectedIndices.value.length ? selectedIndices.value : [selectedBubbleIndex.value]
    updateBubbles(indices.flatMap(index => bubbles.value[index]
      ? [[index, applyTextStyle(bubbles.value[index]!, template)] as [number, Partial<BubbleState>]] : []))
  }

  function handleFormatText(style: RunStyle, reset = false): void {
    if (isBusy.value) return
    const indices = selectedIndices.value.length ? selectedIndices.value : [selectedBubbleIndex.value]
    updateBubbles(indices.flatMap(index => bubbles.value[index]
      ? [[index, formatBubbleText(bubbles.value[index]!, reset ? plainRun(bubbles.value[index]!) : style, reset)] as [number, Partial<BubbleState>]] : []))
  }

  async function fitSelectedText(): Promise<void> {
    const image = currentImage.value
    if (!image || image.documentRevision === undefined || isBusy.value) return
    const indices = selectedIndices.value.length ? [...selectedIndices.value] : [selectedBubbleIndex.value]
    isTextFitting.value = true
    try {
      await queuePageDocumentSave(image.id, image.documentRevision, bubbles.value)
      const revision = currentImage.value?.documentRevision
      if (currentImage.value?.id !== image.id || revision === undefined) return
      const result = await calculatePageTextFit(image.id, {
        baseRevision: revision, bubbleIds: indices.flatMap(index => bubbles.value[index]?.backendBubbleId ?? []),
      })
      if (currentImage.value?.id !== image.id || currentImage.value.documentRevision !== revision) return
      updateBubbles(result.items.flatMap(item => {
        const index = bubbles.value.findIndex(bubble => bubble.backendBubbleId === item.bubbleId)
        const bubble = bubbles.value[index]
        if (!bubble) return []
        const ratio = item.fontSize / bubble.fontSize
        return [[index, bubble.richText ? { richText: { ...deepClone(bubble.richText), runs: bubble.richText.runs.map(run => ({ ...run, style: {
          ...run.style, size_pt: Number(run.style.size_pt ?? bubble.fontSize) * ratio,
          ...(run.style.leading_pt != null ? { leading_pt: Number(run.style.leading_pt) * ratio } : {}),
        } })) } } : { fontSize: item.fontSize }] as [number, Partial<BubbleState>]]
      }))
    } catch (error) { showToast(error instanceof Error ? error.message : '字号适配失败', 'error') }
    finally { isTextFitting.value = false }
  }

  function moveSelectedText(dx: number, dy: number): void {
    if (!isBusy.value && !isDrawingBox.value && !isDrawingMode.value) nudgeText(dx, dy)
  }

  async function handleOcrRecognize(index: number): Promise<void> {
    if (isBusy.value) return
    isOcrLoading.value = true
    try {
      await bubbleOcrRecognize(index)
    } finally {
      isOcrLoading.value = false
    }
  }

  async function handleRepairSelectedBubble(): Promise<void> {
    if (isBusy.value) return
    isRepairLoading.value = true
    try {
      await bubbleRepairSelectedBubble()
    } finally {
      isRepairLoading.value = false
    }
  }


  function activateBrush(mode: 'repair' | 'restore' | 'clone'): void {
    if (isBusy.value) return
    if (maskEditor.active.value) setMaskTool(mode === 'restore' ? 'preserve' : 'repair')
    else toggleBrushMode(mode)
  }

  function deleteSelectedBubblesWhenIdle(): void {
    if (isBusy.value) return
    deleteSelectedBubbles()
  }

  function duplicateSelectedBubbles(): void {
    if (isBusy.value || brushMode.value || maskEditor.active.value || isDrawingBox.value || isDrawingMode.value) return
    duplicateBubbles()
  }

  function handleGlobalMouseMove(event: MouseEvent): void {
    continueBrushPainting(event)
  }

  function handleGlobalMouseUp(): void {
    finishBrushPainting()
  }


  async function applyAndNext(focusText = false): Promise<void> {
    await navigateAfterPersist(() => {
      if (canGoNext.value) {
        imageStore.goToNext()
      } else {
        showToast('已应用当前图片', 'success')
      }
    })
    if (focusText === true) focusAfterNavigate.value = true
  }


  onErrorCaptured((err) => {
    const userMessage = err instanceof Error ? err.message : '操作失败，请重试'
    showToast(userMessage, 'error')

    return false
  })


  onMounted(() => {
    isOwnerActive = true
    try {
      const savedLayout = localStorage.getItem(LAYOUT_MODE_KEY)
      if (savedLayout === 'horizontal' || savedLayout === 'vertical') {
        layoutMode.value = savedLayout
      }
    } catch {
      // 布局偏好不可读时使用默认水平布局。
    }

    // 键盘快捷键需要在编辑工作区获得焦点之外仍然可用。
    // Capture workspace combinations before ProseMirror suppresses native Enter/Escape.
    // The handler leaves normal input, formatting and IME keys with the focused control.
    document.addEventListener('keydown', handleShortcutCapture, true)
    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', cancelBrushShortcut)
    document.addEventListener('mousemove', handleGlobalMouseMove)
    document.addEventListener('mouseup', handleGlobalMouseUp)

    loadBubbleStatesFromImage()
    nextTick(() => {
      if (!isOwnerActive) return
      workspaceRef.value?.focus()
      updateImageDimensions()
      initialFitTimeout = setTimeout(() => {
        initialFitTimeout = null
        fitToScreen()
      }, 100)
    })
  })

  onUnmounted(() => {
    isOwnerActive = false
    document.removeEventListener('keydown', handleShortcutCapture, true)
    document.removeEventListener('keydown', handleKeyDown)
    document.removeEventListener('keyup', handleKeyUp)
    window.removeEventListener('blur', cancelBrushShortcut)
    document.removeEventListener('mousemove', handleGlobalMouseMove)
    document.removeEventListener('mouseup', handleGlobalMouseUp)
    document.removeEventListener('mousemove', handleDrawingMove)
    document.removeEventListener('mouseup', handleDrawingEnd)
    document.removeEventListener('mousemove', handleDragMove)
    document.removeEventListener('mouseup', handleDragEnd)
    document.body.classList.remove('middle-button-drawing')
    pageDocumentRequest += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    clearDelayedFitTimers()
  })

  watch(currentImageIndex, () => {
    loadBubbleStatesFromImage()
  }, { flush: 'sync' })

  watch(selectedBubble, bubble => {
    if (bubble) {
      currentInpaintMethod.value = bubble.inpaintMethod
      currentFillColor.value = bubble.fillColor
    }
  }, { immediate: true })

  return {
    persistCurrentDocument,
    pageHistory,
    undoPage,
    redoPage,
    handleDocumentLoaded,
    currentInpaintMethod,
    currentFillColor,
    workspaceRef,
    colorPickField,
    isPickingColor,
    startImageColorPick,
    cancelColorPick,
    handleWorkspacePointerDown,
    imageComparisonRef,
    focusWorkspaceAfterPointer,
    images,
    currentImageIndex,
    currentImage,
    currentImageWidth,
    currentImageHeight,
    imageCount,
    canGoPrevious,
    canGoNext,
    bubbles,
    selectedBubbleIndex,
    selectedBubble,
    selectedIndices,
    hasBubbles,
    bubbleCount,
    hasSelection,
    viewMode,
    layoutMode,
    showThumbnails,
    syncEnabled,
    isOcrLoading,
    isTranslateLoading,
    isRepairLoading,
    scale,
    originalScale,
    toolbarScale,
    originalTransformStyle,
    translatedTransformStyle,
    isDrawingMode,
    currentDrawingRect,
    handleBubbleSelect,
    handleBubbleMultiSelect,
    handleBubbleDragEnd,
    handleBubbleResizeEnd,
    handleBubbleRotateEnd,
    toggleDrawingMode,
    getDrawingRectStyle,
    deleteSelectedBubbles: deleteSelectedBubblesWhenIdle,
    duplicateSelectedBubbles,
    brushMode,
    cloneSource,
    brushShape,
    brushSize,
    mouseX,
    mouseY,
    isBusy,
    startDividerDrag,
    startPanelResize,
    zoomIn,
    zoomOut,
    resetZoom,
    goToPreviousImage,
    goToNextImage,
    switchToImage,
    selectPreviousBubble,
    selectNextBubble,
    toggleThumbnails,
    toggleLayout,
    toggleViewMode,
    toggleSync,
    fitToScreen,
    handleWheel,
    handleMouseDown,
    handleImageLoad,
    handleApplyStyleToAllBubbles,
    handleExitToolbarAction,
    handleBubbleUpdateWithSync,
    handleFormatText,
    fitSelectedText,
    isSearchReplacing,
    locateSearchResult,
    refreshAfterTextReplace: reRenderFullImage,
    moveSelectedText,
    saveStatus,
    renderError,
    retrySave: reRenderFullImage,
    handleOcrRecognize,
    handleReTranslateBubble,
    handleRepairSelectedBubble,
    activateRepairBrush: () => activateBrush('repair'),
    activateRestoreBrush: () => activateBrush('restore'),
    activateCloneBrush: () => activateBrush('clone'),
    maskEditor,
    toggleMaskMode,
    setMaskTool,
    applyAndNext,
    autoDetectBubbles,
    detectAllImages,
    translateWithCurrentBubbles,
  }
}
