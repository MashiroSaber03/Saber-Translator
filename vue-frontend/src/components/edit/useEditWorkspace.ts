import { ref, computed, watch, onMounted, onUnmounted, onErrorCaptured, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useImageViewer } from '@/composables/useImageViewer'
import { useBrush } from '@/composables/useBrush'
import { useBubbleActions } from '@/composables/useBubbleActions'
import { useEditRender } from '@/composables/useEditRender'
import { useEditWorkspaceExit } from '@/composables/edit/useEditWorkspaceExit'
import { useEditWorkspaceKeyboardShortcuts } from '@/composables/edit/useEditWorkspaceKeyboardShortcuts'
import { useEditWorkspaceProcessingActions } from '@/composables/edit/useEditWorkspaceProcessingActions'
import { useEditWorkspaceResizeActions } from '@/composables/edit/useEditWorkspaceResizeActions'
import {
  forceInitializeBookshelfSession,
  isBookshelfSessionInitialized,
  saveBookshelfPageProgress
} from '@/composables/translation/core/saveStep'
import { showToast } from '@/utils/toast'
import type EditImageComparison from './EditImageComparison.vue'
import { LAYOUT_MODE_KEY } from '@/constants'
import type { BubbleState, InpaintMethod } from '@/types/bubble'

export interface EditWorkspaceProps {
  /** 编辑模式是否激活 */
  isEditModeActive: boolean
}

export type EditWorkspaceEmit = {
  /** 退出编辑模式 */
  (e: 'exit'): void
}

export function useEditWorkspace(props: EditWorkspaceProps, emit: EditWorkspaceEmit) {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const sessionStore = useSessionStore()

  // 使用编辑模式渲染 composable
  const {
    reRenderFullImage
  } = useEditRender({
    onRenderStart: () => console.log('开始重新渲染...'),
    onRenderSuccess: (url) => console.log('渲染成功:', url.substring(0, 50) + '...'),
    onRenderError: (err) => console.error('渲染失败:', err)
  })

  // 使用气泡操作 composable
  const {
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
    repairSelectedBubble: bubbleRepairSelectedBubble,
    handleOcrRecognize: bubbleOcrRecognize
  } = useBubbleActions({
    onReRender: () => reRenderFullImage(),
    onDelayedPreview: () => reRenderFullImage()  // 延迟预览也触发重新渲染
  })

  // 本地绘制辅助变量（用于坐标计算）
  const drawStartX = ref(0)
  const drawStartY = ref(0)

  // 使用笔刷 composable（传入渲染回调）
  const {
    brushMode,
    brushSize,
    mouseX,
    mouseY,
    isBrushKeyDown,
    toggleBrushMode,
    exitBrushMode,
    startBrushPainting,
    continueBrushPainting,
    finishBrushPainting,
    adjustBrushSize
  } = useBrush({
    onBrushComplete: () => reRenderFullImage(),
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

  const {
    isBookshelfMode,
    currentBookId,
    currentChapterId,
    isSaving: isSessionSaving,
    loadingProgress: sessionLoadingProgress,
    error: sessionSaveError,
  } = storeToRefs(sessionStore)

  /** 当前图片宽度（从 Store 响应式获取） */
  const currentImageWidth = computed(() => currentImage.value?.width || 0)

  /** 当前图片高度（从 Store 响应式获取） */
  const currentImageHeight = computed(() => currentImage.value?.height || 0)

  /** 更新当前图片尺寸（在图片加载完成时调用） */
  function updateImageDimensions(): void {
    const img = originalWrapperRef.value?.querySelector('img')
    if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
      imageStore.updateCurrentImageDimensions(img.naturalWidth, img.naturalHeight)
    }
  }

  // ============================================================
  // 模板引用
  // ============================================================

  type EditImageComparisonExposed = InstanceType<typeof EditImageComparison> & {
    originalViewportRef: HTMLElement | null
    originalWrapperRef: HTMLElement | null
    translatedViewportRef: HTMLElement | null
    translatedWrapperRef: HTMLElement | null
    editPanelRef: HTMLElement | null
  }

  const workspaceRef = ref<HTMLElement | null>(null)
  const imageComparisonRef = ref<EditImageComparisonExposed | null>(null)
  const originalViewportRef = computed(() => imageComparisonRef.value?.originalViewportRef ?? null)
  const originalWrapperRef = computed(() => imageComparisonRef.value?.originalWrapperRef ?? null)
  const translatedViewportRef = computed(() => imageComparisonRef.value?.translatedViewportRef ?? null)
  const translatedWrapperRef = computed(() => imageComparisonRef.value?.translatedWrapperRef ?? null)
  const editPanelRef = computed(() => imageComparisonRef.value?.editPanelRef ?? null)

  // ============================================================
  // 视图状态
  // ============================================================

  /** 视图模式: 'dual' | 'original' | 'translated' */
  const viewMode = ref<'dual' | 'original' | 'translated'>('dual')

  /** 布局模式: 'horizontal' | 'vertical' */
  const layoutMode = ref<'horizontal' | 'vertical'>('horizontal')

  /** 是否显示缩略图 */
  const showThumbnails = ref(false)

  /** 是否同步缩放/平移 */
  const syncEnabled = ref(true)

  const {
    startDividerDrag,
    handleDividerDrag,
    stopDividerDrag,
    startPanelResize,
    handlePanelResize,
    stopPanelResize,
  } = useEditWorkspaceResizeActions({
    layoutMode,
    originalViewportRef,
    editPanelRef,
  })

  // ============================================================
  // 独立的修复设置状态（不依赖气泡选中）
  // 对应业务契约 $('#bubbleInpaintMethodNew').val() 和 $('#fillColorNew').val()
  // ============================================================

  /** 当前编辑面板选择的修复方式 */
  const currentInpaintMethod = ref<InpaintMethod>('solid')

  /** 当前编辑面板选择的填充颜色 */
  const currentFillColor = ref('#FFFFFF')

  /** 单气泡 OCR 识别中 */
  const isOcrLoading = ref(false)

  /** 修复气泡背景中 */
  const isRepairLoading = ref(false)

  // ============================================================
  // 图片查看器状态
  // 【业务契约 DualImageViewer】支持两套独立变换状态，syncEnabled 开启时联动
  // ============================================================

  // 原图查看器
  const originalViewer = useImageViewer()
  // 翻译图查看器
  const translatedViewer = useImageViewer()

  // 主缩放比例（用于工具栏显示和统一的缩放操作）
  const scale = computed(() => translatedViewer.scale.value)
  const translateX = computed(() => translatedViewer.translateX.value)
  const translateY = computed(() => translatedViewer.translateY.value)

  // 原图视口的缩放比例（sync关闭时两个视口可能缩放不同）
  const originalScale = computed(() => originalViewer.scale.value)

  // 当前活动的视口（用于拖动时确定操作哪个视口）
  const activeViewport = ref<'original' | 'translated' | null>(null)

  /** 原图变换样式 */
  const originalTransformStyle = computed(() => ({
    transform: `translate(${originalViewer.translateX.value}px, ${originalViewer.translateY.value}px) scale(${originalViewer.scale.value})`
  }))

  /** 翻译图变换样式 */
  const translatedTransformStyle = computed(() => ({
    transform: `translate(${translatedViewer.translateX.value}px, ${translatedViewer.translateY.value}px) scale(${translatedViewer.scale.value})`
  }))

  /** 放大（两个视口同时） */
  function zoomIn(): void {
    translatedViewer.zoomIn()
    if (syncEnabled.value) {
      originalViewer.setTransform(translatedViewer.getTransform())
    }
  }

  /** 缩小（两个视口同时） */
  function zoomOut(): void {
    translatedViewer.zoomOut()
    if (syncEnabled.value) {
      originalViewer.setTransform(translatedViewer.getTransform())
    }
  }

  /** 重置缩放（两个视口同时） */
  function resetZoom(): void {
    translatedViewer.resetZoom()
    if (syncEnabled.value) {
      originalViewer.setTransform(translatedViewer.getTransform())
    }
  }

  // 绘制、气泡操作和笔刷状态由专用 composable 管理。


  // ============================================================
  // 图片导航方法
  // ============================================================

  /** 导航前的公共处理（业务逻辑） */
  function prepareForNavigation(): void {
    // 退出笔刷模式，调用exitBrushMode确保状态正确清理
    if (brushMode.value) {
      exitBrushMode()
    }
    if (exitDialogState.value !== 'saving') {
      closeExitDialog()
    }
    saveBubbleStatesToImage()
  }

  /** 选择第一个气泡（如果存在） */
  function selectFirstBubbleIfExists(): void {
    if (bubbleStore.bubbles.length > 0) {
      bubbleStore.selectBubble(0)
    }
  }

  /** 切换到上一张图片 */
  function goToPreviousImage(): void {
    if (canGoPrevious.value) {
      prepareForNavigation()
      imageStore.goToPrevious()
      // watch(currentImageIndex) 会自动触发 loadBubbleStatesFromImage
    }
  }

  /** 切换到下一张图片 */
  function goToNextImage(): void {
    if (canGoNext.value) {
      prepareForNavigation()
      imageStore.goToNext()
      // watch(currentImageIndex) 会自动触发 loadBubbleStatesFromImage
    }
  }

  /** 切换到指定图片 */
  function switchToImage(index: number): void {
    if (index !== currentImageIndex.value && index >= 0 && index < imageCount.value) {
      prepareForNavigation()
      imageStore.setCurrentImageIndex(index)
      // watch(currentImageIndex) 会自动触发 loadBubbleStatesFromImage
    }
  }

  /** 保存气泡状态到当前图片 */
  function saveBubbleStatesToImage(): void {
    if (!currentImage.value) return
    
    // 保持 null vs [] 语义区分：
    // - null/undefined：从未处理过
    // - []：处理过但用户删光了
    // 只要 currentImage.bubbleStates 曾经是数组（包括空数组），就应该保存当前状态
    const hadBubbleStates = Array.isArray(currentImage.value.bubbleStates)
    
    if (bubbles.value.length > 0) {
      // 有气泡，保存当前状态
      imageStore.updateCurrentBubbleStates([...bubbles.value])
      // 设置手动标注标记，使缩略图显示标记
      imageStore.setManuallyAnnotated(true)
      console.log('已保存气泡状态到当前图片，标记为手动标注')
    } else if (hadBubbleStates) {
      // 用户删光了气泡，保存空数组（保持"处理过"的语义）
      imageStore.updateCurrentBubbleStates([])
      // 删空也是手动操作，保持标记为 true，翻译时会跳过而不是重新检测
      imageStore.setManuallyAnnotated(true)
      console.log('已保存空气泡状态到当前图片（用户主动清空，标记为手动标注）')
    }
    // 如果 bubbleStates 从未是数组且当前也没有气泡，不做任何操作（保持 null 语义）
  }

  const {
    exitDialogState,
    exitDialogError,
    shouldPromptSaveOnExit,
    exitSaveCurrent,
    exitSaveTotal,
    exitSaveHasProgress,
    exitSaveProgressPercent,
    exitSaveMessage,
    closeExitDialog,
    openExitDialog,
    exitEditMode,
    exitWithoutSaving,
    saveAndExit,
  } = useEditWorkspaceExit({
    isBookshelfMode,
    currentBookId,
    currentChapterId,
    isSessionSaving,
    sessionLoadingProgress,
    sessionSaveError,
    saveBubbleStatesToImage,
    saveChapterSession: sessionStore.saveChapterSession,
    emitExit: () => emit('exit'),
  })

  const {
    handleKeyDown,
    handleKeyUp,
  } = useEditWorkspaceKeyboardShortcuts({
    exitDialogState,
    brushMode,
    hasSelection,
    isBrushKeyDown,
    closeExitDialog,
    exitEditMode,
    deleteSelectedBubbles,
    goToPreviousImage,
    goToNextImage,
    applyAndNext,
    toggleBrushMode,
    exitBrushMode,
    zoomIn,
    zoomOut,
    resetZoom,
  })

  /** 从当前图片加载气泡状态 */
  function loadBubbleStatesFromImage(): void {
    if (currentImage.value?.bubbleStates) {
      // skipSync=true 避免冗余同步（数据已经在 imageStore 中）
      bubbleStore.setBubbles([...currentImage.value.bubbleStates], true)
      console.log(`已加载 ${currentImage.value.bubbleStates.length} 个气泡状态`)
    } else {
      // 使用 clearBubblesLocal 仅清除本地状态，不同步到 imageStore
      // 这保持了 null（未处理）和 []（用户主动清空）的语义区分
      bubbleStore.clearBubblesLocal()
    }
    selectFirstBubbleIfExists()
    // 切图时保持当前缩放和位置，不自动 fitToScreen
  }

  const {
    isProcessing,
    progressText,
    progressCurrent,
    progressTotal,
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
    reRenderFullImage,
    loadBubbleStatesFromImage,
    selectFirstBubbleIfExists,
  })

  // ============================================================
  // 气泡导航方法
  // ============================================================

  /** 选择上一个气泡 */
  function selectPreviousBubble(): void {
    bubbleStore.selectPrevious()
    // selectBubbleNew() 刻意不滚动到气泡，避免画面跳动
  }

  /** 选择下一个气泡 */
  function selectNextBubble(): void {
    bubbleStore.selectNext()
    // selectBubbleNew() 刻意不滚动到气泡，避免画面跳动
  }

  // ============================================================
  // 视图控制方法
  // ============================================================

  /** 切换缩略图显示 */
  function toggleThumbnails(): void {
    showThumbnails.value = !showThumbnails.value
  }

  /** 切换布局模式 */
  function toggleLayout(): void {
    layoutMode.value = layoutMode.value === 'horizontal' ? 'vertical' : 'horizontal'
    // 保存到 localStorage
    try {
      localStorage.setItem(LAYOUT_MODE_KEY, layoutMode.value)
    } catch (e) {
      console.warn('保存布局模式失败:', e)
    }
    // 切换布局后延迟 300ms 自动适应屏幕
    setTimeout(() => {
      fitToScreen()
    }, 300)
  }

  /** 切换视图模式 */
  function toggleViewMode(): void {
    const modes: Array<'dual' | 'original' | 'translated'> = ['dual', 'original', 'translated']
    const currentIndex = modes.indexOf(viewMode.value)
    const nextMode = modes[(currentIndex + 1) % modes.length]
    if (nextMode) {
      viewMode.value = nextMode
    }
  }

  /** 切换同步状态 */
  function toggleSync(): void {
    syncEnabled.value = !syncEnabled.value
    console.log('双图同步:', syncEnabled.value ? '开启' : '关闭')
    // 开启同步时，立即同步两个视口的变换状态
    if (syncEnabled.value) {
      originalViewer.setTransform(translatedViewer.getTransform())
    }
  }

  /** 适应屏幕 */
  function fitToScreen(): void {
    const viewport = translatedViewportRef.value || originalViewportRef.value
    const wrapper = translatedWrapperRef.value || originalWrapperRef.value
    if (!viewport || !wrapper) return

    const img = wrapper.querySelector('img')
    if (!img || !img.naturalWidth) return

    const viewportRect = viewport.getBoundingClientRect()
    const scaleX = viewportRect.width / img.naturalWidth
    const scaleY = viewportRect.height / img.naturalHeight
    const newScale = Math.min(scaleX, scaleY) * 0.95 // 留5%边距

    // 居中
    const newTranslateX = (viewportRect.width - img.naturalWidth * newScale) / 2
    const newTranslateY = (viewportRect.height - img.naturalHeight * newScale) / 2

    // 切换图片时两个视口都需要适应屏幕，无论 syncEnabled 状态。
    const transform = { scale: newScale, translateX: newTranslateX, translateY: newTranslateY }
    translatedViewer.setTransform(transform)
    originalViewer.setTransform(transform)
  }

  // ============================================================
  // 鼠标事件处理
  // ============================================================

  /** 处理滚轮缩放 */
  function handleWheel(event: WheelEvent, viewport: 'original' | 'translated'): void {
    // 笔刷模式下调整笔刷大小
    if (brushMode.value) {
      const delta = event.deltaY > 0 ? -5 : 5
      adjustBrushSize(delta)
      return
    }

    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
    const mouseX = event.clientX - rect.left
    const mouseY = event.clientY - rect.top

    const factor = event.deltaY > 0 ? 0.9 : 1.1
    
    // 【业务契约 DualImageViewer】操作对应视口，同步时联动另一个
    const viewer = viewport === 'original' ? originalViewer : translatedViewer
    viewer.zoomAt(mouseX, mouseY, factor)
    
    if (syncEnabled.value) {
      const otherViewer = viewport === 'original' ? translatedViewer : originalViewer
      otherViewer.setTransform(viewer.getTransform())
    }
  }

  /** 处理鼠标按下 */
  function handleMouseDown(event: MouseEvent, viewport: 'original' | 'translated'): void {
    // 笔刷模式下开始涂抹
    if (brushMode.value) {
      const viewportEl = viewport === 'original' ? originalViewportRef.value : translatedViewportRef.value
      if (viewportEl) {
        startBrushPainting(event, viewportEl)
      }
      return
    }

    // 中键绘制新气泡
    if (event.button === 1) {
      isMiddleButtonDown.value = true
      startDrawing(event, viewport)
      event.preventDefault()
      return
    }

    // 绘制模式下左键绘制
    if (isDrawingMode.value && event.button === 0) {
      startDrawing(event, viewport)
      event.preventDefault()
      return
    }

    // 左键拖动
    if (event.button === 0) {
      // 检查是否点击了气泡高亮框
      if ((event.target as HTMLElement).closest('.bubble-highlight-box')) {
        return
      }
      
      // 点击空白处清除多选（非 Shift 时）
      if (!event.shiftKey) {
        handleClearMultiSelect()
      }
      
      // 记录当前操作的视口
      activeViewport.value = viewport
      const viewer = viewport === 'original' ? originalViewer : translatedViewer
      viewer.startDrag(event.clientX, event.clientY)
      
      // 添加全局事件监听
      document.addEventListener('mousemove', handleDragMove)
      document.addEventListener('mouseup', handleDragEnd)
      event.preventDefault()
    }
  }

  /** 处理拖动移动 */
  function handleDragMove(event: MouseEvent): void {
    if (!activeViewport.value) return
    
    const viewer = activeViewport.value === 'original' ? originalViewer : translatedViewer
    viewer.drag(event.clientX, event.clientY)
    
    // 【业务契约 DualImageViewer】同步时联动另一个视口
    if (syncEnabled.value) {
      const otherViewer = activeViewport.value === 'original' ? translatedViewer : originalViewer
      otherViewer.setTransform(viewer.getTransform())
    }
  }

  /** 处理拖动结束 */
  function handleDragEnd(): void {
    if (activeViewport.value) {
      const viewer = activeViewport.value === 'original' ? originalViewer : translatedViewer
      viewer.endDrag()
    }
    activeViewport.value = null
    document.removeEventListener('mousemove', handleDragMove)
    document.removeEventListener('mouseup', handleDragEnd)
  }


  // 记录当前绘制使用的视口
  let drawingViewport: 'original' | 'translated' = 'translated'

  /** 开始绘制新气泡 */
  function startDrawing(event: MouseEvent, viewport: 'original' | 'translated' = 'translated'): void {
    // 记录当前绘制的视口，用于后续坐标计算。
    drawingViewport = viewport
    
    // 获取对应视口的wrapper和scale
    const wrapper = viewport === 'original' ? originalWrapperRef.value : translatedWrapperRef.value
    const viewer = viewport === 'original' ? originalViewer : translatedViewer
    if (!wrapper) return
    
    const wrapperRect = wrapper.getBoundingClientRect()
    
    // 计算鼠标相对于wrapper的位置，然后转换为图片原生坐标
    const imgX = (event.clientX - wrapperRect.left) / viewer.scale.value
    const imgY = (event.clientY - wrapperRect.top) / viewer.scale.value

    drawStartX.value = imgX
    drawStartY.value = imgY
    isDrawingBox.value = true
    currentDrawingRect.value = [imgX, imgY, imgX, imgY]

    // 添加全局事件监听
    document.addEventListener('mousemove', handleDrawingMove)
    document.addEventListener('mouseup', handleDrawingEnd)
  }

  /** 处理绘制移动 */
  function handleDrawingMove(event: MouseEvent): void {
    if (!isDrawingBox.value) return

    // 使用开始绘制时记录的视口。
    const wrapper = drawingViewport === 'original' ? originalWrapperRef.value : translatedWrapperRef.value
    const viewer = drawingViewport === 'original' ? originalViewer : translatedViewer
    if (!wrapper) return

    const wrapperRect = wrapper.getBoundingClientRect()
    const imgX = (event.clientX - wrapperRect.left) / viewer.scale.value
    const imgY = (event.clientY - wrapperRect.top) / viewer.scale.value

    // 更新临时矩形
    currentDrawingRect.value = [
      Math.min(drawStartX.value, imgX),
      Math.min(drawStartY.value, imgY),
      Math.max(drawStartX.value, imgX),
      Math.max(drawStartY.value, imgY)
    ]
  }

  /** 处理绘制结束 */
  function handleDrawingEnd(_event: MouseEvent): void {
    document.removeEventListener('mousemove', handleDrawingMove)
    document.removeEventListener('mouseup', handleDrawingEnd)

    // 先保存中键状态，再重置，用于后续判断是否退出绘制模式
    const wasMiddleButton = isMiddleButtonDown.value

    if (!isDrawingBox.value || !currentDrawingRect.value) {
      isDrawingBox.value = false
      currentDrawingRect.value = null
      isMiddleButtonDown.value = false
      return
    }

    const [x1, y1, x2, y2] = currentDrawingRect.value
    const width = x2 - x1
    const height = y2 - y1

    // 最小尺寸检查
    if (width > 10 && height > 10) {
      // 添加新气泡
      bubbleStore.addBubble(currentDrawingRect.value)
      // 选中新添加的气泡
      bubbleStore.selectBubble(bubbleStore.bubbleCount - 1)
      console.log('已添加新气泡:', currentDrawingRect.value)
    }

    isDrawingBox.value = false
    currentDrawingRect.value = null
    isMiddleButtonDown.value = false

    // 如果不是中键绘制（即通过"添加"按钮进入的绘制模式），绘制完成后退出绘制模式
    if (!wasMiddleButton && isDrawingMode.value) {
      isDrawingMode.value = false
    }
  }

  /** 处理图片加载完成 */
  function handleImageLoad(viewport: 'original' | 'translated'): void {
    // 获取图片元素和尺寸
    const wrapperRef = viewport === 'original' ? originalWrapperRef : translatedWrapperRef
    const img = wrapperRef.value?.querySelector('img')
    const width = img?.naturalWidth || 0
    const height = img?.naturalHeight || 0
    
    console.log(`[EditWorkspace] ${viewport} 图片加载完成，尺寸: ${width}x${height}`)
    
    // 原图加载完成时更新尺寸
    if (viewport === 'original') {
      updateImageDimensions()
    }
    
    // 只在以下情况自动适应屏幕：
    // 1. 初始状态（scale=1, translate=0,0）- 首次进入编辑模式
    // 2. 检测到超大图片（超过4K）- 强制适应以避免渲染问题
    const isInitialState = scale.value === 1 && translateX.value === 0 && translateY.value === 0
    const isLargeImage = width > 3840 || height > 2160
    
    if (viewport === 'original' && (isInitialState || isLargeImage)) {
      if (isLargeImage) {
        console.log(`[EditWorkspace] 检测到大图（超过4K），自动适应屏幕`)
      }
      nextTick(() => {
        setTimeout(() => {
          fitToScreen()
        }, 50)
      })
    }
  }

  /** 处理重新渲染 */
  function handleReRender(): void {
    reRenderFullImage()
  }

  function handleExitToolbarAction(): void {
    if (exitDialogState.value === 'saving') {
      return
    }

    if (shouldPromptSaveOnExit.value) {
      openExitDialog()
      return
    }

    exitEditMode()
  }

  /**
   * 处理气泡更新并同步独立修复设置
   * 即使没有选中气泡，也能更新编辑面板的修复设置状态
   */
  function handleBubbleUpdateWithSync(updates: Partial<BubbleState>): void {
    // 同步修复设置到独立状态（不依赖气泡选中）
    if (updates.inpaintMethod !== undefined) {
      currentInpaintMethod.value = updates.inpaintMethod
    }
    if (updates.fillColor !== undefined) {
      currentFillColor.value = updates.fillColor
    }
    
    // 如果有选中的气泡，才更新气泡状态
    if (selectedBubbleIndex.value >= 0) {
      handleBubbleUpdate(updates)
    }
  }

  /**
   * 重置当前气泡到初始状态
   * 快照在进入编辑模式和切换图片时刷新，由工作区统一持有。
   */
  function handleResetCurrentBubble(index: number): void {
    const initialState = bubbleStore.initialStates[index]
    if (!initialState) {
      console.warn(`无法重置气泡 #${index + 1}：找不到初始状态`)
      showToast('无法重置：找不到初始状态', 'warning')
      return
    }
    
    // 使用初始状态的深拷贝来更新当前气泡
    const clonedState = JSON.parse(JSON.stringify(initialState))
    bubbleStore.updateBubble(index, clonedState)
    console.log(`气泡 #${index + 1} 已重置到初始状态`)
    showToast('气泡已重置', 'success')
    
    // 触发重新渲染
    reRenderFullImage()
  }

  /** 处理重新 OCR 识别单个气泡（带 loading 状态） */
  async function handleOcrRecognize(index: number): Promise<void> {
    isOcrLoading.value = true
    try {
      await bubbleOcrRecognize(index)
    } finally {
      isOcrLoading.value = false
    }
  }

  /** 处理修复选中气泡背景（带 loading 状态） */
  async function handleRepairSelectedBubble(): Promise<void> {
    isRepairLoading.value = true
    try {
      await bubbleRepairSelectedBubble()
    } finally {
      isRepairLoading.value = false
    }
  }

  // ============================================================
  // 笔刷方法 - 使用 useBrush composable
  // ============================================================

  /** 激活修复笔刷 */
  function activateRepairBrush(): void {
    toggleBrushMode('repair')
  }

  /** 激活还原笔刷 */
  function activateRestoreBrush(): void {
    toggleBrushMode('restore')
  }

  /** 全局鼠标移动处理（用于笔刷光标跟踪和涂抹） */
  function handleGlobalMouseMove(event: MouseEvent): void {
    continueBrushPainting(event)
  }

  /** 全局鼠标抬起处理（用于结束笔刷涂抹） */
  function handleGlobalMouseUp(): void {
    finishBrushPainting()
  }

  // ============================================================
  // 其他方法
  // ============================================================

  /** 保存当前编辑结果并跳转到下一张。 */
  async function applyAndNext(): Promise<void> {
    if (exitDialogState.value !== 'saving') {
      closeExitDialog()
    }
    saveBubbleStatesToImage()
    
    // 等待渲染完成后再切图，避免下一张读取到未落盘的画面。
    const renderSucceeded = await reRenderFullImage()
    if (!renderSucceeded) {
      showToast('应用失败，已停留在当前图片，请重试', 'warning')
      return
    }

    const sourceImageIndex = currentImageIndex.value
    const targetImageIndex = canGoNext.value ? sourceImageIndex + 1 : sourceImageIndex

    if (isBookshelfMode.value) {
      try {
        let initialized = await isBookshelfSessionInitialized()
        if (!initialized) {
          const shouldInitialize = confirm('当前章节尚未初始化存档。首次使用“应用并下一张”需要先保存整章原图和基础元数据，是否继续？')
          if (!shouldInitialize) {
            return
          }

          initialized = await forceInitializeBookshelfSession()
          if (!initialized) {
            showToast('初始化章节存档失败，未跳转到下一张', 'error')
            return
          }
        }

        await saveBookshelfPageProgress(sourceImageIndex, targetImageIndex)
      } catch (error) {
        console.error('[EditWorkspace] 书架模式持久化保存失败:', error)
        const message = error instanceof Error
          ? error.message
          : '当前页保存失败，未跳转到下一张'
        showToast(message, 'error')
        return
      }
    }
    
    // 检查是否是最后一张
    if (canGoNext.value) {
      if (brushMode.value) {
        exitBrushMode()
      }
      imageStore.goToNext()
    } else {
      showToast('已是最后一张图片', 'info')
    }
  }

  // ============================================================
  // 生命周期
  // ============================================================

  // ============================================================
  // 错误边界
  // ============================================================

  /** 捕获子组件错误，提供用户友好的错误提示 */
  onErrorCaptured((err, _instance, info) => {
    console.error('[EditWorkspace] 捕获到错误:', err, info)
    
    // 显示用户友好的错误提示
    const userMessage = err instanceof Error ? err.message : '操作失败，请重试'
    showToast(userMessage, 'error')
    
    // 返回 false 阻止错误继续传播
    return false
  })

  // ============================================================
  // 生命周期钩子
  // ============================================================

  onMounted(() => {
    // 加载保存的布局模式
    try {
      const savedLayout = localStorage.getItem(LAYOUT_MODE_KEY)
      if (savedLayout === 'horizontal' || savedLayout === 'vertical') {
        layoutMode.value = savedLayout
      }
    } catch (e) {
      console.warn('加载布局模式失败:', e)
    }

    // 键盘快捷键需要在编辑工作区获得焦点之外仍然可用。
    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('keyup', handleKeyUp)
    // 添加全局鼠标移动监听（用于笔刷光标跟踪和涂抹）
    document.addEventListener('mousemove', handleGlobalMouseMove)
    // 添加全局鼠标抬起监听（用于结束笔刷涂抹）
    document.addEventListener('mouseup', handleGlobalMouseUp)

    // 加载当前图片的气泡状态（loadBubbleStatesFromImage 内部已调用 fitToScreen）
    if (props.isEditModeActive) {
      loadBubbleStatesFromImage()
      nextTick(() => {
        workspaceRef.value?.focus()
      })
    }
  })

  onUnmounted(() => {
    // 移除全局事件监听
    document.removeEventListener('keydown', handleKeyDown)
    document.removeEventListener('keyup', handleKeyUp)
    document.removeEventListener('mousemove', handleGlobalMouseMove)
    document.removeEventListener('mouseup', handleGlobalMouseUp)
    document.removeEventListener('mousemove', handleDrawingMove)
    document.removeEventListener('mouseup', handleDrawingEnd)
    document.removeEventListener('mousemove', handleDividerDrag)
    document.removeEventListener('mouseup', stopDividerDrag)
    document.removeEventListener('mousemove', handlePanelResize)
    document.removeEventListener('mouseup', stopPanelResize)
  })

  // 监听编辑模式激活状态
  watch(() => props.isEditModeActive, (active) => {
    if (active) {
      loadBubbleStatesFromImage()
      nextTick(() => {
        workspaceRef.value?.focus()
        updateImageDimensions()
        // 进入编辑模式时等待图片和容器完成布局，再计算初始缩放。
        // 8K 等超大图片依赖这个时序获得正确视口尺寸。
        setTimeout(() => {
          fitToScreen()
        }, 100)
      })
    } else if (exitDialogState.value !== 'saving') {
      closeExitDialog()
    }
  })

  // 监听当前图片变化（loadBubbleStatesFromImage 内部已调用 fitToScreen）
  watch(currentImageIndex, () => {
    if (props.isEditModeActive) {
      if (exitDialogState.value !== 'saving') {
        closeExitDialog()
      }
      loadBubbleStatesFromImage()
    }
  })

  // 监听选中气泡变化，同步修复设置到独立状态
  // 对应业务契约 selectBubbleNew 中更新 $('#bubbleInpaintMethodNew') 的逻辑
  watch(selectedBubble, (bubble) => {
    if (bubble) {
      currentInpaintMethod.value = bubble.inpaintMethod || 'solid'
      currentFillColor.value = bubble.fillColor || '#FFFFFF'
    }
  }, { immediate: true })

  return {
    workspaceRef,
    imageComparisonRef,
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
    currentInpaintMethod,
    currentFillColor,
    isOcrLoading,
    isTranslateLoading,
    isRepairLoading,
    scale,
    translateX,
    translateY,
    originalScale,
    activeViewport,
    originalTransformStyle,
    translatedTransformStyle,
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
    bubbleRepairSelectedBubble,
    bubbleOcrRecognize,
    brushMode,
    brushSize,
    mouseX,
    mouseY,
    isBrushKeyDown,
    toggleBrushMode,
    exitBrushMode,
    startBrushPainting,
    continueBrushPainting,
    finishBrushPainting,
    adjustBrushSize,
    isProcessing,
    progressText,
    progressCurrent,
    progressTotal,
    exitDialogState,
    exitSaveMessage,
    exitDialogError,
    exitSaveProgressPercent,
    exitSaveHasProgress,
    exitSaveCurrent,
    exitSaveTotal,
    openExitDialog,
    closeExitDialog,
    exitWithoutSaving,
    saveAndExit,
    startDividerDrag,
    startPanelResize,
    zoomIn,
    zoomOut,
    resetZoom,
    prepareForNavigation,
    selectFirstBubbleIfExists,
    goToPreviousImage,
    goToNextImage,
    switchToImage,
    saveBubbleStatesToImage,
    loadBubbleStatesFromImage,
    selectPreviousBubble,
    selectNextBubble,
    toggleThumbnails,
    toggleLayout,
    toggleViewMode,
    toggleSync,
    fitToScreen,
    handleWheel,
    handleMouseDown,
    handleDragMove,
    handleDragEnd,
    startDrawing,
    handleDrawingMove,
    handleDrawingEnd,
    handleImageLoad,
    handleReRender,
    handleExitToolbarAction,
    handleBubbleUpdateWithSync,
    handleResetCurrentBubble,
    handleOcrRecognize,
    handleReTranslateBubble,
    handleRepairSelectedBubble,
    activateRepairBrush,
    activateRestoreBrush,
    handleGlobalMouseMove,
    handleGlobalMouseUp,
    applyAndNext,
    autoDetectBubbles,
    detectAllImages,
    translateWithCurrentBubbles,
  }
}
