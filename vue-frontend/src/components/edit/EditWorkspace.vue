<!--
  编辑模式工作区组件
  提供双图对照、气泡编辑、笔刷工具等功能
  对应原 edit_mode.js 中的编辑模式功能
-->
<template>
  <div
    v-if="isEditModeActive"
    class="edit-workspace"
    :class="[
      `layout-${layoutMode}`,
      { 'drawing-mode': isDrawingMode }
    ]"
    tabindex="0"
    ref="workspaceRef"
  >
    <!-- 顶部工具栏 - 使用拆分的组件 -->
    <EditToolbar
      :current-image-index="currentImageIndex"
      :image-count="imageCount"
      :can-go-previous="canGoPrevious"
      :can-go-next="canGoNext"
      :show-thumbnails="showThumbnails"
      :has-bubbles="hasBubbles"
      :selected-bubble-index="selectedBubbleIndex"
      :bubble-count="bubbleCount"
      :layout-mode="layoutMode"
      :sync-enabled="syncEnabled"
      :scale="scale"
      :is-drawing-mode="isDrawingMode"
      :has-selection="hasSelection"
      :brush-mode="brushMode"
      :brush-size="brushSize"
      :mouse-x="mouseX"
      :mouse-y="mouseY"
      :is-processing="isProcessing"
      :progress-text="progressText"
      :progress-current="progressCurrent"
      :progress-total="progressTotal"
      @go-previous-image="goToPreviousImage"
      @go-next-image="goToNextImage"
      @toggle-thumbnails="toggleThumbnails"
      @select-previous-bubble="selectPreviousBubble"
      @select-next-bubble="selectNextBubble"
      @toggle-layout="toggleLayout"
      @toggle-view-mode="toggleViewMode"
      @toggle-sync="toggleSync"
      @fit-to-screen="fitToScreen"
      @zoom-in="zoomIn"
      @zoom-out="zoomOut"
      @reset-zoom="resetZoom"
      @exit-edit-mode="exitEditMode"
      @auto-detect-bubbles="autoDetectBubbles"
      @detect-all-images="detectAllImages"
      @translate-with-bubbles="translateWithCurrentBubbles"
      @toggle-drawing-mode="toggleDrawingMode"
      @delete-selected-bubbles="deleteSelectedBubbles"
      @repair-selected-bubble="repairSelectedBubble"
      @activate-repair-brush="activateRepairBrush"
      @activate-restore-brush="activateRestoreBrush"
      @apply-and-next="applyAndNext"
    />

    <!-- 缩略图面板 - 使用拆分的组件 -->
    <EditThumbnailPanel
      :visible="showThumbnails"
      :images="images"
      :current-image-index="currentImageIndex"
      @switch-to-image="switchToImage"
    />

    <!-- 主布局区域 -->
    <div class="edit-main-layout">
      <!-- 双图对照区域 -->
      <div class="image-comparison-container">
        <!-- 原图面板 -->
        <div
          v-show="viewMode !== 'translated'"
          class="image-panel original-panel"
          :class="{ collapsed: viewMode === 'translated' || originalPanelCollapsed }"
        >
          <div class="panel-header">
            <span class="panel-title">📖 原图 (日文)</span>
            <button class="panel-toggle" @click="originalPanelCollapsed = !originalPanelCollapsed" title="折叠/展开">
              {{ originalPanelCollapsed ? '+' : '−' }}
            </button>
          </div>
          <div
            ref="originalViewportRef"
            class="image-viewport"
            @wheel.prevent="handleWheel($event, 'original')"
            @mousedown="handleMouseDown($event, 'original')"
            @dblclick="fitToScreen"
          >
            <div
              ref="originalWrapperRef"
              class="image-canvas-wrapper"
              :style="originalTransformStyle"
            >
              <img
                v-if="currentImage?.originalDataURL"
                :src="currentImage.originalDataURL"
                alt="原图"
                @load="handleImageLoad('original')"
              />
              <!-- 气泡高亮覆盖层 -->
              <BubbleOverlay
                v-if="currentImage?.originalDataURL"
                :bubbles="bubbles"
                :selected-index="selectedBubbleIndex"
                :selected-indices="selectedIndices"
                :scale="originalScale"
                :is-drawing-mode="isDrawingMode"
                :is-brush-mode="!!brushMode"
                :image-width="currentImageWidth"
                :image-height="currentImageHeight"
                @select="handleBubbleSelect"
                @multi-select="handleBubbleMultiSelect"
                @drag-start="handleBubbleDragStart"
                @dragging="handleBubbleDragging"
                @drag-end="handleBubbleDragEnd"
                @resize-start="handleBubbleResizeStart"
                @resizing="handleBubbleResizing"
                @resize-end="handleBubbleResizeEnd"
                @rotate-start="handleBubbleRotateStart"
                @rotating="handleBubbleRotating"
                @rotate-end="handleBubbleRotateEnd"
                @draw-bubble="handleDrawBubble"
              />
              <!-- 绘制中的临时矩形 -->
              <div
                v-if="currentDrawingRect"
                class="drawing-rect-edit"
                :style="getDrawingRectStyle()"
              ></div>
            </div>
          </div>
        </div>

        <!-- 分隔条 -->
        <div
          v-if="viewMode === 'dual'"
          class="panel-divider"
          :class="{ 'vertical-divider': layoutMode === 'vertical' }"
          @mousedown="startDividerDrag"
        ></div>

        <!-- 翻译图面板 -->
        <div
          v-show="viewMode !== 'original'"
          class="image-panel translated-panel"
          :class="{ collapsed: viewMode === 'original' || translatedPanelCollapsed }"
        >
          <div class="panel-header">
            <span class="panel-title">📝 翻译图 (中文)</span>
            <button class="panel-toggle" @click="translatedPanelCollapsed = !translatedPanelCollapsed" title="折叠/展开">
              {{ translatedPanelCollapsed ? '+' : '−' }}
            </button>
          </div>
          <div
            ref="translatedViewportRef"
            class="image-viewport"
            @wheel.prevent="handleWheel($event, 'translated')"
            @mousedown="handleMouseDown($event, 'translated')"
            @dblclick="fitToScreen"
          >
            <div
              ref="translatedWrapperRef"
              class="image-canvas-wrapper"
              :style="translatedTransformStyle"
            >
              <img
                v-if="currentImage?.translatedDataURL || currentImage?.originalDataURL"
                :src="currentImage?.translatedDataURL || currentImage?.originalDataURL"
                alt="翻译图"
                @load="handleImageLoad('translated')"
              />
              <!-- 气泡高亮覆盖层 -->
              <BubbleOverlay
                v-if="currentImage?.translatedDataURL || currentImage?.originalDataURL"
                :bubbles="bubbles"
                :selected-index="selectedBubbleIndex"
                :selected-indices="selectedIndices"
                :scale="scale"
                :is-drawing-mode="isDrawingMode"
                :is-brush-mode="!!brushMode"
                :image-width="currentImageWidth"
                :image-height="currentImageHeight"
                @select="handleBubbleSelect"
                @multi-select="handleBubbleMultiSelect"
                @drag-start="handleBubbleDragStart"
                @dragging="handleBubbleDragging"
                @drag-end="handleBubbleDragEnd"
                @resize-start="handleBubbleResizeStart"
                @resizing="handleBubbleResizing"
                @resize-end="handleBubbleResizeEnd"
                @rotate-start="handleBubbleRotateStart"
                @rotating="handleBubbleRotating"
                @rotate-end="handleBubbleRotateEnd"
                @draw-bubble="handleDrawBubble"
              />
              <!-- 绘制中的临时矩形 -->
              <div
                v-if="currentDrawingRect"
                class="drawing-rect-edit translated-drawing-rect"
                :style="getDrawingRectStyle()"
              ></div>
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧/底部编辑面板 - 始终显示 -->
      <div ref="editPanelRef" class="edit-panel-container">
        <!-- 面板调整手柄 -->
        <div
          class="panel-resize-handle vertical"
          @mousedown="startPanelResize"
        >
          ⋮⋮⋮
        </div>
        <!-- 编辑面板内容 -->
        <BubbleEditor
          :bubble="selectedBubble"
          :bubble-index="selectedBubbleIndex"
          @update="handleBubbleUpdateWithSync"
          @re-render="handleReRender"
          @ocr-recognize="handleOcrRecognize"
          @re-translate="handleReTranslateBubble"
          @apply-bubble="handleApplyBubble"
          @reset-current="handleResetCurrentBubble"
        />
      </div>
    </div>
  </div>
</template>


<script setup lang="ts">
/**
 * 编辑模式工作区组件
 * 提供双图对照、气泡编辑、笔刷工具等功能
 */
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageViewer } from '@/composables/useImageViewer'
import { useBrush } from '@/composables/useBrush'
import { useBubbleActions } from '@/composables/useBubbleActions'
import { useEditRender } from '@/composables/useEditRender'
import { useTranslation } from '@/composables/useTranslation'
import { detectBoxes } from '@/api/translate'
import { useSettingsStore } from '@/stores/settingsStore'
import { showToast } from '@/utils/toast'
import BubbleOverlay from './BubbleOverlay.vue'
import BubbleEditor from './BubbleEditor.vue'
import EditToolbar from './EditToolbar.vue'
import EditThumbnailPanel from './EditThumbnailPanel.vue'
import { LAYOUT_MODE_KEY } from '@/constants'
import type { ImageData as AppImageData } from '@/types/image'
import type { BubbleState, InpaintMethod } from '@/types/bubble'

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 是否激活编辑模式 */
  isEditModeActive: boolean
}>()

const emit = defineEmits<{
  /** 退出编辑模式 */
  (e: 'exit'): void
}>()

// ============================================================
// Store 引用
// ============================================================

const imageStore = useImageStore()
const bubbleStore = useBubbleStore()

// 使用翻译 composable（用于"使用当前气泡翻译"功能）
const {
  translateWithCurrentBubbles: translateWithBubbles
} = useTranslation()

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
  handleBubbleDragging,
  handleBubbleDragEnd,
  handleBubbleResizeStart,
  handleBubbleResizing,
  handleBubbleResizeEnd,
  handleBubbleRotateStart,
  handleBubbleRotating,
  handleBubbleRotateEnd,
  toggleDrawingMode,
  handleDrawBubble,
  getDrawingRectStyle,
  handleBubbleUpdate,
  deleteSelectedBubbles,
  repairSelectedBubble,
  handleOcrRecognize
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
  // 【复刻原版】提供当前编辑面板的修复设置，不依赖气泡选中状态
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

/** 当前图片宽度 */
const currentImageWidth = computed(() => {
  const img = originalWrapperRef.value?.querySelector('img')
  return img?.naturalWidth || 2000
})

/** 当前图片高度 */
const currentImageHeight = computed(() => {
  const img = originalWrapperRef.value?.querySelector('img')
  return img?.naturalHeight || 2000
})


// ============================================================
// 模板引用
// ============================================================

const workspaceRef = ref<HTMLElement | null>(null)
const originalViewportRef = ref<HTMLElement | null>(null)
const originalWrapperRef = ref<HTMLElement | null>(null)
const translatedViewportRef = ref<HTMLElement | null>(null)
const translatedWrapperRef = ref<HTMLElement | null>(null)
const editPanelRef = ref<HTMLElement | null>(null)

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

/** 面板折叠状态 */
const originalPanelCollapsed = ref(false)
const translatedPanelCollapsed = ref(false)

// ============================================================
// 【复刻原版】独立的修复设置状态（不依赖气泡选中）
// 对应原版 $('#bubbleInpaintMethodNew').val() 和 $('#fillColorNew').val()
// ============================================================

/** 当前编辑面板选择的修复方式 */
const currentInpaintMethod = ref<InpaintMethod>('solid')

/** 当前编辑面板选择的填充颜色 */
const currentFillColor = ref('#FFFFFF')

// ============================================================
// 进度条状态
// ============================================================

/** 是否正在处理 */
const isProcessing = ref(false)

/** 进度文本 */
const progressText = ref('处理中...')

/** 当前进度 */
const progressCurrent = ref(0)

/** 总进度 */
const progressTotal = ref(0)

// ============================================================
// 图片查看器状态
// 【复刻原版 DualImageViewer】支持两套独立变换状态，syncEnabled 开启时联动
// ============================================================

// 原图查看器
const originalViewer = useImageViewer()
// 翻译图查看器
const translatedViewer = useImageViewer()

// 主缩放比例（用于工具栏显示和统一的缩放操作）
const scale = computed(() => translatedViewer.scale.value)
const translateX = computed(() => translatedViewer.translateX.value)
const translateY = computed(() => translatedViewer.translateY.value)

// 【复刻原版】原图视口的缩放比例（sync关闭时两个视口可能缩放不同）
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

// 绘制和操作状态已迁移到 useBubbleActions composable
// 笔刷状态和方法已迁移到 useBrush composable


// ============================================================
// 分隔条拖拽状态
// ============================================================

const isDraggingDivider = ref(false)
const dividerStartPos = ref(0)

// ============================================================
// 面板调整状态
// ============================================================

const isResizingPanel = ref(false)
const panelResizeStart = ref({ x: 0, y: 0, size: 0 })

// ============================================================
// 图片导航方法
// ============================================================

/** 导航前的公共处理（复刻原版逻辑） */
function prepareForNavigation(): void {
  // 【复刻原版】退出笔刷模式，调用exitBrushMode确保状态正确清理
  if (brushMode.value) {
    exitBrushMode()
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
  
  // 【复刻原版 4.2】保持 null vs [] 语义区分：
  // - null/undefined：从未处理过
  // - []：处理过但用户删光了
  // 只要 currentImage.bubbleStates 曾经是数组（包括空数组），就应该保存当前状态
  const hadBubbleStates = Array.isArray(currentImage.value.bubbleStates)
  
  if (bubbles.value.length > 0) {
    // 有气泡，保存当前状态
    imageStore.updateCurrentBubbleStates([...bubbles.value])
    console.log('已保存气泡状态到当前图片')
  } else if (hadBubbleStates) {
    // 用户删光了气泡，保存空数组（保持"处理过"的语义）
    imageStore.updateCurrentBubbleStates([])
    console.log('已保存空气泡状态到当前图片（用户清空）')
  }
  // 如果 bubbleStates 从未是数组且当前也没有气泡，不做任何操作（保持 null 语义）
}

/** 从当前图片加载气泡状态 */
function loadBubbleStatesFromImage(): void {
  if (currentImage.value?.bubbleStates) {
    // skipSync=true 避免冗余同步（数据已经在 imageStore 中）
    bubbleStore.setBubbles([...currentImage.value.bubbleStates], true)
    console.log(`已加载 ${currentImage.value.bubbleStates.length} 个气泡状态`)
  } else {
    // 【复刻原版】使用 clearBubblesLocal 仅清除本地状态，不同步到 imageStore
    // 这保持了 null（未处理）和 []（用户主动清空）的语义区分
    bubbleStore.clearBubblesLocal()
  }
  selectFirstBubbleIfExists()
  // 【复刻原版】切图时保持当前缩放和位置，不自动 fitToScreen
  // 旧版 navigateImage() 调用 loadImagesToViewer(false) 保持视图位置
}

// ============================================================
// 气泡导航方法
// ============================================================

/** 选择上一个气泡 */
function selectPreviousBubble(): void {
  bubbleStore.selectPrevious()
  // 【复刻原版】selectBubbleNew() 刻意不滚动到气泡，避免画面跳动
}

/** 选择下一个气泡 */
function selectNextBubble(): void {
  bubbleStore.selectNext()
  // 【复刻原版】selectBubbleNew() 刻意不滚动到气泡，避免画面跳动
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
  // 【复刻原版 4.4】切换布局后延迟 300ms 自动适应屏幕
  // 旧版 toggleLayoutMode() 会在切换后调用 fitToScreen
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
  // 【复刻原版】开启同步时，立即同步两个视口的变换状态
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

  // 【修复】切换图片时两个视口都需要适应屏幕，无论 syncEnabled 状态
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
  
  // 【复刻原版 DualImageViewer】操作对应视口，同步时联动另一个
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
    
    // 【复刻原版】点击空白处清除多选（非 Shift 时）
    // 旧版 handleBubbleMouseDown 第2444-2448行
    if (!event.shiftKey) {
      handleClearMultiSelect()
    }
    
    // 【复刻原版】记录当前操作的视口
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
  
  // 【复刻原版 DualImageViewer】同步时联动另一个视口
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
  // 【修复】记录当前绘制的视口，用于后续坐标计算
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

  // 【修复】使用开始绘制时记录的视口
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

  // 【复刻原版】先保存中键状态，再重置，用于后续判断是否退出绘制模式
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

  // 【复刻原版】如果不是中键绘制（即通过"添加"按钮进入的绘制模式），绘制完成后退出绘制模式
  if (!wasMiddleButton && isDrawingMode.value) {
    isDrawingMode.value = false
  }
}

/** 处理图片加载完成 */
function handleImageLoad(viewport: 'original' | 'translated'): void {
  console.log(`${viewport} 图片加载完成`)
  // 首次加载时适应屏幕
  if (scale.value === 1 && translateX.value === 0 && translateY.value === 0) {
    nextTick(() => {
      fitToScreen()
    })
  }
}

// 气泡操作方法已迁移到 useBubbleActions composable

/** 处理重新渲染 */
function handleReRender(): void {
  reRenderFullImage()
}

/**
 * 【复刻原版】处理气泡更新并同步独立修复设置
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

/** 处理应用单个气泡更改 */
function handleApplyBubble(_index: number): void {
  // 【复刻原版 4.5】应用文本时显示 toast 提示
  // 旧版 applyCurrentText() 会 toast "文本已应用"
  showToast('文本已应用', 'success')
  // 应用气泡更改后触发重新渲染
  reRenderFullImage()
}

/**
 * 【复刻原版 4.3】重置当前气泡到初始状态
 * 旧版使用 state.initialBubbleStates 保存进入编辑模式/切图时的快照
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

/** 处理重新翻译单个气泡 */
async function handleReTranslateBubble(index: number): Promise<void> {
  const bubble = bubbles.value[index]
  if (!bubble?.originalText) {
    console.warn('无法重新翻译：缺少气泡或原文')
    return
  }

  try {
    console.log(`开始重新翻译气泡 #${index + 1}`)
    const { translateSingleText } = await import('@/api/translate')
    const { useSettingsStore } = await import('@/stores/settingsStore')
    const settings = useSettingsStore().settings
    
    const response = await translateSingleText({
      original_text: bubble.originalText,
      model_provider: settings.translation.provider,
      api_key: settings.translation.apiKey,
      model_name: settings.translation.modelName,
      custom_base_url: settings.translation.customBaseUrl,
      target_language: settings.targetLanguage,
      prompt_content: settings.translatePrompt
    })

    if (response.success && response.data?.translated_text) {
      bubbleStore.updateBubble(index, { translatedText: response.data.translated_text })
      console.log(`翻译成功: "${response.data.translated_text}"`)
      reRenderFullImage()
    } else {
      console.error('翻译失败:', response.error || '未知错误')
    }
  } catch (error) {
    console.error('翻译出错:', error)
  }
}

/** 初始化图片的文本数组（复刻原版逻辑） */
function initializeTextArrays(image: AppImageData, count: number): void {
  if (!image.bubbleTexts) image.bubbleTexts = []
  if (!image.originalTexts) image.originalTexts = []
  while (image.bubbleTexts.length < count) {
    image.bubbleTexts.push('')
  }
  while (image.originalTexts.length < count) {
    image.originalTexts.push('')
  }
}

/** 从检测响应创建气泡状态数组（复刻原版逻辑） */
function createBubbleStatesFromDetection(
  response: { bubble_coords: number[][]; bubble_angles?: number[]; auto_directions?: string[] },
  image: AppImageData,
  textStyle: { fontSize: number; fontFamily: string; textColor: string; fillColor: string; strokeEnabled: boolean; strokeColor: string; strokeWidth: number; inpaintMethod: string }
): BubbleState[] {
  const autoDirections = response.auto_directions || []
  return response.bubble_coords.map((coords, i) => {
    const x1 = coords[0] ?? 0
    const y1 = coords[1] ?? 0
    const x2 = coords[2] ?? 0
    const y2 = coords[3] ?? 0
    let autoDir: 'vertical' | 'horizontal'
    if (autoDirections[i]) {
      autoDir = autoDirections[i] === 'v' ? 'vertical' : 'horizontal'
    } else {
      autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
    }
    return {
      coords: coords as [number, number, number, number],
      originalText: image.originalTexts?.[i] || '',
      translatedText: image.bubbleTexts?.[i] || '',
      textboxText: '',
      fontSize: textStyle.fontSize,
      fontFamily: textStyle.fontFamily,
      textDirection: autoDir,
      autoTextDirection: autoDir,
      textColor: textStyle.textColor,
      fillColor: textStyle.fillColor,
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth,
      rotationAngle: response.bubble_angles?.[i] || 0,
      inpaintMethod: textStyle.inpaintMethod as 'solid' | 'lama_mpe' | 'litelama',
      position: { x: 0, y: 0 },
      polygon: []
    }
  })
}

/** 自动检测气泡（复刻原版逻辑） */
async function autoDetectBubbles(): Promise<void> {
  const image = currentImage.value
  if (!image?.originalDataURL) {
    showToast('没有有效的图片用于检测', 'warning')
    return
  }

  try {
    showToast('正在自动检测文本框...', 'info')
    
    const match = image.originalDataURL.match(/^data:image\/[^;]+;base64,(.+)$/)
    const imageData = match?.[1] || ''
    if (!imageData) {
      showToast('无法解析图片数据', 'error')
      return
    }
    
    const settingsStore = useSettingsStore()
    const { textDetector, boxExpand, textStyle } = settingsStore.settings
    
    const response = await detectBoxes(imageData, textDetector, {
      box_expand_ratio: boxExpand.ratio,
      box_expand_top: boxExpand.top,
      box_expand_bottom: boxExpand.bottom,
      box_expand_left: boxExpand.left,
      box_expand_right: boxExpand.right
    })
    
    if (response.success && response.bubble_coords) {
      imageStore.updateCurrentImage({
        bubbleCoords: response.bubble_coords,
        bubbleAngles: response.bubble_angles || []
      })
      
      initializeTextArrays(image, response.bubble_coords.length)
      const detectionData = {
        bubble_coords: response.bubble_coords.map(c => [...c]) as number[][],
        bubble_angles: response.bubble_angles,
        auto_directions: response.auto_directions
      }
      const newBubbles = createBubbleStatesFromDetection(detectionData, image, textStyle)
      bubbleStore.setBubbles(newBubbles)
      selectFirstBubbleIfExists()
      
      showToast(`自动检测到 ${response.bubble_coords.length} 个文本框`, 'success')
    } else {
      showToast(response.error || '检测失败', 'error')
    }
  } catch (error) {
    console.error('自动检测失败:', error)
    showToast('自动检测失败', 'error')
  }
}

/** 批量检测所有图片（复刻原版逻辑） */
async function detectAllImages(): Promise<void> {
  if (images.value.length <= 1) {
    showToast('至少需要两张图片才能执行批量检测', 'warning')
    return
  }

  // 【复刻原版】确认对话框
  if (!confirm('此操作将对所有图片进行文本框检测，可能会覆盖已有的检测结果。确定继续吗？')) {
    return
  }

  // 获取检测器设置（在循环外获取，避免重复调用）
  const settingsStore = useSettingsStore()
  const { textDetector, boxExpand, textStyle } = settingsStore.settings

  // 【复刻原版】记录当前索引
  const originalIndex = currentImageIndex.value
  const totalImages = images.value.length
  
  // 初始化进度条
  isProcessing.value = true
  progressText.value = '批量检测中'
  progressTotal.value = totalImages
  progressCurrent.value = 0

  try {
    let totalDetected = 0

    for (let i = 0; i < totalImages; i++) {
      const image = images.value[i]
      if (!image?.originalDataURL) continue

      // 更新进度条
      progressCurrent.value = i + 1

      const match = image.originalDataURL.match(/^data:image\/[^;]+;base64,(.+)$/)
      const imageData = match?.[1] || ''
      if (!imageData) continue
      
      const response = await detectBoxes(imageData, textDetector, {
        box_expand_ratio: boxExpand.ratio,
        box_expand_top: boxExpand.top,
        box_expand_bottom: boxExpand.bottom,
        box_expand_left: boxExpand.left,
        box_expand_right: boxExpand.right
      })

      if (response.success && response.bubble_coords) {
        const img = images.value[i]
        if (img) {
          img.bubbleCoords = response.bubble_coords
          img.bubbleAngles = response.bubble_angles || []
          
          initializeTextArrays(img, response.bubble_coords.length)
          const detectionData = {
            bubble_coords: response.bubble_coords.map(c => [...c]) as number[][],
            bubble_angles: response.bubble_angles,
            auto_directions: response.auto_directions
          }
          img.bubbleStates = createBubbleStatesFromDetection(detectionData, img, textStyle)
          
          totalDetected += response.bubble_coords.length
          
          // 【复刻原版】如果是当前图片，同时更新显示
          if (i === currentImageIndex.value) {
            loadBubbleStatesFromImage()
          }
        }
      }
    }

    // 完成 - 更新进度条
    progressText.value = '检测完成'
    progressCurrent.value = totalImages

    // 【复刻原版】返回原始图片并刷新显示
    if (originalIndex !== currentImageIndex.value) {
      imageStore.setCurrentImageIndex(originalIndex)
    }
    loadBubbleStatesFromImage()
    
    showToast(`批量检测完成！共处理 ${totalImages} 张图片，检测到 ${totalDetected} 个文本框`, 'success')
    
    // 延迟隐藏进度条
    setTimeout(() => {
      isProcessing.value = false
    }, 2000)
  } catch (error) {
    console.error('批量检测失败:', error)
    showToast('批量检测失败', 'error')
    isProcessing.value = false
  }
}

/** 使用当前气泡翻译 - 委托给 useTranslation composable */
async function translateWithCurrentBubbles(): Promise<void> {
  const image = currentImage.value
  if (!image?.originalDataURL) {
    showToast('没有有效的图片用于翻译', 'warning')
    return
  }

  if (bubbles.value.length === 0) {
    showToast('没有文本框可用于翻译，请先检测或添加文本框', 'warning')
    return
  }

  showToast('正在使用当前文本框翻译...', 'info')

  try {
    const success = await translateWithBubbles()
    if (success) {
      showToast('翻译成功！', 'success')
      selectFirstBubbleIfExists()
    }
  } catch (error) {
    console.error('翻译失败:', error)
    showToast(`翻译失败: ${error instanceof Error ? error.message : '未知错误'}`, 'error')
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
// 分隔条拖拽
// ============================================================

/** 开始拖拽分隔条 */
function startDividerDrag(event: MouseEvent): void {
  isDraggingDivider.value = true
  dividerStartPos.value = layoutMode.value === 'horizontal' ? event.clientX : event.clientY
  document.body.style.cursor = layoutMode.value === 'horizontal' ? 'col-resize' : 'row-resize'
  document.body.style.userSelect = 'none'

  document.addEventListener('mousemove', handleDividerDrag)
  document.addEventListener('mouseup', stopDividerDrag)
  event.preventDefault()
}

/** 处理分隔条拖拽 */
function handleDividerDrag(event: MouseEvent): void {
  if (!isDraggingDivider.value) return

  const container = originalViewportRef.value?.parentElement?.parentElement
  if (!container) return

  const containerRect = container.getBoundingClientRect()
  
  if (layoutMode.value === 'horizontal') {
    const mouseX = event.clientX - containerRect.left
    const totalWidth = containerRect.width
    const leftPercent = Math.max(20, Math.min(80, (mouseX / totalWidth) * 100))
    
    const originalPanel = container.querySelector('.original-panel') as HTMLElement
    const translatedPanel = container.querySelector('.translated-panel') as HTMLElement
    if (originalPanel && translatedPanel) {
      originalPanel.style.flex = `0 0 ${leftPercent}%`
      translatedPanel.style.flex = `0 0 ${100 - leftPercent}%`
    }
  } else {
    const mouseY = event.clientY - containerRect.top
    const totalHeight = containerRect.height
    const topPercent = Math.max(20, Math.min(80, (mouseY / totalHeight) * 100))
    
    const originalPanel = container.querySelector('.original-panel') as HTMLElement
    const translatedPanel = container.querySelector('.translated-panel') as HTMLElement
    if (originalPanel && translatedPanel) {
      originalPanel.style.flex = `0 0 ${topPercent}%`
      translatedPanel.style.flex = `0 0 ${100 - topPercent}%`
    }
  }
}

/** 停止分隔条拖拽 */
function stopDividerDrag(): void {
  isDraggingDivider.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.removeEventListener('mousemove', handleDividerDrag)
  document.removeEventListener('mouseup', stopDividerDrag)
}


// ============================================================
// 编辑面板调整
// ============================================================

/** 开始调整面板大小 */
function startPanelResize(event: MouseEvent): void {
  isResizingPanel.value = true
  const panel = editPanelRef.value
  if (!panel) return

  panelResizeStart.value = {
    x: event.clientX,
    y: event.clientY,
    size: layoutMode.value === 'horizontal' ? panel.offsetWidth : panel.offsetHeight
  }

  document.body.style.cursor = layoutMode.value === 'horizontal' ? 'ew-resize' : 'ns-resize'
  document.body.style.userSelect = 'none'

  document.addEventListener('mousemove', handlePanelResize)
  document.addEventListener('mouseup', stopPanelResize)
  event.preventDefault()
}

/** 处理面板大小调整 */
function handlePanelResize(event: MouseEvent): void {
  if (!isResizingPanel.value || !editPanelRef.value) return

  if (layoutMode.value === 'horizontal') {
    const deltaX = panelResizeStart.value.x - event.clientX
    let newWidth = panelResizeStart.value.size + deltaX
    newWidth = Math.max(300, Math.min(window.innerWidth * 0.6, newWidth))
    editPanelRef.value.style.flex = `0 0 ${newWidth}px`
    editPanelRef.value.style.minWidth = `${newWidth}px`
  } else {
    const deltaY = panelResizeStart.value.y - event.clientY
    let newHeight = panelResizeStart.value.size + deltaY
    newHeight = Math.max(200, Math.min(window.innerHeight * 0.5, newHeight))
    editPanelRef.value.style.flex = `0 0 ${newHeight}px`
    editPanelRef.value.style.height = `${newHeight}px`
  }
}

/** 停止面板大小调整 */
function stopPanelResize(): void {
  isResizingPanel.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
  document.removeEventListener('mousemove', handlePanelResize)
  document.removeEventListener('mouseup', stopPanelResize)
}

// ============================================================
// 快捷键处理
// ============================================================

/** 处理键盘事件 */
function handleKeyDown(event: KeyboardEvent): void {
  const target = event.target as HTMLElement
  const key = event.key.toLowerCase()
  
  // 【复刻原版 edit_mode.js handleEditModeKeydown】
  // 笔刷快捷键 R/U 和导航快捷键 A/D 只在 textarea 中禁用（用户可能想输入文字）
  // 在其他所有元素（包括 select、input[type=number]、input[type=color] 等）中都允许触发
  if (key === 'r' || key === 'u' || key === 'a' || key === 'd') {
    if (target.tagName === 'TEXTAREA') return
    // 让其他输入元素失去焦点，以便快捷键正常工作
    if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'BUTTON') {
      target.blur()
    }
  } else {
    // 其他快捷键在输入框中不处理
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return
  }

  switch (event.key) {
    case 'Escape':
      // 【复刻原版】Escape 退出编辑模式（原版没有此快捷键，但保留作为增强）
      exitEditMode()
      break
    case 'Delete':
    case 'Backspace':
      // 【复刻原版】笔刷模式下不处理删除
      if (!brushMode.value && hasSelection.value) {
        deleteSelectedBubbles()
        event.preventDefault()
      }
      break
    case 'a':
    case 'A':
      // 【复刻原版】笔刷模式下不处理导航
      if (!brushMode.value) {
        goToPreviousImage()
        event.preventDefault()
      }
      break
    case 'd':
    case 'D':
      // 【复刻原版】笔刷模式下不处理导航
      if (!brushMode.value) {
        goToNextImage()
        event.preventDefault()
      }
      break
    case 'Enter':
      // 【复刻原版】Ctrl+Enter 应用并跳转下一张，笔刷模式下不处理
      if (event.ctrlKey && !brushMode.value) {
        applyAndNext()
        event.preventDefault()
      }
      break
    case 'r':
    case 'R':
      // 【复刻原版】R键进入修复笔刷模式
      if (!isBrushKeyDown.value) {
        toggleBrushMode('repair')
        event.preventDefault()
      }
      break
    case 'u':
    case 'U':
      // 【复刻原版】U键进入还原笔刷模式
      if (!isBrushKeyDown.value) {
        toggleBrushMode('restore')
        event.preventDefault()
      }
      break
    // 以下是 Vue 版增强的快捷键（原版没有，但不影响复刻）
    case '+':
    case '=':
      zoomIn()
      event.preventDefault()
      break
    case '-':
      zoomOut()
      event.preventDefault()
      break
    case '0':
      resetZoom()
      event.preventDefault()
      break
  }
}

/** 处理键盘释放 */
function handleKeyUp(event: KeyboardEvent): void {
  // 【复刻原版】R/U键释放时退出笔刷模式，调用exitBrushMode确保状态正确清理
  if (event.key === 'r' || event.key === 'R' || event.key === 'u' || event.key === 'U') {
    exitBrushMode()
    event.preventDefault()
  }
}

// ============================================================
// 其他方法
// ============================================================

/** 【修复问题3】应用更改并跳转下一张（复刻原版逻辑） */
async function applyAndNext(): Promise<void> {
  saveBubbleStatesToImage()
  
  // 【修复问题3】直接await reRenderFullImage，确保渲染完成后再切图
  await reRenderFullImage()
  
  // 【复刻原版】检查是否是最后一张
  if (canGoNext.value) {
    goToNextImage()
  } else {
    showToast('已是最后一张图片', 'info')
  }
}

/** 退出编辑模式 */
function exitEditMode(): void {
  saveBubbleStatesToImage()
  emit('exit')
}


// ============================================================
// 生命周期
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

  // 【修复问题1】添加全局键盘事件监听（document级别，复刻原版）
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
    })
  }
})

// 监听当前图片变化（loadBubbleStatesFromImage 内部已调用 fitToScreen）
watch(currentImageIndex, () => {
  if (props.isEditModeActive) {
    loadBubbleStatesFromImage()
  }
})

// 【复刻原版】监听选中气泡变化，同步修复设置到独立状态
// 对应原版 selectBubbleNew 中更新 $('#bubbleInpaintMethodNew') 的逻辑
watch(selectedBubble, (bubble) => {
  if (bubble) {
    currentInpaintMethod.value = bubble.inpaintMethod || 'solid'
    currentFillColor.value = bubble.fillColor || '#FFFFFF'
  }
}, { immediate: true })
</script>

<!-- 
  编辑工作区样式完全由全局 public/css/edit-mode.css 控制
  不在此处添加任何 scoped 样式，以避免覆盖全局布局样式
-->
