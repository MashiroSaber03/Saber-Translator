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

<style scoped>
/* ===================================
   编辑模式样式 - 完整迁移自 edit-mode.css
   =================================== */

/* ============ 编辑工作区 - 全屏覆盖 ============ */
.edit-workspace {
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: #1a1a2e;
  z-index: 9999;
  overflow: hidden;
  margin: 0;
  border-radius: 0;
}

/* ============ 顶部工具栏 - 双行布局 ============ */
.edit-toolbar-wrapper {
  flex-shrink: 0;
  background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.edit-toolbar {
  display: flex;
  align-items: center;
  padding: 8px 15px;
  gap: 10px;
}

.toolbar-row-1 {
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.toolbar-row-2 {
  background: rgba(0, 0, 0, 0.15);
}

.toolbar-spacer {
  flex: 1;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: rgba(255, 255, 255, 0.2);
  margin: 0 5px;
}

/* 图片导航 */
.image-navigator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.image-navigator .nav-btn {
  width: 36px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: rgba(102, 126, 234, 0.3);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.image-navigator .nav-btn:hover {
  background: rgba(102, 126, 234, 0.5);
}

.image-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.image-indicator {
  color: #fff;
  font-size: 14px;
  padding: 6px 12px;
  background: rgba(102, 126, 234, 0.2);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.image-indicator:hover {
  background: rgba(102, 126, 234, 0.4);
}

.image-indicator span {
  font-weight: bold;
  color: #667eea;
}

.thumb-toggle-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.thumb-toggle-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.thumb-toggle-btn.active {
  background: rgba(102, 126, 234, 0.5);
}

/* 编辑模式缩略图面板 */
.edit-thumbnails-panel {
  position: relative;
  top: auto;
  right: auto;
  bottom: auto;
  left: auto;
  width: auto;
  max-height: none;
  background: rgba(0, 0, 0, 0.3);
  padding: 10px 15px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.thumbnails-scroll {
  display: flex;
  flex-direction: row;
  gap: 10px;
  overflow-x: auto;
  overflow-y: hidden;
  height: auto;
  padding: 5px 0;
}

.thumbnails-scroll::-webkit-scrollbar {
  height: 6px;
}

.thumbnails-scroll::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.thumbnails-scroll::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}

.edit-thumbnail-item {
  flex-shrink: 0;
  width: 60px;
  height: 80px;
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.2s;
  position: relative;
}

.edit-thumbnail-item:hover {
  border-color: rgba(255, 255, 255, 0.5);
  transform: scale(1.05);
}

.edit-thumbnail-item.active {
  border-color: #667eea;
  box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
}

.edit-thumbnail-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.edit-thumbnail-item .thumb-index {
  position: absolute;
  bottom: 2px;
  right: 2px;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 3px;
}

.bubble-navigator {
  display: flex;
  align-items: center;
  gap: 10px;
}

.bubble-navigator .nav-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.bubble-navigator .nav-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.bubble-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.bubble-indicator {
  color: #fff;
  font-size: 14px;
  padding: 6px 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 6px;
}

.bubble-indicator span {
  font-weight: bold;
  color: #00ff88;
}

/* 视图控制按钮 */
.view-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.view-controls button {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.view-controls button:hover {
  background: rgba(255, 255, 255, 0.2);
}

.view-controls #zoomLevel {
  min-width: 50px;
  text-align: center;
  color: #fff;
  font-size: 13px;
  padding: 0 8px;
}

.view-mode-btn {
  font-size: 18px !important;
}

.view-mode-btn.single-mode .dual-icon {
  opacity: 0.5;
}

/* 快捷操作 */
.quick-actions {
  display: flex;
  gap: 10px;
}

/* 主要按钮 */
.edit-toolbar .primary-btn,
.quick-actions .primary-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
  color: #1a1a2e;
  font-weight: 600;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.edit-toolbar .primary-btn:hover,
.quick-actions .primary-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
}

/* 次要按钮 */
.edit-toolbar .secondary-btn,
.quick-actions .secondary-btn {
  padding: 8px 16px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 6px;
  background: transparent;
  color: #fff;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.edit-toolbar .secondary-btn:hover,
.quick-actions .secondary-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(255, 255, 255, 0.5);
}

/* ============ 双图对照区域 ============ */
.edit-main-layout {
  flex: 1;
  display: flex;
  flex-direction: row;
  gap: 0;
  min-height: 0;
}

.image-comparison-container {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  padding: 8px;
  gap: 0;
  min-width: 0;
}

/* 图片面板 */
.image-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 150px;
  background: #16213e;
  border-radius: 8px;
  overflow: hidden;
  transition: flex 0.3s ease;
}

.image-panel.collapsed {
  flex: 0 0 40px;
  min-width: 40px;
}

.image-panel.collapsed .image-viewport {
  display: none;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.panel-title {
  color: #fff;
  font-size: 13px;
  font-weight: 500;
}

.original-panel .panel-title {
  color: #ff6b6b;
}

.translated-panel .panel-title {
  color: #00ff88;
}

.panel-toggle {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.panel-toggle:hover {
  background: rgba(255, 255, 255, 0.2);
}

/* 图片视口 */
.image-viewport {
  flex: 1;
  overflow: hidden;
  position: relative;
  cursor: grab;
  background-color: #0d1b2a;
  background-image:
    linear-gradient(45deg, #1a1a2e 25%, transparent 25%),
    linear-gradient(-45deg, #1a1a2e 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #1a1a2e 75%),
    linear-gradient(-45deg, transparent 75%, #1a1a2e 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
}

.image-viewport:active {
  cursor: grabbing;
}

.image-viewport:focus {
  outline: 2px solid #3498db;
  outline-offset: -2px;
}

/* 图片画布包装器 */
.image-canvas-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  transform-origin: 0 0;
  will-change: transform;
}

.image-canvas-wrapper img {
  display: block;
  max-width: none;
  user-select: none;
  -webkit-user-drag: none;
  pointer-events: none;
}

/* 分隔条 */
.panel-divider {
  width: 8px;
  background: #0f0f23;
  cursor: col-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
  flex-shrink: 0;
}

.panel-divider:hover {
  background: #3498db;
}

.divider-handle {
  color: #444;
  font-size: 12px;
  writing-mode: vertical-lr;
  user-select: none;
}

.panel-divider:hover .divider-handle {
  color: #fff;
}

/* ============ 底部编辑面板 ============ */
.edit-panel-container {
  background: var(--card-bg-color, #fff);
  border-left: 1px solid var(--border-color, #e0e0e0);
  min-height: 0;
  display: flex;
  flex-direction: row;
  flex: 0 0 600px;
  min-width: 520px;
  max-width: 65%;
  overflow: hidden;
}

.panel-resize-handle {
  background: var(--bg-color, #f0f0f0);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
  font-size: 10px;
  letter-spacing: 3px;
  transition: background 0.2s;
  flex-shrink: 0;
}

.panel-resize-handle.vertical {
  width: 10px;
  cursor: ew-resize;
  writing-mode: vertical-rl;
}

.panel-resize-handle:hover {
  background: #3498db;
  color: #fff;
}

.edit-panel-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 15px;
  overflow: auto;
  min-height: 0;
}

.text-block {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
}

.text-column-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--border-color, #e9ecef);
}

.column-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-color, #495057);
}

.original-text-column .column-title {
  color: #e74c3c;
}

.translated-text-column .column-title {
  color: #27ae60;
}

.re-ocr-btn,
.re-translate-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--bg-color, #f8f9fa);
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.re-ocr-btn:hover,
.re-translate-btn:hover {
  background: #3498db;
  color: #fff;
}

.text-editor {
  flex: 1;
  width: 100%;
  min-height: 60px;
  padding: 12px;
  border: 2px solid var(--border-color, #e9ecef);
  border-radius: 8px;
  font-size: 15px;
  line-height: 1.6;
  resize: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  box-sizing: border-box;
  font-family: inherit;
}

.text-editor:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
}

.original-editor {
  background: #fff8f8;
  font-family: 'Noto Sans JP', 'Yu Gothic', 'MS Gothic', sans-serif;
}

.translated-editor {
  background: #f8fff8;
}

.text-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.text-actions button {
  padding: 6px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: var(--card-bg-color, white);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.text-actions button:hover {
  background: var(--bg-color, #f8f9fa);
  border-color: #adb5bd;
}

.apply-text-btn {
  background: #27ae60 !important;
  color: white !important;
  border-color: #27ae60 !important;
}

.apply-text-btn:hover {
  background: #219a52 !important;
}

/* ============ 样式设置区域 ============ */
.style-settings-section {
  width: 100%;
  padding: 16px;
  background: #f5f6fb;
  border-radius: 10px;
  border: 1px solid rgba(82, 92, 105, 0.12);
  overflow-y: auto;
  box-sizing: border-box;
}

/* ============ 操作按钮 ============ */
.edit-action-buttons {
  display: flex;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color, #e9ecef);
  margin-top: 12px;
}

.edit-action-buttons button {
  flex: 1;
  padding: 8px 12px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-apply {
  background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
  color: white;
}

.btn-apply:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
}

.btn-apply-all {
  background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
  color: white;
}

.btn-apply-all:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

.btn-reset {
  background: var(--bg-color, #f0f0f0);
  color: #6c757d;
  border: 1px solid var(--border-color, #dee2e6);
}

.btn-reset:hover {
  background: #e9ecef;
}

/* ============ 暗色模式适配 ============ */
[data-theme="dark"] .edit-toolbar {
  background: linear-gradient(135deg, #0d1b2a 0%, #1a1a2e 100%);
}

[data-theme="dark"] .edit-panel-container {
  background: #2d3748;
  border-color: #4a5568;
}

[data-theme="dark"] .panel-resize-handle {
  background: #1a202c;
}

[data-theme="dark"] .text-editor {
  background: #1a202c;
  border-color: #4a5568;
  color: #e2e8f0;
}

[data-theme="dark"] .original-editor {
  background: #2d1f1f;
}

[data-theme="dark"] .translated-editor {
  background: #1f2d1f;
}

[data-theme="dark"] .style-settings-section {
  background: #1a202c;
  border-color: #4a5568;
}

/* ============ 气泡操作工具组 ============ */
.annotation-tools {
  display: flex;
  align-items: center;
  gap: 4px;
}

.annotation-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 6px 10px;
  border: 1px solid rgba(255, 193, 7, 0.5);
  border-radius: 6px;
  background: rgba(255, 193, 7, 0.15);
  color: #ffc107;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.annotation-btn:hover {
  background: rgba(255, 193, 7, 0.3);
  border-color: #ffc107;
}

.annotation-btn.active {
  background: rgba(255, 193, 7, 0.5);
  border-color: #ffc107;
  box-shadow: 0 0 8px rgba(255, 193, 7, 0.4);
}

.annotation-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ============ 绘制模式 ============ */
.drawing-mode .image-viewport {
  cursor: crosshair !important;
}

.drawing-mode .image-canvas-wrapper {
  cursor: crosshair !important;
}

body.middle-button-drawing,
body.middle-button-drawing * {
  cursor: crosshair !important;
}

.drawing-rect-edit {
  position: absolute;
  border: 2px dashed #00d4ff;
  background: rgba(0, 212, 255, 0.1);
  box-sizing: border-box;
  pointer-events: none;
  z-index: 25;
}

.drawing-mode-hint {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  padding: 6px 16px;
  background: rgba(255, 193, 7, 0.9);
  color: #000;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  z-index: 30;
  pointer-events: none;
  white-space: nowrap;
}

/* ============ 笔刷工具样式 ============ */
.brush-btn {
  position: relative;
}

.brush-btn.active {
  background: rgba(76, 175, 80, 0.3) !important;
  border-color: #4CAF50 !important;
  color: #4CAF50 !important;
}

#restoreBrushBtn.active {
  background: rgba(33, 150, 243, 0.3) !important;
  border-color: #2196F3 !important;
  color: #2196F3 !important;
}

.brush-size-display {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.7);
  padding: 4px 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  margin-left: 5px;
}

.edit-workspace.brush-mode-active .image-viewport {
  cursor: none !important;
}

.edit-workspace.brush-mode-active[data-brush-mode="repair"] .image-viewport {
  background: rgba(76, 175, 80, 0.05);
}

.edit-workspace.brush-mode-active[data-brush-mode="restore"] .image-viewport {
  background: rgba(33, 150, 243, 0.05);
}

.edit-workspace.brush-mode-active::before {
  content: '';
  position: fixed;
  top: 60px;
  left: 50%;
  transform: translateX(-50%);
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  z-index: 10000;
  pointer-events: none;
  animation: brushModeHint 0.3s ease;
}

.edit-workspace.brush-mode-active[data-brush-mode="repair"]::before {
  content: '修复笔刷 - 滚轮调整大小';
  background: rgba(76, 175, 80, 0.9);
  color: white;
}

.edit-workspace.brush-mode-active[data-brush-mode="restore"]::before {
  content: '还原笔刷 - 滚轮调整大小';
  background: rgba(33, 150, 243, 0.9);
  color: white;
}

@keyframes brushModeHint {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}

.brush-cursor {
  pointer-events: none;
  mix-blend-mode: normal;
}

.edit-workspace.brush-mode-active .image-canvas-wrapper {
  pointer-events: auto;
}

[data-theme="dark"] .brush-size-display {
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.8);
}

/* ============ 布局切换按钮 ============ */
.layout-toggle-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.layout-toggle-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.layout-toggle-btn.active {
  background: rgba(102, 126, 234, 0.5);
}

/* ============ 上下布局模式 ============ */
.edit-workspace.layout-vertical .edit-main-layout {
  flex-direction: column;
}

.edit-workspace.layout-vertical .image-comparison-container {
  flex: 1;
  min-height: 0;
}

.edit-workspace.layout-vertical .edit-thumbnails-panel {
  position: absolute;
  top: 90px;
  right: 10px;
  bottom: auto;
  left: auto;
  width: 80px;
  max-height: calc(100% - 100px - 45%);
  padding: 8px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 10px;
  z-index: 100;
  border-bottom: none;
  display: flex;
  flex-direction: column;
}

.edit-workspace.layout-vertical .thumbnails-scroll {
  flex-direction: column;
  overflow-y: auto;
  overflow-x: hidden;
  height: 100%;
  gap: 8px;
}

.edit-workspace.layout-vertical .edit-thumbnail-item {
  width: 64px;
  height: 85px;
}

.edit-workspace.layout-vertical .edit-panel-container {
  flex: 0 0 auto;
  width: 100%;
  max-width: 100%;
  min-width: 100%;
  max-height: 45%;
  min-height: 200px;
  border-left: none;
  border-top: 1px solid var(--border-color, #e0e0e0);
  flex-direction: column;
}

.edit-workspace.layout-vertical .edit-panel-content {
  flex-direction: row;
  flex-wrap: wrap;
  gap: 15px;
  overflow-x: auto;
  overflow-y: auto;
  padding: 12px 15px;
}

.edit-workspace.layout-vertical .text-block {
  flex: 1 1 300px;
  min-width: 280px;
  max-width: 450px;
}

.edit-workspace.layout-vertical .text-editor {
  min-height: 80px;
  max-height: 150px;
}

.edit-workspace.layout-vertical .style-settings-section {
  flex: 1 1 350px;
  min-width: 320px;
  max-width: 600px;
  max-height: none;
  overflow-y: visible;
}

.edit-workspace.layout-vertical .panel-resize-handle.vertical {
  width: 100%;
  height: 10px;
  cursor: ns-resize;
  writing-mode: horizontal-tb;
}

.edit-workspace.layout-vertical.thumbnails-visible .image-comparison-container {
  margin-right: 95px;
}

.edit-workspace.layout-vertical .office-toolbar {
  flex-direction: row;
  flex-wrap: wrap;
  align-items: flex-start;
}

.edit-workspace.layout-vertical .toolbar-row {
  flex-wrap: nowrap;
}

.edit-workspace.layout-vertical .edit-action-buttons {
  flex-wrap: wrap;
  justify-content: flex-start;
}

.edit-workspace.layout-vertical .edit-action-buttons button {
  flex: 0 0 auto;
  min-width: 100px;
}

[data-theme="dark"] .edit-workspace.layout-vertical .edit-panel-container {
  border-top-color: #4a5568;
}

/* 过渡动画 */
.edit-workspace {
  transition: none;
}

.edit-main-layout {
  transition: flex-direction 0.3s ease;
}

.edit-panel-container {
  transition: flex 0.3s ease, max-height 0.3s ease, border 0.3s ease;
}

.edit-thumbnails-panel {
  transition: all 0.3s ease;
}

/* ============ 编辑模式进度条 ============ */
.edit-progress-container {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 20px;
  min-width: 200px;
  max-width: 350px;
  animation: progressFadeIn 0.3s ease;
}

@keyframes progressFadeIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.edit-progress-info {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.edit-progress-text {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  font-weight: 500;
}

.edit-progress-count {
  font-size: 12px;
  color: #00ff88;
  font-weight: 600;
  font-family: 'Consolas', 'Monaco', monospace;
}

.edit-progress-bar {
  flex: 1;
  height: 6px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
  overflow: hidden;
  min-width: 80px;
}

.edit-progress-fill {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, #00ff88, #00d4ff);
  border-radius: 3px;
  transition: width 0.3s ease;
  box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
}

.edit-progress-fill.animating {
  background: linear-gradient(90deg, #00ff88, #00d4ff, #00ff88);
  background-size: 200% 100%;
  animation: progressShine 1.5s ease-in-out infinite;
}

@keyframes progressShine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.edit-progress-container.completed .edit-progress-fill {
  background: #00ff88;
}

.edit-progress-container.completed .edit-progress-text {
  color: #00ff88;
}

/* ============ 响应式调整 ============ */
@media (max-width: 1200px) {
  .style-settings-section {
    flex: none;
    max-width: none;
    width: 100%;
  }
}

@media (max-width: 768px) {
  .image-comparison-container {
    flex-direction: column;
  }

  .panel-divider {
    width: 100%;
    height: 8px;
    cursor: ns-resize;
  }

  .divider-handle {
    writing-mode: horizontal-tb;
  }
}
</style>
