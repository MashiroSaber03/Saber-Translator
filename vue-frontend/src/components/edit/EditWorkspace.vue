<template>
  <div
    v-if="isEditModeActive"
    class="edit-workspace"
    :class="[
      `edit-workspace--layout-${layoutMode}`,
      { 'edit-workspace--drawing-mode': isDrawingMode },
      { 'edit-workspace--brush-mode-active': !!brushMode }
    ]"
    :data-brush-mode="brushMode || undefined"
    tabindex="0"
    ref="workspaceRef"
  >
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
      :is-repair-loading="isRepairLoading"
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
      @exit-edit-mode="handleExitToolbarAction"
      @auto-detect-bubbles="autoDetectBubbles"
      @detect-all-images="detectAllImages"
      @translate-with-bubbles="translateWithCurrentBubbles"
      @toggle-drawing-mode="toggleDrawingMode"
      @delete-selected-bubbles="deleteSelectedBubbles"
      @repair-selected-bubble="handleRepairSelectedBubble"
      @activate-repair-brush="activateRepairBrush"
      @activate-restore-brush="activateRestoreBrush"
      @apply-and-next="applyAndNext"
    />

    <EditThumbnailPanel
      :visible="showThumbnails"
      :images="images"
      :current-image-index="currentImageIndex"
      @switch-to-image="switchToImage"
    />

    <EditImageComparison
      ref="imageComparisonRef"
      :view-mode="viewMode"
      :layout-mode="layoutMode"
      :current-image="currentImage"
      :bubbles="bubbles"
      :selected-bubble="selectedBubble"
      :selected-bubble-index="selectedBubbleIndex"
      :selected-indices="selectedIndices"
      :scale="scale"
      :original-scale="originalScale"
      :is-drawing-mode="isDrawingMode"
      :brush-mode="brushMode"
      :current-image-width="currentImageWidth"
      :current-image-height="currentImageHeight"
      :current-drawing-rect="currentDrawingRect"
      :drawing-rect-style="getDrawingRectStyle()"
      :original-transform-style="originalTransformStyle"
      :translated-transform-style="translatedTransformStyle"
      :is-ocr-loading="isOcrLoading"
      :is-translate-loading="isTranslateLoading"
      @wheel-panel="handleWheel"
      @mouse-down-panel="handleMouseDown"
      @image-load="handleImageLoad"
      @fit-to-screen="fitToScreen"
      @start-divider-drag="startDividerDrag"
      @start-panel-resize="startPanelResize"
      @bubble-select="handleBubbleSelect"
      @bubble-multi-select="handleBubbleMultiSelect"
      @bubble-drag-start="handleBubbleDragStart"
      @bubble-drag-end="handleBubbleDragEnd"
      @bubble-resize-start="handleBubbleResizeStart"
      @bubble-resize-end="handleBubbleResizeEnd"
      @bubble-rotate-start="handleBubbleRotateStart"
      @bubble-rotate-end="handleBubbleRotateEnd"
      @draw-bubble="handleDrawBubble"
      @bubble-update="handleBubbleUpdateWithSync"
      @apply-to-all-style="handleApplyStyleToAllBubbles"
      @ocr-recognize="handleOcrRecognize"
      @re-translate="handleReTranslateBubble"
      @reset-current="handleResetCurrentBubble"
    />
  </div>
</template>

<script setup lang="ts">

import EditImageComparison from './EditImageComparison.vue'
import EditToolbar from './EditToolbar.vue'
import EditThumbnailPanel from './EditThumbnailPanel.vue'
import { useEditWorkspace, type EditWorkspaceEmit, type EditWorkspaceProps } from './useEditWorkspace'

const props = defineProps<EditWorkspaceProps>()
const emit = defineEmits<EditWorkspaceEmit>()

const {
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
  isOcrLoading,
  isTranslateLoading,
  isRepairLoading,
  scale,
  originalScale,
  originalTransformStyle,
  translatedTransformStyle,
  isDrawingMode,
  currentDrawingRect,
  handleBubbleSelect,
  handleBubbleMultiSelect,
  handleBubbleDragStart,
  handleBubbleDragEnd,
  handleBubbleResizeStart,
  handleBubbleResizeEnd,
  handleBubbleRotateStart,
  handleBubbleRotateEnd,
  toggleDrawingMode,
  handleDrawBubble,
  getDrawingRectStyle,
  deleteSelectedBubbles,
  brushMode,
  brushSize,
  mouseX,
  mouseY,
  isProcessing,
  progressText,
  progressCurrent,
  progressTotal,
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
  handleResetCurrentBubble,
  handleOcrRecognize,
  handleReTranslateBubble,
  handleRepairSelectedBubble,
  activateRepairBrush,
  activateRestoreBrush,
  applyAndNext,
  autoDetectBubbles,
  detectAllImages,
  translateWithCurrentBubbles,
} = useEditWorkspace(props, emit)
</script>

<style scoped>
.edit-workspace {
  --edit-workspace-shell-background: var(--color-surface-inverse);

  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: var(--edit-workspace-shell-background);
  z-index: var(--z-overlay);
  overflow: hidden;
  margin: 0;
  border-radius: 0;
  flex-shrink: 0;
  transition: none;
}
</style>
