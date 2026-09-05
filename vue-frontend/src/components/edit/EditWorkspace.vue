<template>
  <div
    class="edit-workspace"
    tabindex="0"
    ref="workspaceRef"
    @pointerdown.capture="handleWorkspacePointerDown"
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
      :is-busy="isBusy"
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
      @pointer-action-complete="focusWorkspaceAfterToolbarPointer"
    />

    <div v-if="colorPickField" class="edit-workspace__color-pick-status" role="status">
      <span>设置{{ BUBBLE_COLOR_LABELS[colorPickField] }}：点击原图或结果图中的可见像素取色，按 Esc 取消。</span>
      <UiButton size="sm" variant="secondary" @click="cancelColorPick">取消取色</UiButton>
    </div>

    <EditThumbnailPanel
      :visible="showThumbnails"
      :images="images"
      :current-image-index="currentImageIndex"
      :is-busy="isBusy"
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
      :is-busy="isBusy"
      :is-picking-color="isPickingColor"
      @wheel-panel="handleWheel"
      @mouse-down-panel="handleMouseDown"
      @image-load="handleImageLoad"
      @fit-to-screen="fitToScreen"
      @start-divider-drag="startDividerDrag"
      @start-panel-resize="startPanelResize"
      @bubble-select="handleBubbleSelect"
      @bubble-multi-select="handleBubbleMultiSelect"
      @bubble-drag-end="handleBubbleDragEnd"
      @bubble-resize-end="handleBubbleResizeEnd"
      @bubble-rotate-end="handleBubbleRotateEnd"
      @bubble-update="handleBubbleUpdateWithSync"
      @apply-to-all-style="handleApplyStyleToAllBubbles"
      @ocr-recognize="handleOcrRecognize"
      @re-translate="handleReTranslateBubble"
      @pick-color="startImageColorPick"
    />
  </div>
</template>

<script setup lang="ts">

import EditImageComparison from './EditImageComparison.vue'
import EditToolbar from './EditToolbar.vue'
import EditThumbnailPanel from './EditThumbnailPanel.vue'
import { useEditWorkspace, type EditWorkspaceEmit } from './useEditWorkspace'
import UiButton from '@/components/ui/UiButton.vue'
import { BUBBLE_COLOR_LABELS } from './bubbleColorFields'

const emit = defineEmits<EditWorkspaceEmit>()

const {
  workspaceRef,
  colorPickField,
  isPickingColor,
  startImageColorPick,
  cancelColorPick,
  handleWorkspacePointerDown,
  imageComparisonRef,
  focusWorkspaceAfterToolbarPointer,
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
  handleBubbleDragEnd,
  handleBubbleResizeEnd,
  handleBubbleRotateEnd,
  toggleDrawingMode,
  getDrawingRectStyle,
  deleteSelectedBubbles,
  brushMode,
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
  handleOcrRecognize,
  handleReTranslateBubble,
  handleRepairSelectedBubble,
  activateRepairBrush,
  activateRestoreBrush,
  applyAndNext,
  autoDetectBubbles,
  detectAllImages,
  translateWithCurrentBubbles,
} = useEditWorkspace(emit)
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

.edit-workspace__color-pick-status {
  position: absolute;
  right: 16px;
  bottom: 16px;
  left: 16px;
  z-index: var(--z-local-panel);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 16px;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  font-size: 13px;
}

</style>
