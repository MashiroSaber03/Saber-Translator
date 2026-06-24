<template>
  <div
    v-if="isEditModeActive"
    class="edit-workspace"
    :class="[
      `layout-${layoutMode}`,
      { 'drawing-mode': isDrawingMode },
      { 'brush-mode-active': !!brushMode }
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

    <EditExitSaveModal
      v-if="exitDialogState !== 'closed'"
      :state="exitDialogState"
      :message="exitSaveMessage"
      :error="exitDialogError"
      :progress-percent="exitSaveProgressPercent"
      :has-progress="exitSaveHasProgress"
      :current="exitSaveCurrent"
      :total="exitSaveTotal"
      @cancel="closeExitDialog"
      @exit-without-saving="exitWithoutSaving"
      @save-and-exit="saveAndExit"
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
      @re-render="handleReRender"
      @ocr-recognize="handleOcrRecognize"
      @re-translate="handleReTranslateBubble"
      @reset-current="handleResetCurrentBubble"
    />
  </div>
</template>

<script setup lang="ts">

import EditExitSaveModal from './EditExitSaveModal.vue'
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
  exitDialogState,
  exitSaveMessage,
  exitDialogError,
  exitSaveProgressPercent,
  exitSaveHasProgress,
  exitSaveCurrent,
  exitSaveTotal,
  closeExitDialog,
  exitWithoutSaving,
  saveAndExit,
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
  handleReRender,
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
  --edit-shell-start: #16213e;
  --edit-shell-end: #1a1a2e;
  --edit-shell-divider: rgba(255, 255, 255, .1);
  --edit-shell-divider-soft: rgba(255, 255, 255, .05);
  --edit-shell-overlay: rgba(0, 0, 0, .15);
  --edit-shell-chip: rgba(102, 126, 234, .2);
  --edit-shell-chip-hover: rgba(102, 126, 234, .4);
  --edit-shell-chip-active: rgba(102, 126, 234, .5);
  --edit-shell-control: rgba(102, 126, 234, .3);
  --edit-shell-progress: rgba(0, 0, 0, .3);
  --edit-shell-progress-muted: rgba(0, 0, 0, .3);
  --edit-accent: #0f8;
  --edit-action-start: #0f8;
  --edit-action-end: #00cc6a;
  --edit-action-text: #1a1a2e;
  --edit-action-border: rgba(255, 255, 255, .3);
  --edit-action-border-hover: rgba(255, 255, 255, .5);
  --edit-panel-divider: #e9ecef;
  --edit-panel-text: #495057;
  --edit-panel-success: #27ae60;
  --edit-control-bg: #f8f9fa;
  --edit-original-bg: var(--color-surface-editor-original);
  --edit-translated-bg: #f8fff8;
  --edit-style-bg: #f5f6fb;
  --edit-style-border: rgba(82, 92, 105, .12);
  --edit-toolbar-border: rgba(96, 110, 140, .22);
  --edit-toolbar-row-border: rgba(226, 232, 240, .9);
  --edit-toolbar-row-start: #fbfcff;
  --edit-toolbar-row-end: #f4f6ff;
  --edit-toolbar-label: #57607c;
  --edit-toolbar-divider: rgba(15, 23, 42, .08);
  --edit-input-border: #cfd6e4;
  --edit-input-text: #1f2430;
  --edit-input-border-hover: #8aa0f6;
  --edit-input-border-focus: #5b73f2;
  --edit-muted-border-hover: #adb5bd;
  --edit-shadow-focus-blue: rgba(52, 152, 219, .15);
  --edit-shadow-toolbar: rgba(15, 23, 42, .12);
  --edit-shadow-input-focus: rgba(88, 125, 255, .18);
  --edit-shadow-action: rgba(0, 255, 136, .3);
  --edit-workspace-canvas-border-default: rgba(255, 255, 255, .3);
  --edit-workspace-canvas-border-strong: rgba(255, 255, 255, .5);
  --edit-workspace-canvas-border-muted: #27ae60;
  --edit-workspace-canvas-border-subtle: #219a52;
  --edit-workspace-canvas-border-hover: rgba(82, 92, 105, .12);
  --edit-workspace-canvas-border-active: #dee2e6;
  --edit-workspace-canvas-border-focus: rgba(255, 193, 7, .5);
  --edit-workspace-canvas-border-selected: #ffc107;
  --edit-workspace-canvas-border-danger: #4caf50;
  --edit-workspace-canvas-shadow-default: rgba(52, 152, 219, .15);
  --edit-workspace-canvas-shadow-raised: rgba(39, 174, 96, .3);
  --edit-workspace-canvas-shadow-floating: rgba(52, 152, 219, .3);
  --edit-workspace-canvas-shadow-strong: rgba(255, 193, 7, .4);
  --edit-workspace-canvas-surface-base: #219a52;
  --edit-workspace-canvas-surface-raised: #2ecc71;
  --edit-workspace-canvas-surface-muted: #5dade2;
  --edit-workspace-canvas-surface-subtle: #f0f0f0;
  --edit-workspace-canvas-surface-hover: #e9ecef;
  --edit-workspace-canvas-surface-active: rgba(255, 193, 7, .15);
  --edit-workspace-canvas-surface-selected: rgba(255, 193, 7, .3);
  --edit-workspace-canvas-surface-overlay: rgba(255, 193, 7, .5);
  --edit-workspace-canvas-surface-inverse: rgba(255, 193, 7, .9);
  --edit-workspace-canvas-surface-contrast: rgba(76, 175, 80, .3);
  --edit-workspace-canvas-surface-tint: rgba(76, 175, 80, .9);
  --edit-workspace-canvas-surface-soft: rgba(33, 150, 243, .9);
  --edit-workspace-canvas-text-primary: #6c757d;
  --edit-workspace-canvas-text-secondary: #ffc107;
  --edit-workspace-canvas-text-muted: #000;
  --edit-workspace-canvas-text-subtle: #4caf50;
  --edit-workspace-canvas-text-supporting: rgba(255, 255, 255, .7);
  --edit-workspace-shell-border-default: rgba(255, 255, 255, .1);
  --edit-workspace-shell-border-strong: rgba(255, 255, 255, .05);
  --edit-workspace-shell-border-muted: rgba(255, 255, 255, .5);
  --edit-workspace-shell-shadow-default: rgba(102, 126, 234, .5);
  --edit-workspace-shell-surface-base: rgba(0, 0, 0, .15);
  --edit-workspace-shell-surface-raised: rgba(255, 255, 255, .3);
  --edit-workspace-shell-surface-muted: rgba(0, 0, 0, .7);
  --edit-workspace-shell-text-primary: #667eea;
  --edit-workspace-sidebars-shadow-default: rgba(0, 255, 136, .5);
  --edit-workspace-sidebars-surface-base: rgba(102, 126, 234, .5);
  --edit-workspace-sidebars-surface-raised: rgba(0, 0, 0, .7);
  --edit-workspace-sidebars-surface-muted: rgba(0, 0, 0, .3);
  --edit-workspace-sidebars-surface-subtle: #0f8;
  --edit-workspace-sidebars-surface-hover: #00d4ff;
  --edit-workspace-sidebars-text-primary: rgba(255, 255, 255, .9);
  --edit-workspace-sidebars-text-secondary: #0f8;

  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: var(--edit-shell-end);
  z-index: var(--z-overlay);
  overflow: hidden;
  margin: 0;
  border-radius: 0;
  flex-shrink: 0;
  transition: none;
}
</style>
