<!--
  编辑模式工作区组件
  提供双图对照、气泡编辑、笔刷工具等功能
-->
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

    <!-- 缩略图面板 - 使用拆分的组件 -->
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
/* ===================================
   编辑模式样式 - 当前样式
   =================================== */

/* ============ 编辑工作区 - 全屏覆盖 ============ */
.edit-workspace {
  /* owner tokens: edit-workspace */
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
}

/* ============ 顶部工具栏 - 双行布局 ============ */
.edit-toolbar-wrapper {
  flex-shrink: 0;
  background: linear-gradient(135deg, var(--edit-shell-start) 0%, var(--edit-shell-end) 100%);
  border-bottom: 1px solid var(--edit-workspace-shell-border-default);
}

.edit-toolbar {
  display: flex;
  align-items: center;
  padding: 8px 15px;
  gap: 10px;
}

.toolbar-row-1 {
  border-bottom: 1px solid var(--edit-workspace-shell-border-strong);
}

.toolbar-row-2 {
  background: var(--edit-workspace-shell-surface-base);
}

.toolbar-spacer {
  flex: 1;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: var(--color-surface-overlay-medium);
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
  background: var(--edit-shell-control);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.image-navigator .nav-btn:hover {
  background: var(--edit-shell-chip-active);
}

.image-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.image-indicator {
  color: var(--color-text-inverse);
  font-size: 14px;
  padding: 6px 12px;
  background: var(--edit-shell-chip);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.image-indicator:hover {
  background: var(--edit-shell-chip-hover);
}

.image-indicator span {
  font-weight: bold;
  color: var(--edit-workspace-shell-text-primary);
}

.thumb-toggle-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: var(--color-surface-overlay-light);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.thumb-toggle-btn:hover {
  background: var(--color-surface-overlay-medium);
}

.thumb-toggle-btn.active {
  background: var(--edit-shell-chip-active);
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
  background: var(--edit-shell-progress);
  padding: 10px 15px;
  border-bottom: 1px solid var(--edit-workspace-shell-border-default);
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
  background: var(--color-surface-overlay-light);
  border-radius: 3px;
}

.thumbnails-scroll::-webkit-scrollbar-thumb {
  background: var(--edit-workspace-shell-surface-raised);
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
  border-color: var(--edit-workspace-shell-border-muted);
  transform: scale(1.05);
}

.edit-thumbnail-item.active {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 10px var(--edit-workspace-shell-shadow-default);
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
  background: var(--edit-workspace-shell-surface-muted);
  color: var(--color-text-inverse);
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
  background: var(--color-surface-overlay-light);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.bubble-navigator .nav-btn:hover {
  background: var(--color-surface-overlay-medium);
}

.bubble-navigator .nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.bubble-indicator {
  color: var(--color-text-inverse);
  font-size: 14px;
  padding: 6px 12px;
  background: var(--color-surface-overlay-light);
  border-radius: 6px;
}

.bubble-indicator span {
  font-weight: bold;
  color: var(--edit-accent);
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
  background: var(--color-surface-overlay-light);
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s;
}

.view-controls button:hover {
  background: var(--color-surface-overlay-medium);
}

.view-controls .zoom-level {
  min-width: 50px;
  text-align: center;
  color: var(--color-text-inverse);
  font-size: 13px;
  padding: 0 8px;
}

.view-mode-btn {
  font-size: 18px;
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
.edit-toolbar .action-primary,
.quick-actions .action-primary {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, var(--edit-action-start) 0%, var(--edit-action-end) 100%);
  color: var(--edit-action-text);
  font-weight: 600;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.edit-toolbar .action-primary:hover,
.quick-actions .action-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--edit-shadow-action);
}
/* 次要按钮 */
.edit-toolbar .action-secondary,
.quick-actions .action-secondary {
  padding: 8px 16px;
  border: 1px solid var(--edit-workspace-canvas-border-default);
  border-radius: 6px;
  background: transparent;
  color: var(--color-text-inverse);
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.edit-toolbar .action-secondary:hover,
.quick-actions .action-secondary:hover {
  background: var(--color-surface-overlay-light);
  border-color: var(--edit-workspace-canvas-border-strong);
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
  border-bottom: 2px solid var(--color-border-muted, var(--edit-panel-divider));
}

.column-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--color-text-strong, var(--edit-panel-text));
}

.original-text-column .column-title {
  color: var(--color-text-danger-strong);
}

.translated-text-column .column-title {
  color: var(--edit-panel-success);
}

.re-ocr-btn,
.re-translate-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--color-surface-app, var(--edit-control-bg));
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.re-ocr-btn:hover,
.re-translate-btn:hover {
  background: var(--color-surface-accent);
  color: var(--color-text-inverse);
}

.text-editor {
  flex: 1;
  width: 100%;
  min-height: 60px;
  padding: 12px;
  border: 2px solid var(--color-border-muted, var(--edit-panel-divider));
  border-radius: 8px;
  font-size: 15px;
  line-height: 1.6;
  resize: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  font-family: inherit;
}

.text-editor:focus {
  outline: none;
  border-color: var(--color-border-accent);
  box-shadow: 0 0 0 3px var(--edit-workspace-canvas-shadow-default);
}

.original-editor {
  background: var(--color-surface-editor-original);
  font-family: var(--font-jp);
}

.translated-editor {
  background: var(--edit-translated-bg);
}

.text-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.text-actions button {
  padding: 6px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  background: var(--color-surface-card, white);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.text-actions button:hover {
  background: var(--color-surface-app, var(--edit-control-bg));
  border-color: var(--edit-muted-border-hover);
}

.text-actions .apply-text-btn {
  background: var(--color-surface-success);
  color: var(--color-text-inverse);
  border-color: var(--edit-workspace-canvas-border-muted);
  font-weight: 600;
}

.text-actions .apply-text-btn:hover {
  background: var(--edit-workspace-canvas-surface-base);
  border-color: var(--edit-workspace-canvas-border-subtle);
  color: var(--color-text-inverse);
}

/* ============ 样式设置区域 ============ */
.style-settings-section {
  width: 100%;
  padding: 16px;
  background: var(--edit-style-bg);
  border-radius: 10px;
  border: 1px solid var(--edit-workspace-canvas-border-hover);
  overflow-y: auto;
}

/* ============ 操作按钮 ============ */
.edit-action-buttons {
  display: flex;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted, var(--edit-panel-divider));
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
  background: linear-gradient(135deg, var(--color-surface-success) 0%, var(--edit-workspace-canvas-surface-raised) 100%);
  color: white;
}

.btn-apply:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--edit-workspace-canvas-shadow-raised);
}

.btn-apply-all {
  background: linear-gradient(135deg, var(--color-surface-accent) 0%, var(--edit-workspace-canvas-surface-muted) 100%);
  color: white;
}

.btn-apply-all:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--edit-workspace-canvas-shadow-floating);
}

.btn-reset {
  background: var(--color-surface-app, var(--edit-workspace-canvas-surface-subtle));
  color: var(--edit-workspace-canvas-text-primary);
  border: 1px solid var(--color-border-muted, var(--edit-workspace-canvas-border-active));
}

.btn-reset:hover {
  background: var(--edit-workspace-canvas-surface-hover);
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
  border: 1px solid var(--edit-workspace-canvas-border-focus);
  border-radius: 6px;
  background: var(--edit-workspace-canvas-surface-active);
  color: var(--edit-workspace-canvas-text-secondary);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.annotation-btn:hover {
  background: var(--edit-workspace-canvas-surface-selected);
  border-color: var(--edit-workspace-canvas-border-selected);
}

.annotation-btn.active {
  background: var(--edit-workspace-canvas-surface-overlay);
  border-color: var(--edit-workspace-canvas-border-selected);
  box-shadow: 0 0 8px var(--edit-workspace-canvas-shadow-strong);
}

.annotation-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.drawing-mode-hint {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  padding: 6px 16px;
  background: var(--edit-workspace-canvas-surface-inverse);
  color: var(--edit-workspace-canvas-text-muted);
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  z-index: var(--z-local-sticky);
  pointer-events: none;
  white-space: nowrap;
}

/* ============ 笔刷工具样式 ============ */
.brush-btn {
  position: relative;
}

.brush-btn.active {
  background: var(--edit-workspace-canvas-surface-contrast);
  border-color: var(--edit-workspace-canvas-border-danger);
  color: var(--edit-workspace-canvas-text-subtle);
}

.brush-size-display {
  font-size: 12px;
  color: var(--edit-workspace-canvas-text-supporting);
  padding: 4px 8px;
  background: var(--edit-shell-progress);
  border-radius: 4px;
  margin-left: 5px;
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
  z-index: var(--z-modal);
  pointer-events: none;
  animation: brushModeHint 0.3s ease;
}

.edit-workspace.brush-mode-active[data-brush-mode="repair"]::before {
  content: '修复笔刷 - 滚轮调整大小';
  background: var(--edit-workspace-canvas-surface-tint);
  color: white;
}

.edit-workspace.brush-mode-active[data-brush-mode="restore"]::before {
  content: '还原笔刷 - 滚轮调整大小';
  background: var(--edit-workspace-canvas-surface-soft);
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

/* ============ 布局切换按钮 ============ */
.layout-toggle-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  background: var(--color-surface-overlay-light);
  color: var(--color-text-inverse);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.layout-toggle-btn:hover {
  background: var(--color-surface-overlay-medium);
}

.layout-toggle-btn.active {
  background: var(--edit-workspace-sidebars-surface-base);
}

/* ============ 上下布局模式 ============ */
.edit-workspace.layout-vertical .edit-thumbnails-panel {
  position: absolute;
  top: 90px;
  right: 10px;
  bottom: auto;
  left: auto;
  width: 80px;
  max-height: calc(100% - 100px - 45%);
  padding: 8px;
  background: var(--edit-workspace-sidebars-surface-raised);
  border-radius: 10px;
  z-index: var(--z-dropdown);
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

/* 过渡动画 */
.edit-workspace {
  transition: none;
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
  background: var(--edit-workspace-sidebars-surface-muted);
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
  color: var(--edit-workspace-sidebars-text-primary);
  font-weight: 500;
}

.edit-progress-count {
  font-size: 12px;
  color: var(--edit-workspace-sidebars-text-secondary);
  font-weight: 600;
  font-family: var(--font-mono);
}

.edit-progress-bar {
  flex: 1;
  height: 6px;
  background: var(--color-surface-overlay-light-soft);
  border-radius: 3px;
  overflow: hidden;
  min-width: 80px;
}

.edit-progress-fill {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--edit-workspace-sidebars-surface-subtle), var(--edit-workspace-sidebars-surface-hover));
  border-radius: 3px;
  transition: width 0.3s ease;
  box-shadow: 0 0 8px var(--edit-workspace-sidebars-shadow-default);
}

.edit-progress-fill.animating {
  background: linear-gradient(90deg, var(--edit-workspace-sidebars-surface-subtle), var(--edit-workspace-sidebars-surface-hover), var(--edit-workspace-sidebars-surface-subtle));
  background-size: 200% 100%;
  animation: progressShine 1.5s ease-in-out infinite;
}

@keyframes progressShine {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

.edit-progress-container.completed .edit-progress-fill {
  background: var(--edit-workspace-sidebars-surface-subtle);
}

.edit-progress-container.completed .edit-progress-text {
  color: var(--edit-workspace-sidebars-text-secondary);
}

/* ============ 响应式调整 ============ */
@media (--breakpoint-2xl-down) {
  .style-settings-section {
    flex: none;
    max-width: none;
    width: 100%;
  }
}
</style>
