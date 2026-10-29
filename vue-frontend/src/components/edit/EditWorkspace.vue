<template>
  <div
    class="edit-workspace"
    :class="{ 'edit-workspace--picking-color': isPickingColor }"
    tabindex="0"
    ref="workspaceRef"
    @pointerdown.capture="handleWorkspacePointerDown"
  >
    <div class="edit-workspace__content">
      <EditToolbar
        v-show="!fineActive"
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
        :scale="toolbarScale"
        :is-drawing-mode="isDrawingMode"
        :has-selection="hasSelection"
        :brush-mode="brushMode"
        :clone-source="cloneSource"
        :brush-size="brushSize"
        :brush-shape="brushShape"
        @brush-shape="brushShape = $event"
        :mouse-x="mouseX"
        :mouse-y="mouseY"
        :is-busy="isBusy"
        :is-repair-loading="isRepairLoading"
        :mask-mode="maskEditor.active.value"
        :can-preview-clean="!!currentImage?.cleanAssetUrl"
        :show-clean-preview="showCleanPreview"
        :original-text-opacity="originalTextOpacity"
        :translated-text-opacity="translatedTextOpacity"
        :can-overlay-source="viewMode !== 'original' && !!currentImage?.sourceAssetUrl"
        :can-fade-translated="viewMode !== 'original' && !showCleanPreview && !!currentImage?.translatedAssetUrl && !!currentImage?.cleanAssetUrl"
        @go-previous-image="goToPreviousImage"
        @go-next-image="goToNextImage"
        @toggle-thumbnails="toggleThumbnails"
        @select-previous-bubble="selectPreviousBubble"
        @select-next-bubble="selectNextBubble"
        @toggle-layout="toggleLayout"
        @toggle-view-mode="toggleViewMode"
        @toggle-clean-preview="cleanPreviewRequested = !showCleanPreview"
        @update-original-text-opacity="originalTextOpacity = $event"
        @update-translated-text-opacity="translatedTextOpacity = $event"
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
        @duplicate-selected-bubbles="duplicateSelectedBubbles"
        @repair-selected-bubble="handleRepairSelectedBubble"
        @activate-repair-brush="activateRepairBrush"
        @activate-restore-brush="activateRestoreBrush"
        @activate-clone-brush="activateCloneBrush"
        @toggle-mask-mode="toggleMaskMode"
        @apply-and-next="applyAndNext"
        @pointer-action-complete="focusWorkspaceAfterPointer"
      >
        <template #workspace-tools>
          <div class="edit-workspace__modes edit-workspace__modes--inline">
            <UiButton size="sm" variant="primary" :disabled="modeSwitching" @click="setFineMode(false)">普通编辑</UiButton>
            <UiButton size="sm" variant="inverse" :disabled="modeSwitching || isBusy" @click="setFineMode(true)">精细编辑</UiButton>
            <UiIconButton variant="inverse" size="sm" label="撤销页面操作" :title="pageHistory.error.value || '撤销当前页上一次操作 (Ctrl+Z)；输入框内撤销文字输入'" :disabled="isBusy || !pageHistory.canUndo.value || maskEditor.dirty.value" @click="undoPage"><UiIcon :name="pageHistory.busy.value ? 'loading' : 'rotate-ccw'" size="16" /></UiIconButton>
            <UiIconButton variant="inverse" size="sm" label="重做页面操作" title="重做 (Ctrl+Shift+Z / Ctrl+Y)" :disabled="isBusy || !pageHistory.canRedo.value || maskEditor.dirty.value" @click="redoPage"><UiIcon name="rotate-cw" size="16" /></UiIconButton>
          </div>
          <UiButton variant="inverse" size="xs" class="edit-workspace__save-status" :class="{ 'edit-workspace__save-status--error': renderError }" :disabled="isBusy" title="自动保存；点击重试保存与出图 (Ctrl+S)" @click="retrySave">{{ pageHistory.busy.value ? '正在恢复…' : saveStatus }}</UiButton>
        </template>
        <template #mask-tools>
          <EditMaskPanel
            :tool="brushMode ? maskEditor.tool.value : null" :visibility="maskEditor.visibility.value" :preview="maskEditor.state.value" :busy="isBusy"
            :loading="maskEditor.loading.value" :saving="maskEditor.saving.value" :applying="maskEditor.applying.value"
            :dirty="maskEditor.dirty.value" :error="maskEditor.error.value"
            :inpaint-method="currentInpaintMethod" :fill-color="currentFillColor" :brush-size="brushSize"
            @method="handleBubbleUpdateWithSync({ inpaintMethod: $event })"
            @fill-color="handleBubbleUpdateWithSync({ fillColor: $event })"
            @tool="setMaskTool" @visibility="maskEditor.visibility.value = $event" @apply="maskEditor.apply"
            @close="toggleMaskMode" @retry="maskEditor.retry" @discard="maskEditor.discard"
          />
        </template>
      </EditToolbar>

      <div v-if="colorPickField" class="edit-workspace__color-pick-status" role="status">
        <span>设置{{ BUBBLE_COLOR_LABELS[colorPickField] }}：点击原图或结果图中的可见像素取色，按 Esc 取消。</span>
        <UiButton size="sm" variant="secondary" @click="cancelColorPick">取消取色</UiButton>
      </div>

      <EditThumbnailPanel
        v-show="!fineActive"
        :visible="showThumbnails"
        :images="images"
        :current-image-index="currentImageIndex"
        :is-busy="isBusy"
        @switch-to-image="switchToImage"
      />

      <EditImageComparison
        v-show="!fineActive"
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
        :clone-source="cloneSource"
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
        :mask-overlay="maskEditor.active.value ? maskEditor.overlay.value : null"
        :mask-mode="maskEditor.active.value"
        :show-clean-preview="showCleanPreview"
        :original-text-opacity="originalTextOpacity"
        :translated-text-opacity="translatedTextOpacity"
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
        @format-text="handleFormatText"
        @fit-text="fitSelectedText"
        @move-text="moveSelectedText"
        @apply-style="handleApplyStyleToAllBubbles"
        @ocr-recognize="handleOcrRecognize"
        @re-translate="handleReTranslateBubble"
        @pick-color="startImageColorPick"
      >
        <template #editor-header>
          <ChapterTextSearch v-if="currentImage?.chapterId" :chapter-id="currentImage.chapterId" :disabled="isBusy || !!brushMode" :prepare="persistCurrentDocument" @busy="isSearchReplacing = $event" @locate="locateSearchResult" @changed="refreshAfterTextReplace" />
        </template>
      </EditImageComparison>
      <FineEditor v-if="fineMounted && currentImage?.id" ref="fineEditorRef" :inert="switching || !fineActive || undefined" :page-id="currentImage.id" :active="fineActive" :inpaint-method="currentImage.inpaintMethod" :fill-color="currentImage.fillColor" @busy="fineBusy = $event" @document-loaded="handleDocumentLoaded" @method="handleBubbleUpdateWithSync({ inpaintMethod: $event }, true)" @fill-color="handleBubbleUpdateWithSync({ fillColor: $event }, true)">
        <template #toolbar-start>
          <div class="edit-workspace__modes edit-workspace__modes--inline">
            <UiButton size="sm" variant="inverse" :disabled="modeSwitching" @click="setFineMode(false)">普通编辑</UiButton>
            <UiButton size="sm" variant="primary" :disabled="modeSwitching" @click="setFineMode(true)">精细编辑</UiButton>
          </div>
          <div class="edit-workspace__page-nav" aria-label="图片导航">
            <UiIconButton size="sm" variant="inverse" label="上一张" :disabled="!canGoPrevious || modeSwitching" @click="navigateFine(false)"><UiIcon name="chevron-left" :size="17" /></UiIconButton>
            <span class="edit-workspace__page-index">{{ currentImageIndex + 1 }} / {{ imageCount }}</span>
            <UiIconButton size="sm" variant="inverse" label="下一张" :disabled="!canGoNext || modeSwitching" @click="navigateFine(true)"><UiIcon name="chevron-right" :size="17" /></UiIconButton>
          </div>
        </template>
        <template #toolbar-end>
          <UiButton size="sm" variant="inverse" :disabled="modeSwitching" @click="exitFine">退出编辑</UiButton>
        </template>
      </FineEditor>
    </div>
  </div>
</template>

<script setup lang="ts">

import { computed, ref } from 'vue'
import { onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'
import EditImageComparison from './EditImageComparison.vue'
import EditToolbar from './EditToolbar.vue'
import EditThumbnailPanel from './EditThumbnailPanel.vue'
import EditMaskPanel from './EditMaskPanel.vue'
import FineEditor from './FineEditor.vue'
import ChapterTextSearch from './ChapterTextSearch.vue'
import { useEditWorkspace, type EditWorkspaceEmit } from './useEditWorkspace'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { BUBBLE_COLOR_LABELS } from './bubbleColorFields'
import { showToast } from '@/utils/toast'
import { flushPageDocument } from '@/services/pageDocumentPersistence'

const emit = defineEmits<EditWorkspaceEmit>()
const fineActive = ref(false)
const fineMounted = ref(false)
const switching = ref(false)
const fineBusy = ref(false)
const modeSwitching = computed(() => switching.value || fineBusy.value)
const fineEditorRef = ref<InstanceType<typeof FineEditor>>()

async function setFineMode(value: boolean): Promise<void> {
  if (value === fineActive.value || modeSwitching.value) return
  if (!guardUnsavedMask()) return
  switching.value = true
  try {
    if (fineActive.value) {
      await fineEditorRef.value?.saveNow()
      if (currentImage.value?.id) await flushPageDocument(currentImage.value.id)
    } else await persistCurrentDocument()
    if (value) {
      if (maskEditor.active.value) toggleMaskMode()
      fineMounted.value = true
    }
    fineActive.value = value
  } catch (error) { showToast(error instanceof Error ? error.message : '请先保存当前修改', 'error') }
  finally { switching.value = false }
}
async function exitFine(): Promise<void> {
  switching.value = true
  try {
    await fineEditorRef.value?.saveNow()
    if (currentImage.value?.id) await flushPageDocument(currentImage.value.id)
    await handleExitToolbarAction()
  } catch (error) { showToast(error instanceof Error ? error.message : '请先保存当前修改', 'error') }
  finally { switching.value = false }
}
async function navigateFine(next: boolean): Promise<void> {
  switching.value = true
  try {
    await fineEditorRef.value?.saveNow()
    if (next) await goToNextImage(); else await goToPreviousImage()
  } catch (error) { showToast(error instanceof Error ? error.message : '请先保存当前修改', 'error') }
  finally { switching.value = false }
}

const {
  persistCurrentDocument,
  pageHistory,
  undoPage,
  redoPage,
  handleDocumentLoaded,
  currentInpaintMethod,
  currentFillColor,
  maskEditor,
  toggleMaskMode,
  setMaskTool,
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
  deleteSelectedBubbles,
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
  refreshAfterTextReplace,
  moveSelectedText,
  saveStatus,
  renderError,
  retrySave,
  handleOcrRecognize,
  handleReTranslateBubble,
  handleRepairSelectedBubble,
  activateRepairBrush,
  activateRestoreBrush,
  activateCloneBrush,
  applyAndNext,
  autoDetectBubbles,
  detectAllImages,
  translateWithCurrentBubbles,
} = useEditWorkspace(emit, computed(() => !fineActive.value))

const cleanPreviewRequested = ref(false)
const showCleanPreview = computed(() => cleanPreviewRequested.value && !!currentImage.value?.cleanAssetUrl)
const originalTextOpacity = ref(0)
const translatedTextOpacity = ref(100)

function guardUnsavedMask(): boolean {
  if (!maskEditor.dirty.value) return true
  showToast('请先保存当前修复范围，或放弃未保存的修改', 'warning')
  return false
}

function guardWorkspaceLeave(): boolean | Promise<boolean> {
  if (!guardUnsavedMask() || modeSwitching.value) return false
  if (!fineActive.value) return true
  switching.value = true
  const saved = fineEditorRef.value?.saveNow()
  return Promise.resolve(saved).then(() => true).catch(error => {
    showToast(error instanceof Error ? error.message : '请先保存当前修改', 'error')
    return false
  }).finally(() => { switching.value = false })
}
onBeforeRouteLeave(guardWorkspaceLeave)
onBeforeRouteUpdate(guardWorkspaceLeave)
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

.edit-workspace__content { position: relative; display: flex; flex: 1; min-height: 0; flex-direction: column; }

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

.edit-workspace__modes {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 7px 14px;
  color: var(--color-text-inverse);
  background: var(--color-surface-inverse);
  border-bottom: 1px solid var(--color-overlay-inverse-subtle);
  font-size: 12px;
}
.edit-workspace__modes--inline { flex-shrink: 0; flex-wrap: nowrap; padding: 0; background: transparent; border: 0; }
.edit-workspace__save-status.edit-workspace__save-status--error { color: var(--color-status-error-bright); }
.edit-workspace__save-status { width: 88px; max-width: 88px; flex-shrink: 0; overflow: hidden; white-space: nowrap; font-size: 11px; color: var(--color-action-success-bright); }
.edit-workspace__page-nav { display: flex; flex-shrink: 0; align-items: center; gap: 5px; padding: 0 10px; border-inline: 1px solid var(--color-overlay-inverse-muted); }
.edit-workspace__page-index { min-width: 50px; text-align: center; font-variant-numeric: tabular-nums; }

</style>
