<script setup lang="ts">
import { ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { BubbleCoords, BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import BubbleEditor from './BubbleEditor.vue'
import BubbleOverlay from './BubbleOverlay.vue'

type ViewMode = 'dual' | 'original' | 'translated'
type LayoutMode = 'horizontal' | 'vertical'
type ViewportName = 'original' | 'translated'

defineProps<{
  viewMode: ViewMode
  layoutMode: LayoutMode
  currentImage: ImageData | null | undefined
  bubbles: BubbleState[]
  selectedBubble: BubbleState | null | undefined
  selectedBubbleIndex: number
  selectedIndices: number[]
  scale: number
  originalScale: number
  isDrawingMode: boolean
  brushMode: string | null
  currentImageWidth: number
  currentImageHeight: number
  currentDrawingRect: [number, number, number, number] | null
  drawingRectStyle: Record<string, string>
  originalTransformStyle: Record<string, string>
  translatedTransformStyle: Record<string, string>
  isOcrLoading: boolean
  isTranslateLoading: boolean
}>()

const emit = defineEmits<{
  wheelPanel: [event: WheelEvent, viewport: ViewportName]
  mouseDownPanel: [event: MouseEvent, viewport: ViewportName]
  imageLoad: [viewport: ViewportName]
  fitToScreen: []
  startDividerDrag: [event: MouseEvent]
  startPanelResize: [event: MouseEvent]
  bubbleSelect: [index: number]
  bubbleMultiSelect: [index: number]
  bubbleDragStart: [index: number, event: MouseEvent]
  bubbleDragEnd: [index: number, newCoords: BubbleCoords]
  bubbleResizeStart: [index: number, handle: string, event: MouseEvent]
  bubbleResizeEnd: [index: number, newCoords: BubbleCoords]
  bubbleRotateStart: [index: number, event: MouseEvent]
  bubbleRotateEnd: [index: number, angle: number]
  drawBubble: [rect: [number, number, number, number]]
  bubbleUpdate: [updates: Partial<BubbleState>]
  reRender: []
  ocrRecognize: [index: number]
  reTranslate: [index: number]
  resetCurrent: [index: number]
}>()

const originalViewportRef = ref<HTMLElement | null>(null)
const originalWrapperRef = ref<HTMLElement | null>(null)
const translatedViewportRef = ref<HTMLElement | null>(null)
const translatedWrapperRef = ref<HTMLElement | null>(null)
const editPanelRef = ref<HTMLElement | null>(null)
const originalPanelCollapsed = ref(false)
const translatedPanelCollapsed = ref(false)

defineExpose({
  originalViewportRef,
  originalWrapperRef,
  translatedViewportRef,
  translatedWrapperRef,
  editPanelRef,
})
</script>

<template>
  <div
    class="edit-main-layout"
    :class="[
      `layout-${layoutMode}`,
      { 'drawing-mode': isDrawingMode },
      { 'brush-mode-active': !!brushMode },
    ]"
    :data-brush-mode="brushMode || undefined"
  >
    <div class="image-comparison-container">
      <div
        v-show="viewMode !== 'translated'"
        class="image-panel original-panel"
        :class="{ collapsed: viewMode === 'translated' || originalPanelCollapsed }"
      >
        <div class="panel-header">
          <span class="panel-title">📖 原图 (日文)</span>
          <UiButton
            variant="toolbar"
            class="panel-toggle"
            title="折叠/展开"
            @click="originalPanelCollapsed = !originalPanelCollapsed"
          >
            {{ originalPanelCollapsed ? '+' : '−' }}
          </UiButton>
        </div>
        <div
          ref="originalViewportRef"
          class="image-viewport"
          @wheel.prevent="emit('wheelPanel', $event, 'original')"
          @mousedown="emit('mouseDownPanel', $event, 'original')"
          @dblclick="emit('fitToScreen')"
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
              @load="emit('imageLoad', 'original')"
            >
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
              @select="emit('bubbleSelect', $event)"
              @multi-select="emit('bubbleMultiSelect', $event)"
              @drag-start="(index, event) => emit('bubbleDragStart', index, event)"
              @drag-end="(index, newCoords) => emit('bubbleDragEnd', index, newCoords)"
              @resize-start="(index, handle, event) => emit('bubbleResizeStart', index, handle, event)"
              @resize-end="(index, newCoords) => emit('bubbleResizeEnd', index, newCoords)"
              @rotate-start="(index, event) => emit('bubbleRotateStart', index, event)"
              @rotate-end="(index, angle) => emit('bubbleRotateEnd', index, angle)"
              @draw-bubble="emit('drawBubble', $event)"
            />
            <div
              v-if="currentDrawingRect"
              class="drawing-rect-edit"
              :style="drawingRectStyle"
            ></div>
          </div>
        </div>
      </div>

      <div
        v-if="viewMode === 'dual'"
        class="panel-divider"
        :class="{ 'vertical-divider': layoutMode === 'vertical' }"
        @mousedown="emit('startDividerDrag', $event)"
      >
        <span class="divider-handle">⋮</span>
      </div>

      <div
        v-show="viewMode !== 'original'"
        class="image-panel translated-panel"
        :class="{ collapsed: viewMode === 'original' || translatedPanelCollapsed }"
      >
        <div class="panel-header">
          <span class="panel-title">📝 翻译图 (中文)</span>
          <UiButton
            variant="toolbar"
            class="panel-toggle"
            title="折叠/展开"
            @click="translatedPanelCollapsed = !translatedPanelCollapsed"
          >
            {{ translatedPanelCollapsed ? '+' : '−' }}
          </UiButton>
        </div>
        <div
          ref="translatedViewportRef"
          class="image-viewport"
          @wheel.prevent="emit('wheelPanel', $event, 'translated')"
          @mousedown="emit('mouseDownPanel', $event, 'translated')"
          @dblclick="emit('fitToScreen')"
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
              @load="emit('imageLoad', 'translated')"
            >
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
              @select="emit('bubbleSelect', $event)"
              @multi-select="emit('bubbleMultiSelect', $event)"
              @drag-start="(index, event) => emit('bubbleDragStart', index, event)"
              @drag-end="(index, newCoords) => emit('bubbleDragEnd', index, newCoords)"
              @resize-start="(index, handle, event) => emit('bubbleResizeStart', index, handle, event)"
              @resize-end="(index, newCoords) => emit('bubbleResizeEnd', index, newCoords)"
              @rotate-start="(index, event) => emit('bubbleRotateStart', index, event)"
              @rotate-end="(index, angle) => emit('bubbleRotateEnd', index, angle)"
              @draw-bubble="emit('drawBubble', $event)"
            />
            <div
              v-if="currentDrawingRect"
              class="drawing-rect-edit translated-drawing-rect"
              :style="drawingRectStyle"
            ></div>
          </div>
        </div>
      </div>
    </div>

    <div ref="editPanelRef" class="edit-panel-container">
      <div
        class="panel-resize-handle vertical"
        @mousedown="emit('startPanelResize', $event)"
      >
        ⋮⋮⋮
      </div>
      <BubbleEditor
        :bubble="selectedBubble"
        :bubble-index="selectedBubbleIndex"
        :is-ocr-loading="isOcrLoading"
        :is-translate-loading="isTranslateLoading"
        @update="emit('bubbleUpdate', $event)"
        @re-render="emit('reRender')"
        @ocr-recognize="emit('ocrRecognize', $event)"
        @re-translate="emit('reTranslate', $event)"
        @reset-current="emit('resetCurrent', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
/* ============ 双图对照区域 ============ */
.edit-main-layout {
  /* owner tokens: edit-image-comparison */
  --edit-image-comparison-panel-background: #16213e;
  --edit-image-comparison-panel-header-background: rgba(0, 0, 0, .3);
  --edit-image-comparison-panel-divider-border: rgba(255, 255, 255, .1);
  --edit-image-comparison-original-title-text: #ff6b6b;
  --edit-image-comparison-translated-title-text: #0f8;
  --edit-image-comparison-viewport-background: #0d1b2a;
  --edit-image-comparison-divider-background: #0f0f23;
  --edit-image-comparison-divider-handle-text: #444;
  --edit-image-comparison-resize-handle-background: #f0f0f0;
  --edit-image-comparison-drawing-rect-border: #00d4ff;
  --edit-image-comparison-drawing-rect-background: rgba(0, 212, 255, .1);
  --edit-image-comparison-repair-mode-background: rgba(76, 175, 80, .05);
  --edit-image-comparison-restore-mode-background: rgba(33, 150, 243, .05);

  display: flex;
  flex: 1;
  flex-direction: row;
  gap: 0;
  min-height: 0;
  transition: flex-direction 0.3s ease;
}

.image-comparison-container {
  display: flex;
  flex: 1 1 auto;
  gap: 0;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  padding: 8px;
}

.image-panel {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 150px;
  overflow: hidden;
  border-radius: 8px;
  background: var(--edit-image-comparison-panel-background);
  transition: flex 0.3s ease;
}

.image-panel.collapsed {
  flex: 0 0 40px;
  min-width: 40px;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--edit-image-comparison-panel-divider-border);
  background: var(--edit-image-comparison-panel-header-background);
}

.panel-title {
  color: var(--color-text-inverse);
  font-size: 13px;
  font-weight: 500;
}

.original-panel .panel-title {
  color: var(--edit-image-comparison-original-title-text);
}

.translated-panel .panel-title {
  color: var(--edit-image-comparison-translated-title-text);
}

.panel-toggle {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 4px;
  background: var(--color-overlay-inverse-subtle);
  color: var(--color-text-inverse);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.panel-toggle:hover {
  background: var(--color-overlay-inverse-muted);
}

.image-viewport {
  position: relative;
  flex: 1;
  overflow: hidden;
  background-color: var(--edit-image-comparison-viewport-background);
  backface-visibility: hidden;
  cursor: grab;
  transform: translateZ(0);
}

.image-viewport:active {
  cursor: grabbing;
}

.image-viewport:focus {
  outline: 2px solid var(--color-border-accent);
  outline-offset: -2px;
}

.image-canvas-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  backface-visibility: hidden;
  perspective: 1000px;
  transform-origin: 0 0;
  will-change: transform;
}

.image-canvas-wrapper img {
  display: block;
  max-width: none;
  backface-visibility: hidden;
  image-rendering: crisp-edges;
  pointer-events: none;
  transform: translateZ(0);
  user-select: none;
  -webkit-user-drag: none;
}

.panel-divider {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 8px;
  background: var(--edit-image-comparison-divider-background);
  cursor: col-resize;
  transition: background 0.2s;
}

.panel-divider:hover {
  background: var(--color-surface-accent);
}

.divider-handle {
  color: var(--edit-image-comparison-divider-handle-text);
  font-size: 12px;
  writing-mode: vertical-lr;
  user-select: none;
}

.panel-divider:hover .divider-handle {
  color: var(--color-text-inverse);
}

.edit-panel-container {
  display: flex;
  flex: 0 0 600px;
  flex-direction: row;
  min-width: 520px;
  min-height: 0;
  max-width: 65%;
  overflow: hidden;
  border-left: 1px solid var(--color-border-muted, var(--color-border-default));
  background: var(--color-surface-card, var(--color-surface-base));
  transition: flex 0.3s ease, max-height 0.3s ease, border 0.3s ease;
}

.panel-resize-handle {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: var(--color-surface-app, var(--edit-image-comparison-resize-handle-background));
  color: var(--color-text-muted);
  font-size: 10px;
  letter-spacing: 0;
  transition: background 0.2s;
}

.panel-resize-handle.vertical {
  width: 10px;
  cursor: ew-resize;
  writing-mode: vertical-rl;
}

.panel-resize-handle:hover {
  background: var(--color-surface-accent);
  color: var(--color-text-inverse);
}

.drawing-mode .image-viewport,
.drawing-mode .image-canvas-wrapper {
  cursor: crosshair;
}

.drawing-rect-edit {
  position: absolute;
  z-index: var(--z-local-popover);
  border: 2px dashed var(--edit-image-comparison-drawing-rect-border);
  background: var(--edit-image-comparison-drawing-rect-background);
  pointer-events: none;
}

.brush-mode-active .image-viewport {
  cursor: none;
}

.brush-mode-active[data-brush-mode="repair"] .image-viewport {
  background: var(--edit-image-comparison-repair-mode-background);
}

.brush-mode-active[data-brush-mode="restore"] .image-viewport {
  background: var(--edit-image-comparison-restore-mode-background);
}

.brush-mode-active .image-canvas-wrapper {
  pointer-events: auto;
}

.image-panel.collapsed .image-viewport {
  display: none;
}

.layout-vertical.edit-main-layout {
  flex-direction: column;
}

.layout-vertical .image-comparison-container {
  flex: 1;
  min-height: 0;
}

.layout-vertical .edit-panel-container {
  flex: 0 0 auto;
  flex-direction: column;
  width: 100%;
  min-width: 100%;
  min-height: 200px;
  max-width: 100%;
  max-height: 45%;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  border-left: none;
}

.layout-vertical .panel-resize-handle.vertical {
  width: 100%;
  height: 10px;
  cursor: ns-resize;
  writing-mode: horizontal-tb;
}

@media (--breakpoint-md-down) {
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
