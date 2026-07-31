<script setup lang="ts">
import { computed, ref } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { BubbleCoords, BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import BubbleEditor from './BubbleEditor.vue'
import BubbleOverlay from './BubbleOverlay.vue'

type ViewMode = 'dual' | 'original' | 'translated'
type LayoutMode = 'horizontal' | 'vertical'
type ViewportName = 'original' | 'translated'

const props = defineProps<{
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

const processedImageUrl = computed(
  () => props.currentImage?.translatedAssetUrl
    || props.currentImage?.cleanAssetUrl
    || props.currentImage?.sourceAssetUrl
    || '',
)

const processedImageLabel = computed(() =>
  props.currentImage?.translatedAssetUrl
    ? '翻译图 (中文)'
    : props.currentImage?.cleanAssetUrl
      ? '消字图'
      : '结果图',
)

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
  applyToAllStyle: [updates: Partial<BubbleState>]
  ocrRecognize: [index: number]
  reTranslate: [index: number]
  resetCurrent: [index: number]
}>()

const originalViewportRef = ref<HTMLElement | null>(null)
const originalWrapperRef = ref<HTMLElement | null>(null)
const originalImageRef = ref<HTMLImageElement | null>(null)
const originalPanelRef = ref<HTMLElement | null>(null)
const translatedViewportRef = ref<HTMLElement | null>(null)
const translatedWrapperRef = ref<HTMLElement | null>(null)
const translatedImageRef = ref<HTMLImageElement | null>(null)
const translatedPanelRef = ref<HTMLElement | null>(null)
const editPanelRef = ref<HTMLElement | null>(null)
const originalPanelCollapsed = ref(false)
const translatedPanelCollapsed = ref(false)

defineExpose({
  originalViewportRef,
  originalWrapperRef,
  originalImageRef,
  originalPanelRef,
  translatedViewportRef,
  translatedWrapperRef,
  translatedImageRef,
  translatedPanelRef,
  editPanelRef,
})
</script>

<template>
  <div
    class="edit-image-comparison"
    :class="[
      `edit-image-comparison--layout-${layoutMode}`,
      { 'edit-image-comparison--drawing': isDrawingMode },
      { 'edit-image-comparison--brush-active': !!brushMode },
    ]"
    :data-brush-mode="brushMode || undefined"
  >
    <div class="edit-image-comparison__canvas-region">
      <div
        ref="originalPanelRef"
        v-show="viewMode !== 'translated'"
        class="edit-image-comparison__image-panel edit-image-comparison__image-panel--original"
        :class="{ 'edit-image-comparison__image-panel--collapsed': viewMode === 'translated' || originalPanelCollapsed }"
      >
        <div class="edit-image-comparison__panel-header">
          <span class="edit-image-comparison__panel-title">
            <UiIcon name="book-open" size="14" />
            <span>原图 (日文)</span>
          </span>
          <UiIconButton
            class="edit-image-comparison__panel-toggle"
            :label="originalPanelCollapsed ? '展开原图面板' : '折叠原图面板'"
            title="折叠/展开"
            variant="inverse"
            size="xs"
            @click="originalPanelCollapsed = !originalPanelCollapsed"
          >
            <UiIcon :name="originalPanelCollapsed ? 'plus' : 'minus'" size="14" />
          </UiIconButton>
        </div>
        <div
          ref="originalViewportRef"
          class="edit-image-comparison__viewport"
          @wheel.prevent="emit('wheelPanel', $event, 'original')"
          @mousedown="emit('mouseDownPanel', $event, 'original')"
          @dblclick="emit('fitToScreen')"
        >
          <div
            ref="originalWrapperRef"
            class="edit-image-comparison__canvas-wrapper"
            :style="originalTransformStyle"
          >
            <img
              v-if="currentImage?.sourceAssetUrl"
              ref="originalImageRef"
              class="edit-image-comparison__image"
              :src="currentImage.sourceAssetUrl"
              alt="原图"
              @load="emit('imageLoad', 'original')"
            >
            <BubbleOverlay
              v-if="currentImage?.sourceAssetUrl"
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
              class="edit-image-comparison__drawing-rect"
              :style="drawingRectStyle"
            ></div>
          </div>
        </div>
      </div>

      <div
        v-if="viewMode === 'dual'"
        class="edit-image-comparison__divider"
        :class="{ 'edit-image-comparison__divider--vertical': layoutMode === 'vertical' }"
        @mousedown="emit('startDividerDrag', $event)"
      >
        <span class="edit-image-comparison__divider-handle">⋮</span>
      </div>

      <div
        ref="translatedPanelRef"
        v-show="viewMode !== 'original'"
        class="edit-image-comparison__image-panel edit-image-comparison__image-panel--translated"
        :class="{ 'edit-image-comparison__image-panel--collapsed': viewMode === 'original' || translatedPanelCollapsed }"
      >
        <div class="edit-image-comparison__panel-header">
          <span class="edit-image-comparison__panel-title">
            <UiIcon name="file-text" size="14" />
            <span>{{ processedImageLabel }}</span>
          </span>
          <UiIconButton
            class="edit-image-comparison__panel-toggle"
            :label="translatedPanelCollapsed ? '展开翻译图面板' : '折叠翻译图面板'"
            title="折叠/展开"
            variant="inverse"
            size="xs"
            @click="translatedPanelCollapsed = !translatedPanelCollapsed"
          >
            <UiIcon :name="translatedPanelCollapsed ? 'plus' : 'minus'" size="14" />
          </UiIconButton>
        </div>
        <div
          ref="translatedViewportRef"
          class="edit-image-comparison__viewport"
          @wheel.prevent="emit('wheelPanel', $event, 'translated')"
          @mousedown="emit('mouseDownPanel', $event, 'translated')"
          @dblclick="emit('fitToScreen')"
        >
          <div
            ref="translatedWrapperRef"
            class="edit-image-comparison__canvas-wrapper"
            :style="translatedTransformStyle"
          >
            <img
              v-if="processedImageUrl"
              ref="translatedImageRef"
              class="edit-image-comparison__image"
              :src="processedImageUrl"
              :alt="processedImageLabel"
              @load="emit('imageLoad', 'translated')"
            >
            <BubbleOverlay
              v-if="processedImageUrl"
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
              class="edit-image-comparison__drawing-rect edit-image-comparison__drawing-rect--translated"
              :style="drawingRectStyle"
            ></div>
          </div>
        </div>
      </div>
    </div>

    <div ref="editPanelRef" class="edit-image-comparison__editor-panel">
      <div
        class="edit-image-comparison__editor-resize-handle edit-image-comparison__editor-resize-handle--vertical"
        @mousedown="emit('startPanelResize', $event)"
      >
        ⋮⋮⋮
      </div>
      <BubbleEditor
        :bubble="selectedBubble ?? null"
        :bubble-index="selectedBubbleIndex"
        :is-ocr-loading="isOcrLoading"
        :is-translate-loading="isTranslateLoading"
        @update="emit('bubbleUpdate', $event)"
        @apply-to-all-style="emit('applyToAllStyle', $event)"
        @ocr-recognize="emit('ocrRecognize', $event)"
        @re-translate="emit('reTranslate', $event)"
        @reset-current="emit('resetCurrent', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
.edit-image-comparison {
  --edit-image-comparison-panel-background: var(--color-surface-inverse-panel);
  --edit-image-comparison-panel-header-background: var(--color-overlay-scrim-subtle);
  --edit-image-comparison-panel-divider-border: var(--color-overlay-inverse-subtle);
  --edit-image-comparison-original-title-text: var(--color-status-error-bright);
  --edit-image-comparison-translated-title-text: var(--color-action-success-bright);
  --edit-image-comparison-viewport-background: var(--color-surface-inverse-canvas);
  --edit-image-comparison-divider-background: var(--color-surface-inverse-depth);
  --edit-image-comparison-divider-handle-text: var(--color-text-muted);
  --edit-image-comparison-drawing-rect-border: var(--color-status-info-bright);
  --edit-image-comparison-drawing-rect-background: color-mix(in srgb, var(--color-status-info-bright) 10%, transparent);
  --edit-image-comparison-repair-mode-background: color-mix(in srgb, var(--color-status-success) 5%, transparent);
  --edit-image-comparison-restore-mode-background: color-mix(in srgb, var(--color-status-info) 5%, transparent);

  display: flex;
  flex: 1;
  flex-direction: row;
  gap: 0;
  min-height: 0;
  transition: flex-direction 0.3s ease;
}

.edit-image-comparison__canvas-region {
  display: flex;
  flex: 1 1 auto;
  gap: 0;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  padding: 8px;
}

.edit-image-comparison__image-panel {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 150px;
  overflow: hidden;
  border-radius: 8px;
  background: var(--edit-image-comparison-panel-background);
  transition: flex 0.3s ease;
}

.edit-image-comparison__image-panel--collapsed {
  flex: 0 0 40px;
  min-width: 40px;
}

.edit-image-comparison__panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--edit-image-comparison-panel-divider-border);
  background: var(--edit-image-comparison-panel-header-background);
}

.edit-image-comparison__panel-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--color-text-inverse);
  font-size: 13px;
  font-weight: 500;
}

.edit-image-comparison__image-panel--original .edit-image-comparison__panel-title {
  color: var(--edit-image-comparison-original-title-text);
}

.edit-image-comparison__image-panel--translated .edit-image-comparison__panel-title {
  color: var(--edit-image-comparison-translated-title-text);
}

.edit-image-comparison__panel-toggle {
  color: var(--color-text-inverse);
}

.edit-image-comparison__viewport {
  position: relative;
  flex: 1;
  overflow: hidden;
  background-color: var(--edit-image-comparison-viewport-background);
  backface-visibility: hidden;
  cursor: grab;
  transform: translateZ(0);
}

.edit-image-comparison__viewport:active {
  cursor: grabbing;
}

.edit-image-comparison__viewport:focus {
  outline: 2px solid var(--color-border-accent);
  outline-offset: -2px;
}

.edit-image-comparison__canvas-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  backface-visibility: hidden;
  perspective: 1000px;
  transform-origin: 0 0;
  will-change: transform;
}

.edit-image-comparison__image {
  display: block;
  max-width: none;
  backface-visibility: hidden;
  image-rendering: crisp-edges;
  pointer-events: none;
  transform: translateZ(0);
  user-select: none;
  -webkit-user-drag: none;
}

.edit-image-comparison__divider {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 8px;
  background: var(--edit-image-comparison-divider-background);
  cursor: col-resize;
  transition: background 0.2s;
}

.edit-image-comparison__divider:hover {
  background: var(--color-surface-accent);
}

.edit-image-comparison__divider-handle {
  color: var(--edit-image-comparison-divider-handle-text);
  font-size: 12px;
  writing-mode: vertical-lr;
  user-select: none;
}

.edit-image-comparison__divider:hover .edit-image-comparison__divider-handle {
  color: var(--color-text-inverse);
}

.edit-image-comparison__divider--vertical {
  width: 100%;
  height: 8px;
  cursor: ns-resize;
}

.edit-image-comparison__divider--vertical .edit-image-comparison__divider-handle {
  writing-mode: horizontal-tb;
}

.edit-image-comparison__editor-panel {
  display: flex;
  flex: 0 0 600px;
  flex-direction: row;
  min-width: 520px;
  min-height: 0;
  max-width: 65%;
  overflow: hidden;
  border-left: 1px solid var(--color-border-muted);
  background: var(--color-surface-card);
  transition: flex 0.3s ease, max-height 0.3s ease, border 0.3s ease;
}

.edit-image-comparison__editor-resize-handle {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: var(--color-surface-app);
  color: var(--color-text-muted);
  font-size: 10px;
  letter-spacing: 0;
  transition: background 0.2s;
}

.edit-image-comparison__editor-resize-handle--vertical {
  width: 10px;
  cursor: ew-resize;
  writing-mode: vertical-rl;
}

.edit-image-comparison__editor-resize-handle:hover {
  background: var(--color-surface-accent);
  color: var(--color-text-inverse);
}

.edit-image-comparison--drawing .edit-image-comparison__viewport,
.edit-image-comparison--drawing .edit-image-comparison__canvas-wrapper {
  cursor: crosshair;
}

.edit-image-comparison__drawing-rect {
  position: absolute;
  z-index: var(--z-local-popover);
  border: 2px dashed var(--edit-image-comparison-drawing-rect-border);
  background: var(--edit-image-comparison-drawing-rect-background);
  pointer-events: none;
}

.edit-image-comparison--brush-active .edit-image-comparison__viewport {
  cursor: none;
}

.edit-image-comparison--brush-active[data-brush-mode="repair"] .edit-image-comparison__viewport {
  background: var(--edit-image-comparison-repair-mode-background);
}

.edit-image-comparison--brush-active[data-brush-mode="restore"] .edit-image-comparison__viewport {
  background: var(--edit-image-comparison-restore-mode-background);
}

.edit-image-comparison--brush-active .edit-image-comparison__canvas-wrapper {
  pointer-events: auto;
}

.edit-image-comparison__image-panel--collapsed .edit-image-comparison__viewport {
  display: none;
}

.edit-image-comparison--layout-vertical {
  flex-direction: column;
}

.edit-image-comparison--layout-vertical .edit-image-comparison__canvas-region {
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.edit-image-comparison--layout-vertical .edit-image-comparison__image-panel {
  min-width: 0;
  min-height: 150px;
}

.edit-image-comparison--layout-vertical .edit-image-comparison__image-panel--collapsed {
  flex: 0 0 40px;
  min-width: 0;
  min-height: 40px;
}

.edit-image-comparison--layout-vertical .edit-image-comparison__editor-panel {
  flex: 0 0 auto;
  flex-direction: column;
  width: 100%;
  min-width: 100%;
  min-height: 200px;
  max-width: 100%;
  max-height: 45%;
  border-top: 1px solid var(--color-border-muted);
  border-left: none;
}

.edit-image-comparison--layout-vertical .edit-image-comparison__editor-resize-handle--vertical {
  width: 100%;
  height: 10px;
  cursor: ns-resize;
  writing-mode: horizontal-tb;
}

@media (--breakpoint-md-down) {
  .edit-image-comparison__canvas-region {
    flex-direction: column;
  }

  .edit-image-comparison__divider {
    width: 100%;
    height: 8px;
    cursor: ns-resize;
  }

  .edit-image-comparison__divider-handle {
    writing-mode: horizontal-tb;
  }
}
</style>
