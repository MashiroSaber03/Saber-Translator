<template>
  <div
    class="bubble-overlay"
    :class="{
      'bubble-overlay--brush-mode': isBrushMode,
      'bubble-overlay--disabled': disabled,
    }"
    :style="{ '--scale': interactionScale() }"
    ref="overlayRef"
  >
    <template
      v-for="(bubble, index) in bubbles"
      :key="bubble.backendBubbleId ?? bubble.clientMutationId ?? index"
    >
      <div
        class="bubble-overlay__highlight-box"
        :class="{
          'bubble-overlay__highlight-box--selected': index === selectedIndex,
          'bubble-overlay__highlight-box--multi-selected': selectedIndices.length > 1 && selectedIndexSet.has(index) && index !== selectedIndex
        }"
        :style="getBubbleStyle(bubble, index)"
        :data-index="index"
        role="button"
        :tabindex="disabled ? -1 : 0"
        :aria-label="`气泡 ${index + 1}`"
        :aria-pressed="index === selectedIndex || selectedIndexSet.has(index)"
        :aria-disabled="disabled || undefined"
        @mousedown.stop="handleBubbleMouseDown(index, $event)"
        @keydown.enter.stop.prevent="selectBubbleFromKeyboard(index)"
        @keydown.space.stop.prevent="selectBubbleFromKeyboard(index)"
      >
        <span class="bubble-overlay__index">{{ index + 1 }}</span>
        <template v-if="index === selectedIndex">
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--nw"
            @mousedown.stop="handleResizeStart('nw', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--n"
            @mousedown.stop="handleResizeStart('n', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--ne"
            @mousedown.stop="handleResizeStart('ne', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--e"
            @mousedown.stop="handleResizeStart('e', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--se"
            @mousedown.stop="handleResizeStart('se', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--s"
            @mousedown.stop="handleResizeStart('s', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--sw"
            @mousedown.stop="handleResizeStart('sw', index, $event)"
          ></div>
          <div
            class="bubble-overlay__resize-handle bubble-overlay__resize-handle--w"
            @mousedown.stop="handleResizeStart('w', index, $event)"
          ></div>
          <div class="bubble-overlay__rotate-line"></div>
          <div
            class="bubble-overlay__rotate-handle"
            title="拖拽旋转"
            @mousedown.stop="handleRotateStart(index, $event)"
          ></div>
        </template>
      </div>
    </template>
  </div>
</template>
<script setup lang="ts">
import { computed, ref, onUnmounted } from 'vue'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import { calculateDraggedCoords } from '@/utils/bubbleDrag'
import { calculateResizedCoords, type ResizeHandle } from '@/utils/bubbleResize'
import { buildBubbleOverlayStyle } from './bubbleOverlayGeometry'
import { useBubbleOverlayInteractionState } from './useBubbleOverlayInteractionState'
const {
  isDragging,
  draggingIndex,
  dragOffsetX,
  dragOffsetY,
  dragInitialX,
  dragInitialY,
  isResizing,
  resizingIndex,
  resizeCurrentCoords,
  isRotating,
  rotatingIndex,
  rotateCurrentAngle,
  dragStartX,
  dragStartY,
  resizeHandle,
  resizeStartX,
  resizeStartY,
  resizeInitialCoords,
  rotateStartAngle,
  rotateInitialAngle,
  rotateCenterX,
  rotateCenterY,
  resetDragging,
  resetResizing,
  resetRotating
} = useBubbleOverlayInteractionState()
const props = defineProps<{
  bubbles: BubbleState[]
  selectedIndex: number
  selectedIndices: number[]
  scale: number
  disabled?: boolean
  isBrushMode?: boolean
  imageWidth: number
  imageHeight: number
}>()
const emit = defineEmits<{
  (e: 'select', index: number): void
  (e: 'multiSelect', index: number): void
  (e: 'dragEnd', index: number, newCoords: BubbleCoords): void
  (e: 'resizeEnd', index: number, newCoords: BubbleCoords): void
  (e: 'rotateEnd', index: number, angle: number): void
}>()
const overlayRef = ref<HTMLElement | null>(null)
const selectedIndexSet = computed(() => new Set(props.selectedIndices))
function getBubbleStyle(bubble: BubbleState, index: number): Record<string, string> {
  return buildBubbleOverlayStyle({
    bubble,
    index,
    isDragging: isDragging.value,
    draggingIndex: draggingIndex.value,
    dragInitialX: dragInitialX.value,
    dragInitialY: dragInitialY.value,
    dragOffsetX: dragOffsetX.value,
    dragOffsetY: dragOffsetY.value,
    isResizing: isResizing.value,
    resizingIndex: resizingIndex.value,
    resizeCurrentCoords: resizeCurrentCoords.value,
    isRotating: isRotating.value,
    rotatingIndex: rotatingIndex.value,
    rotateCurrentAngle: rotateCurrentAngle.value
  })
}
function interactionScale(): number {
  return Number.isFinite(props.scale) && props.scale > 0 ? props.scale : 1
}

function selectBubbleFromKeyboard(index: number): void {
  if (props.disabled || props.isBrushMode) return
  emit('select', index)
}

function hasImageBounds(): boolean {
  return Number.isFinite(props.imageWidth)
    && Number.isFinite(props.imageHeight)
    && props.imageWidth > 0
    && props.imageHeight > 0
}

function handleBubbleMouseDown(index: number, event: MouseEvent): void {
  if (props.disabled || props.isBrushMode) return
  if (event.button !== 0) return
  event.preventDefault()
  event.stopPropagation()
  if (event.shiftKey) {
    emit('multiSelect', index)
    return
  }
  if (index !== props.selectedIndex) {
    emit('select', index)
    return
  }
  startDragging(index, event)
}
function startDragging(index: number, event: MouseEvent): void {
  const bubble = props.bubbles[index]
  if (!bubble || !hasImageBounds()) return
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  isDragging.value = true
  draggingIndex.value = index
  dragStartX.value = event.clientX
  dragStartY.value = event.clientY
  dragOffsetX.value = 0
  dragOffsetY.value = 0
  dragInitialX.value = bubble.coords[0]
  dragInitialY.value = bubble.coords[1]
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}
function updateDragging(event: MouseEvent): void {
  const scale = interactionScale()
  const deltaX = (event.clientX - dragStartX.value) / scale
  const deltaY = (event.clientY - dragStartY.value) / scale
  dragOffsetX.value = deltaX
  dragOffsetY.value = deltaY
}
function finishDragging(event: MouseEvent): void {
  const wasIndex = draggingIndex.value
  const movement = Math.hypot(
    event.clientX - dragStartX.value,
    event.clientY - dragStartY.value,
  )
  resetDragging()
  if (!hasImageBounds() || movement < 2) return
  const scale = interactionScale()
  const deltaX = (event.clientX - dragStartX.value) / scale
  const deltaY = (event.clientY - dragStartY.value) / scale
  const bubble = props.bubbles[wasIndex]
  if (!bubble) return
  const [x1, y1, x2, y2] = bubble.coords
  const newCoords = calculateDraggedCoords(
    [dragInitialX.value, dragInitialY.value, dragInitialX.value + (x2 - x1), dragInitialY.value + (y2 - y1)],
    deltaX,
    deltaY,
    props.imageWidth,
    props.imageHeight
  )
  emit('dragEnd', wasIndex, newCoords)
}
function handleResizeStart(handle: ResizeHandle, index: number, event: MouseEvent): void {
  if (props.disabled || props.isBrushMode) return
  if (event.button !== 0) return
  const bubble = props.bubbles[index]
  if (!bubble || !hasImageBounds()) return
  event.preventDefault()
  event.stopPropagation()
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  isResizing.value = true
  resizingIndex.value = index
  resizeHandle.value = handle
  resizeStartX.value = event.clientX
  resizeStartY.value = event.clientY
  resizeInitialCoords.value = [...bubble.coords] as BubbleCoords
  resizeCurrentCoords.value = [...bubble.coords] as BubbleCoords
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}
function updateResizing(event: MouseEvent): void {
  if (!resizeInitialCoords.value || !resizeHandle.value) return
  const scale = interactionScale()
  const deltaX = (event.clientX - resizeStartX.value) / scale
  const deltaY = (event.clientY - resizeStartY.value) / scale
  const bubble = props.bubbles[resizingIndex.value]
  const nextCoords = calculateResizedCoords(
    resizeInitialCoords.value,
    resizeHandle.value,
    deltaX,
    deltaY,
    {
      rotationAngle: bubble?.rotationAngle || 0,
      minSize: 10,
    }
  )
  if (!nextCoords) return
  resizeCurrentCoords.value = nextCoords
}
function finishResizing(event: MouseEvent): void {
  const initialCoords = resizeInitialCoords.value
  const handle = resizeHandle.value
  const index = resizingIndex.value
  if (!initialCoords || !handle || !hasImageBounds()) {
    resetResizing()
    return
  }
  if (Math.hypot(
    event.clientX - resizeStartX.value,
    event.clientY - resizeStartY.value,
  ) < 2) {
    resetResizing()
    return
  }
  const scale = interactionScale()
  const deltaX = (event.clientX - resizeStartX.value) / scale
  const deltaY = (event.clientY - resizeStartY.value) / scale
  const bubble = props.bubbles[index]
  const nextCoords = calculateResizedCoords(
    initialCoords,
    handle,
    deltaX,
    deltaY,
    {
      rotationAngle: bubble?.rotationAngle || 0,
      minSize: 10,
      imageWidth: props.imageWidth,
      imageHeight: props.imageHeight,
      clampToImage: true,
      round: true,
    }
  )
  if (!nextCoords) {
    resetResizing()
    return
  }
  emit('resizeEnd', index, nextCoords)
  resetResizing()
}
function handleRotateStart(index: number, event: MouseEvent): void {
  if (props.disabled || props.isBrushMode) return
  if (event.button !== 0) return
  const bubble = props.bubbles[index]
  if (!bubble || !overlayRef.value) return
  event.preventDefault()
  event.stopPropagation()
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  isRotating.value = true
  rotatingIndex.value = index
  const [x1, y1, x2, y2] = bubble.coords
  const scale = interactionScale()
  const overlayRect = overlayRef.value.getBoundingClientRect()
  rotateCenterX.value = overlayRect.left + ((x1 + x2) / 2) * scale
  rotateCenterY.value = overlayRect.top + ((y1 + y2) / 2) * scale
  const dx = event.clientX - rotateCenterX.value
  const dy = event.clientY - rotateCenterY.value
  rotateStartAngle.value = Math.atan2(dy, dx) * 180 / Math.PI
  rotateInitialAngle.value = bubble.rotationAngle
  rotateCurrentAngle.value = bubble.rotationAngle
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}
function updateRotating(event: MouseEvent): void {
  const dx = event.clientX - rotateCenterX.value
  const dy = event.clientY - rotateCenterY.value
  const currentAngle = Math.atan2(dy, dx) * 180 / Math.PI
  const deltaAngle = currentAngle - rotateStartAngle.value
  let newAngle = rotateInitialAngle.value + deltaAngle
  while (newAngle > 180) newAngle -= 360
  while (newAngle < -180) newAngle += 360
  if (event.shiftKey) {
    newAngle = Math.round(newAngle / 15) * 15
  }
  rotateCurrentAngle.value = newAngle
}
function finishRotating(): void {
  const index = rotatingIndex.value
  const finalAngle = rotateCurrentAngle.value
  const changed = Math.abs(finalAngle - rotateInitialAngle.value) >= 0.01
  resetRotating()
  if (changed && index >= 0 && props.bubbles[index]) {
    emit('rotateEnd', index, finalAngle)
  }
}
function handleMouseMove(event: MouseEvent): void {
  if (isDragging.value) {
    updateDragging(event)
  } else if (isResizing.value) {
    updateResizing(event)
  } else if (isRotating.value) {
    updateRotating(event)
  }
}
function handleMouseUp(event: MouseEvent): void {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  if (isDragging.value) {
    finishDragging(event)
  } else if (isResizing.value) {
    finishResizing(event)
  } else if (isRotating.value) {
    finishRotating()
  }
}
onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})
</script>

<style scoped>
.bubble-overlay {
  --bubble-overlay-box-border: color-mix(in srgb, var(--color-status-warning-bright) 80%, transparent);
  --bubble-overlay-box-background: color-mix(in srgb, var(--color-status-warning-bright) 10%, transparent);
  --bubble-overlay-box-hover-border: var(--color-status-error-bright);
  --bubble-overlay-box-hover-background: color-mix(in srgb, var(--color-status-error-bright) 20%, transparent);
  --bubble-overlay-selection-border: var(--color-action-success-bright);
  --bubble-overlay-selection-background: color-mix(in srgb, var(--color-action-success-bright) 15%, transparent);
  --bubble-overlay-selection-glow: color-mix(in srgb, var(--color-action-success-bright) 50%, transparent);
  --bubble-overlay-multi-selection-border: var(--color-status-error-vivid);
  --bubble-overlay-multi-selection-background: color-mix(in srgb, var(--color-status-error-vivid) 25%, transparent);
  --bubble-overlay-multi-selection-glow: color-mix(in srgb, var(--color-status-error-vivid) 60%, transparent);
  --bubble-overlay-index-background: var(--color-overlay-backdrop-strong);
  --bubble-overlay-selected-index-background: color-mix(in srgb, var(--color-action-success-bright) 90%, transparent);
  --bubble-overlay-selected-index-text: var(--color-surface-inverse);
  --bubble-overlay-handle-background: var(--color-action-success-bright);
  --bubble-overlay-handle-active-background: var(--color-action-success-bright-active);
  --bubble-overlay-handle-border: var(--color-text-inverse);
  --bubble-overlay-handle-shadow: var(--color-overlay-scrim-subtle);
  --bubble-overlay-rotate-line-background: color-mix(in srgb, var(--color-action-success-bright) 60%, transparent);
  --bubble-overlay-rotate-handle-glow: color-mix(in srgb, var(--color-action-success-bright) 80%, transparent);
  --bubble-overlay-rotate-handle-hover-glow: var(--color-action-success-bright);

  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  user-select: none;
  /* Inverse scaling keeps overlay controls usable while the image zoom changes. */
  /* GPU compositing keeps large overlays stable while panning and zooming. */
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: contents;
}

.bubble-overlay__highlight-box {
  position: absolute;
  border: calc(2px / var(--scale, 1)) solid var(--bubble-overlay-box-border);
  background: var(--bubble-overlay-box-background);
  cursor: pointer;
  pointer-events: auto;
  overflow: visible;
  /* Transform-heavy bubble boxes are isolated from surrounding layout. */
  will-change: transform, left, top, width, height;
  contain: layout style;
}

.bubble-overlay__highlight-box:hover {
  border-color: var(--bubble-overlay-box-hover-border);
  background: var(--bubble-overlay-box-hover-background);
}

.bubble-overlay__highlight-box--selected {
  border: calc(3px / var(--scale, 1)) solid var(--bubble-overlay-selection-border);
  background: var(--bubble-overlay-selection-background);
  box-shadow: 0 0 calc(15px / var(--scale, 1)) var(--bubble-overlay-selection-glow);
  z-index: var(--z-local-toolbar);
  cursor: grab;
}

.bubble-overlay__highlight-box--selected:active {
  cursor: grabbing;
}

.bubble-overlay__highlight-box--multi-selected {
  border: calc(3px / var(--scale, 1)) solid var(--bubble-overlay-multi-selection-border);
  background: var(--bubble-overlay-multi-selection-background);
  box-shadow: 0 0 calc(12px / var(--scale, 1)) var(--bubble-overlay-multi-selection-glow);
}

.bubble-overlay__index {
  position: absolute;
  top: calc(-20px / var(--scale, 1));
  left: 0;
  background: var(--bubble-overlay-index-background);
  color: var(--color-text-inverse);
  font-size: calc(11px / var(--scale, 1));
  padding: calc(2px / var(--scale, 1)) calc(6px / var(--scale, 1));
  border-radius: calc(3px / var(--scale, 1));
  pointer-events: none;
  transform-origin: left top;
}

.bubble-overlay__highlight-box--selected .bubble-overlay__index {
  background: var(--bubble-overlay-selected-index-background);
  color: var(--bubble-overlay-selected-index-text);
}

.bubble-overlay__resize-handle {
  display: block;
  position: absolute;
  width: calc(10px / var(--scale, 1));
  height: calc(10px / var(--scale, 1));
  background: var(--bubble-overlay-handle-background);
  border: calc(2px / var(--scale, 1)) solid var(--bubble-overlay-handle-border);
  border-radius: calc(3px / var(--scale, 1));
  pointer-events: auto;
  z-index: var(--z-local-panel);
  box-shadow: 0 0 calc(3px / var(--scale, 1)) var(--bubble-overlay-handle-shadow);
}

.bubble-overlay__resize-handle:hover {
  background: var(--bubble-overlay-handle-active-background);
  transform: scale(1.2);
}

.bubble-overlay__resize-handle--nw { top: calc(-5px / var(--scale, 1)); left: calc(-5px / var(--scale, 1)); cursor: nwse-resize; }
.bubble-overlay__resize-handle--n { top: calc(-5px / var(--scale, 1)); left: 50%; margin-left: calc(-5px / var(--scale, 1)); cursor: ns-resize; }
.bubble-overlay__resize-handle--ne { top: calc(-5px / var(--scale, 1)); right: calc(-5px / var(--scale, 1)); cursor: nesw-resize; }
.bubble-overlay__resize-handle--e { top: 50%; right: calc(-5px / var(--scale, 1)); margin-top: calc(-5px / var(--scale, 1)); cursor: ew-resize; }
.bubble-overlay__resize-handle--se { bottom: calc(-5px / var(--scale, 1)); right: calc(-5px / var(--scale, 1)); cursor: nwse-resize; }
.bubble-overlay__resize-handle--s { bottom: calc(-5px / var(--scale, 1)); left: 50%; margin-left: calc(-5px / var(--scale, 1)); cursor: ns-resize; }
.bubble-overlay__resize-handle--sw { bottom: calc(-5px / var(--scale, 1)); left: calc(-5px / var(--scale, 1)); cursor: nesw-resize; }
.bubble-overlay__resize-handle--w { top: 50%; left: calc(-5px / var(--scale, 1)); margin-top: calc(-5px / var(--scale, 1)); cursor: ew-resize; }

.bubble-overlay__rotate-line {
  display: block;
  position: absolute;
  top: calc(-25px / var(--scale, 1));
  left: 50%;
  transform: translateX(-50%);
  width: calc(2px / var(--scale, 1));
  height: calc(20px / var(--scale, 1));
  background: var(--bubble-overlay-rotate-line-background);
  pointer-events: none;
}

.bubble-overlay__rotate-handle {
  display: block;
  position: absolute;
  top: calc(-35px / var(--scale, 1));
  left: 50%;
  transform: translateX(-50%);
  width: calc(12px / var(--scale, 1));
  height: calc(12px / var(--scale, 1));
  background: var(--bubble-overlay-handle-background);
  border: calc(2px / var(--scale, 1)) solid var(--bubble-overlay-handle-border);
  border-radius: 50%;
  cursor: grab;
  pointer-events: auto;
  z-index: var(--z-local-overlay);
  box-shadow: 0 0 calc(6px / var(--scale, 1)) var(--bubble-overlay-rotate-handle-glow);
  transition: transform 0.15s, box-shadow 0.15s;
}

.bubble-overlay__rotate-handle:hover {
  transform: translateX(-50%) scale(1.2);
  box-shadow: 0 0 calc(10px / var(--scale, 1)) var(--bubble-overlay-rotate-handle-hover-glow);
}

.bubble-overlay__rotate-handle:active {
  cursor: grabbing;
  background: var(--bubble-overlay-handle-active-background);
}

/* 笔刷模式下让事件穿透到下层 viewport。 */
.bubble-overlay--brush-mode .bubble-overlay__highlight-box,
.bubble-overlay--brush-mode .bubble-overlay__resize-handle,
.bubble-overlay--brush-mode .bubble-overlay__rotate-handle,
.bubble-overlay--disabled .bubble-overlay__highlight-box,
.bubble-overlay--disabled .bubble-overlay__resize-handle,
.bubble-overlay--disabled .bubble-overlay__rotate-handle {
  pointer-events: none;
}

</style>
