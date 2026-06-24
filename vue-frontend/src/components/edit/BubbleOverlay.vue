<template>
  <div
    class="bubble-overlay"
    :class="{ 'brush-mode': isBrushMode }"
    :style="{ '--scale': scale || 1 }"
    ref="overlayRef"
    @mousedown="handleOverlayMouseDown"
  >
    <template v-for="(bubble, index) in bubbles" :key="index">
      <div
        class="bubble-highlight-box"
        :class="{
          selected: index === selectedIndex,
          'multi-selected': selectedIndices.length > 1 && selectedIndices.includes(index) && index !== selectedIndex
        }"
        :style="getBubbleStyle(bubble, index)"
        :data-index="index"
        :data-coords="JSON.stringify(bubble.coords)"
        :data-rotation="bubble.rotationAngle || 0"
        @click.stop="handleClick(index, $event)"
        @mousedown.stop="handleBubbleMouseDown(index, $event)"
      >
        <span class="bubble-index">{{ index + 1 }}</span>
        <template v-if="index === selectedIndex">
          <div
            class="resize-handle nw"
            data-handle="nw"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('nw', index, $event)"
          ></div>
          <div
            class="resize-handle n"
            data-handle="n"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('n', index, $event)"
          ></div>
          <div
            class="resize-handle ne"
            data-handle="ne"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('ne', index, $event)"
          ></div>
          <div
            class="resize-handle e"
            data-handle="e"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('e', index, $event)"
          ></div>
          <div
            class="resize-handle se"
            data-handle="se"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('se', index, $event)"
          ></div>
          <div
            class="resize-handle s"
            data-handle="s"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('s', index, $event)"
          ></div>
          <div
            class="resize-handle sw"
            data-handle="sw"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('sw', index, $event)"
          ></div>
          <div
            class="resize-handle w"
            data-handle="w"
            :data-parent-index="index"
            @mousedown.stop="handleResizeStart('w', index, $event)"
          ></div>
          <div class="rotate-line"></div>
          <div
            class="rotate-handle"
            title="拖拽旋转"
            :data-parent-index="index"
            @mousedown.stop="handleRotateStart(index, $event)"
          ></div>
        </template>
      </div>
    </template>
    <div
      v-if="drawingRect"
      class="drawing-rect"
      :style="getDrawingRectStyle()"
    ></div>
  </div>
</template>
<script setup lang="ts">

import { ref, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import { useBubbleStore } from '@/stores/bubbleStore'
import { calculateResizedCoords, type ResizeHandle } from '@/utils/bubbleResize'
import { buildBubbleOverlayStyle, buildDrawingRectStyle } from './bubbleOverlayGeometry'
const bubbleStore = useBubbleStore()
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
  rotateCurrentAngle
} = storeToRefs(bubbleStore)
const props = defineProps<{
  bubbles: BubbleState[]
  selectedIndex: number
  selectedIndices: number[]
  scale: number
  isDrawingMode: boolean
  isBrushMode?: boolean
  imageWidth?: number
  imageHeight?: number
}>()
const emit = defineEmits<{
  (e: 'select', index: number): void
  (e: 'multiSelect', index: number): void
  (e: 'dragStart', index: number, event: MouseEvent): void
  (e: 'dragEnd', index: number, newCoords: BubbleCoords): void
  (e: 'resizeStart', index: number, handle: string, event: MouseEvent): void
  (e: 'resizeEnd', index: number, newCoords: BubbleCoords): void
  (e: 'rotateStart', index: number, event: MouseEvent): void
  (e: 'rotateEnd', index: number, angle: number): void
  (e: 'drawBubble', coords: BubbleCoords): void
}>()
// 状态定义（本地状态，拖动状态从store共享）
const overlayRef = ref<HTMLElement | null>(null)
// 拖拽辅助状态（本地）
const dragStartX = ref(0)
const dragStartY = ref(0)
// 调整大小辅助状态（本地）
const resizeHandle = ref<ResizeHandle | ''>('')
const resizeStartX = ref(0)
const resizeStartY = ref(0)
const resizeInitialCoords = ref<BubbleCoords | null>(null)
// 旋转辅助状态（本地）
const rotateStartAngle = ref(0)
const rotateInitialAngle = ref(0)
const rotateCenterX = ref(0)
const rotateCenterY = ref(0)
const isDrawing = ref(false)
const drawStartX = ref(0)
const drawStartY = ref(0)
const drawingRect = ref<BubbleCoords | null>(null)
const isMiddleButtonDown = ref(false)
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
function getDrawingRectStyle(): Record<string, string> {
  return buildDrawingRectStyle(drawingRect.value)
}
function getMousePositionInImage(event: MouseEvent): { x: number; y: number } | null {
  if (!overlayRef.value) return null
  const rect = overlayRef.value.getBoundingClientRect()
  const scale = props.scale || 1
  // 计算鼠标相对于 overlay 的位置，然后转换为图片原生坐标
  const x = (event.clientX - rect.left) / scale
  const y = (event.clientY - rect.top) / scale
  return { x, y }
}
function handleClick(index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  // Shift+点击已在 mousedown 中处理，这里跳过
  if (event.shiftKey) {
    return
  }
  // 普通点击：单选
  emit('select', index)
}
function handleOverlayMouseDown(event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  if (event.button === 1) {
    event.preventDefault()
    isMiddleButtonDown.value = true
    document.body.classList.add('middle-button-drawing')
    startDrawing(event)
    return
  }
  if (event.button !== 0) return
  if (props.isDrawingMode) {
    startDrawing(event)
  }
}
function handleBubbleMouseDown(index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  if (event.button !== 0) return
  // 阻止默认行为（文本选择等）
  event.preventDefault()
  event.stopPropagation()
  // Shift+点击进行多选
  if (event.shiftKey) {
    emit('multiSelect', index)
    return
  }
  // 如果点击的不是当前选中的气泡，先选中它
  if (index !== props.selectedIndex) {
    emit('select', index)
    return
  }
  startDragging(index, event)
}
function startDragging(index: number, event: MouseEvent): void {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  isDragging.value = true
  draggingIndex.value = index
  dragStartX.value = event.clientX
  dragStartY.value = event.clientY
  dragOffsetX.value = 0
  dragOffsetY.value = 0
  const bubble = props.bubbles[index]
  if (bubble) {
    dragInitialX.value = bubble.coords[0]
    dragInitialY.value = bubble.coords[1]
  }
  emit('dragStart', index, event)
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}
function updateDragging(event: MouseEvent): void {
  const scale = props.scale || 1
  const deltaX = (event.clientX - dragStartX.value) / scale
  const deltaY = (event.clientY - dragStartY.value) / scale
  // 直接更新ref值，Vue会自动触发重新渲染
  dragOffsetX.value = deltaX
  dragOffsetY.value = deltaY
}
function finishDragging(event: MouseEvent): void {
  // 立即重置状态，防止重复触发
  const wasIndex = draggingIndex.value
  isDragging.value = false
  draggingIndex.value = -1
  dragOffsetX.value = 0
  dragOffsetY.value = 0
  const scale = props.scale || 1
  const deltaX = (event.clientX - dragStartX.value) / scale
  const deltaY = (event.clientY - dragStartY.value) / scale
  const bubble = props.bubbles[wasIndex]
  if (!bubble) return
  const [x1, y1, x2, y2] = bubble.coords
  const width = x2 - x1
  const height = y2 - y1
  let newX1 = Math.round(dragInitialX.value + deltaX)
  let newY1 = Math.round(dragInitialY.value + deltaY)
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000
  const safeWidth = Math.min(width, imgWidth)
  const safeHeight = Math.min(height, imgHeight)
  newX1 = Math.max(0, Math.min(newX1, imgWidth - safeWidth))
  newY1 = Math.max(0, Math.min(newY1, imgHeight - safeHeight))
  const newCoords: BubbleCoords = [newX1, newY1, newX1 + safeWidth, newY1 + safeHeight]
  emit('dragEnd', wasIndex, newCoords)
}
function handleResizeStart(handle: string, index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  if (event.button !== 0) return
  event.preventDefault()
  event.stopPropagation()
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  isResizing.value = true
  resizingIndex.value = index
  resizeHandle.value = handle as ResizeHandle
  resizeStartX.value = event.clientX
  resizeStartY.value = event.clientY
  const bubble = props.bubbles[index]
  if (bubble) {
    resizeInitialCoords.value = [...bubble.coords] as BubbleCoords
    resizeCurrentCoords.value = [...bubble.coords] as BubbleCoords
  }
  emit('resizeStart', index, handle, event)
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}
function updateResizing(event: MouseEvent): void {
  if (!resizeInitialCoords.value || !resizeHandle.value) return
  const scale = props.scale || 1
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
  // 直接更新ref值，Vue会自动触发重新渲染
  resizeCurrentCoords.value = nextCoords
}
function resetResizingState(): void {
  isResizing.value = false
  resizingIndex.value = -1
  resizeInitialCoords.value = null
  resizeCurrentCoords.value = null
  resizeHandle.value = ''
}
function finishResizing(event: MouseEvent): void {
  if (!resizeInitialCoords.value || !resizeHandle.value) return
  const scale = props.scale || 1
  const deltaX = (event.clientX - resizeStartX.value) / scale
  const deltaY = (event.clientY - resizeStartY.value) / scale
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000
  const bubble = props.bubbles[resizingIndex.value]
  const nextCoords = calculateResizedCoords(
    resizeInitialCoords.value,
    resizeHandle.value,
    deltaX,
    deltaY,
    {
      rotationAngle: bubble?.rotationAngle || 0,
      minSize: 10,
      imageWidth: imgWidth,
      imageHeight: imgHeight,
      clampToImage: true,
      round: true,
    }
  )
  if (!nextCoords) {
    resetResizingState()
    return
  }
  emit('resizeEnd', resizingIndex.value, nextCoords)
  resetResizingState()
}
function handleRotateStart(index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  if (event.button !== 0) return
  event.preventDefault()
  event.stopPropagation()
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  isRotating.value = true
  rotatingIndex.value = index
  // 获取气泡框的中心点（相对于视口）
  const bubble = props.bubbles[index]
  if (!bubble || !overlayRef.value) return
  const [x1, y1, x2, y2] = bubble.coords
  const scale = props.scale || 1
  const overlayRect = overlayRef.value.getBoundingClientRect()
  rotateCenterX.value = overlayRect.left + ((x1 + x2) / 2) * scale
  rotateCenterY.value = overlayRect.top + ((y1 + y2) / 2) * scale
  const dx = event.clientX - rotateCenterX.value
  const dy = event.clientY - rotateCenterY.value
  rotateStartAngle.value = Math.atan2(dy, dx) * 180 / Math.PI
  rotateInitialAngle.value = bubble.rotationAngle || 0
  rotateCurrentAngle.value = bubble.rotationAngle || 0
  emit('rotateStart', index, event)
  document.body.classList.add('rotating-box')
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
  // 按住 Shift 键时吸附到 15° 的倍数
  if (event.shiftKey) {
    newAngle = Math.round(newAngle / 15) * 15
  }
  // 直接更新ref值，Vue会自动触发重新渲染
  rotateCurrentAngle.value = newAngle
}
function finishRotating(_event: MouseEvent): void {
  document.body.classList.remove('rotating-box')
  const index = rotatingIndex.value
  const finalAngle = rotateCurrentAngle.value
  emit('rotateEnd', index, finalAngle)
  isRotating.value = false
  rotatingIndex.value = -1
}
function startDrawing(event: MouseEvent): void {
  const pos = getMousePositionInImage(event)
  if (!pos) return
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000
  if (pos.x < 0 || pos.x > imgWidth || pos.y < 0 || pos.y > imgHeight) return
  isDrawing.value = true
  drawStartX.value = pos.x
  drawStartY.value = pos.y
  drawingRect.value = [pos.x, pos.y, pos.x, pos.y]
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}
function updateDrawing(event: MouseEvent): void {
  const pos = getMousePositionInImage(event)
  if (!pos || !drawingRect.value) return
  drawingRect.value = [drawStartX.value, drawStartY.value, pos.x, pos.y]
}
function finishDrawing(event: MouseEvent): void {
  const pos = getMousePositionInImage(event)
  if (isMiddleButtonDown.value) {
    isMiddleButtonDown.value = false
    document.body.classList.remove('middle-button-drawing')
  }
  if (!pos || !drawingRect.value) {
    isDrawing.value = false
    drawingRect.value = null
    return
  }
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000
  const x1 = Math.max(0, Math.round(Math.min(drawStartX.value, pos.x)))
  const y1 = Math.max(0, Math.round(Math.min(drawStartY.value, pos.y)))
  const x2 = Math.min(imgWidth, Math.round(Math.max(drawStartX.value, pos.x)))
  const y2 = Math.min(imgHeight, Math.round(Math.max(drawStartY.value, pos.y)))
  const minSize = 10
  if (x2 - x1 < minSize || y2 - y1 < minSize) {
    isDrawing.value = false
    drawingRect.value = null
    return
  }
  emit('drawBubble', [x1, y1, x2, y2])
  isDrawing.value = false
  drawingRect.value = null
}
function handleMouseMove(event: MouseEvent): void {
  if (isDragging.value) {
    updateDragging(event)
  } else if (isResizing.value) {
    updateResizing(event)
  } else if (isRotating.value) {
    updateRotating(event)
  } else if (isDrawing.value) {
    updateDrawing(event)
  }
}
function handleMouseUp(event: MouseEvent): void {
  // 立即解绑全局事件，防止重复触发
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  if (event.button === 1 || isMiddleButtonDown.value) {
    isMiddleButtonDown.value = false
    document.body.classList.remove('middle-button-drawing')
  }
  if (isDragging.value) {
    finishDragging(event)
  } else if (isResizing.value) {
    finishResizing(event)
  } else if (isRotating.value) {
    finishRotating(event)
  } else if (isDrawing.value) {
    finishDrawing(event)
  }
}
onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  document.body.classList.remove('middle-button-drawing')
  document.body.classList.remove('rotating-box')
})
</script>

<style scoped>
/*
 * 【屏幕像素适配】使用 CSS 变量 --scale 实现反向缩放
 * 这样边框、手柄等 UI 元素在屏幕上保持固定大小，不随图片缩放而变化
 * 解决高分辨率图片缩小显示时手柄过小难以操作的问题
 */
.bubble-overlay {
  --bubble-overlay-border-default: rgba(255, 200, 0, .8);
  --bubble-overlay-border-strong: #ff6b6b;
  --bubble-overlay-border-muted: #0f8;
  --bubble-overlay-border-subtle: #ff1744;
  --bubble-overlay-border-hover: #fff;
  --bubble-overlay-shadow-default: rgba(0, 255, 136, .5);
  --bubble-overlay-shadow-raised: rgba(255, 23, 68, .6);
  --bubble-overlay-shadow-floating: rgba(0, 0, 0, .3);
  --bubble-overlay-shadow-strong: rgba(0, 255, 136, .8);
  --bubble-overlay-shadow-soft: #0f8;
  --bubble-overlay-surface-base: rgba(255, 200, 0, .1);
  --bubble-overlay-surface-raised: rgba(255, 107, 107, .2);
  --bubble-overlay-surface-muted: rgba(0, 255, 136, .15);
  --bubble-overlay-surface-subtle: rgba(255, 23, 68, .25);
  --bubble-overlay-surface-hover: rgba(0, 0, 0, .7);
  --bubble-overlay-surface-active: rgba(0, 255, 136, .9);
  --bubble-overlay-surface-selected: #0f8;
  --bubble-overlay-surface-overlay: #00cc6a;
  --bubble-overlay-surface-inverse: rgba(0, 255, 136, .6);
  --bubble-overlay-surface-contrast: rgba(0, 255, 136, .1);
  --bubble-overlay-text-primary: #1a1a2e;

  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  user-select: none;
  /* GPU compositing keeps large overlays stable while panning and zooming. */
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: contents;
}
/* 矩形气泡高亮框 - 使用反向缩放保持边框在屏幕上的固定宽度 */
.bubble-highlight-box {
  position: absolute;
  /* 边框宽度反向缩放：屏幕上始终显示为 2px */
  border: calc(2px / var(--scale, 1)) solid var(--bubble-overlay-border-default);
  background: var(--bubble-overlay-surface-base);
  cursor: pointer;
  pointer-events: auto;
  overflow: visible;
  /* Transform-heavy bubble boxes are isolated from surrounding layout. */
  will-change: transform, left, top, width, height;
  contain: layout style;
}

.bubble-highlight-box:hover {
  border-color: var(--bubble-overlay-border-strong);
  background: var(--bubble-overlay-surface-raised);
}

.bubble-highlight-box.selected {
  /* 选中时边框稍粗：屏幕上始终显示为 3px */
  border: calc(3px / var(--scale, 1)) solid var(--bubble-overlay-border-muted);
  background: var(--bubble-overlay-surface-muted);
  box-shadow: 0 0 calc(15px / var(--scale, 1)) var(--bubble-overlay-shadow-default);
  z-index: var(--z-local-toolbar);
  cursor: grab;
}

.bubble-highlight-box.selected:active {
  cursor: grabbing;
}

.bubble-highlight-box.multi-selected {
  border: calc(3px / var(--scale, 1)) solid var(--bubble-overlay-border-subtle);
  background: var(--bubble-overlay-surface-subtle);
  box-shadow: 0 0 calc(12px / var(--scale, 1)) var(--bubble-overlay-shadow-raised);
}
/* 气泡索引标签 - 反向缩放保持屏幕上固定大小 */
.bubble-index {
  position: absolute;
  /* 位置也需要反向缩放 */
  top: calc(-20px / var(--scale, 1));
  left: 0;
  background: var(--bubble-overlay-surface-hover);
  color: var(--color-text-inverse);
  font-size: calc(11px / var(--scale, 1));
  padding: calc(2px / var(--scale, 1)) calc(6px / var(--scale, 1));
  border-radius: calc(3px / var(--scale, 1));
  pointer-events: none;
  /* 使用 transform-origin 确保从左上角缩放 */
  transform-origin: left top;
}

.bubble-highlight-box.selected .bubble-index {
  background: var(--bubble-overlay-surface-active);
  color: var(--bubble-overlay-text-primary);
}
/* 调整手柄 - 反向缩放保持屏幕上 10x10px */
.resize-handle {
  display: block;
  position: absolute;
  /* 尺寸反向缩放 */
  width: calc(10px / var(--scale, 1));
  height: calc(10px / var(--scale, 1));
  background: var(--bubble-overlay-surface-selected);
  border: calc(2px / var(--scale, 1)) solid var(--bubble-overlay-border-hover);
  border-radius: calc(3px / var(--scale, 1));
  pointer-events: auto;
  z-index: var(--z-local-panel);
  box-shadow: 0 0 calc(3px / var(--scale, 1)) var(--bubble-overlay-shadow-floating);
}

.resize-handle:hover {
  background: var(--bubble-overlay-surface-overlay);
  /* 悬停时放大效果仍然有效 */
  transform: scale(1.2);
}
/* 手柄位置 - 偏移量也需要反向缩放（手柄10px，偏移5px使其居中对齐边框） */
.resize-handle.nw { top: calc(-5px / var(--scale, 1)); left: calc(-5px / var(--scale, 1)); cursor: nwse-resize; }
.resize-handle.n { top: calc(-5px / var(--scale, 1)); left: 50%; margin-left: calc(-5px / var(--scale, 1)); cursor: ns-resize; }
.resize-handle.ne { top: calc(-5px / var(--scale, 1)); right: calc(-5px / var(--scale, 1)); cursor: nesw-resize; }
.resize-handle.e { top: 50%; right: calc(-5px / var(--scale, 1)); margin-top: calc(-5px / var(--scale, 1)); cursor: ew-resize; }
.resize-handle.se { bottom: calc(-5px / var(--scale, 1)); right: calc(-5px / var(--scale, 1)); cursor: nwse-resize; }
.resize-handle.s { bottom: calc(-5px / var(--scale, 1)); left: 50%; margin-left: calc(-5px / var(--scale, 1)); cursor: ns-resize; }
.resize-handle.sw { bottom: calc(-5px / var(--scale, 1)); left: calc(-5px / var(--scale, 1)); cursor: nesw-resize; }
.resize-handle.w { top: 50%; left: calc(-5px / var(--scale, 1)); margin-top: calc(-5px / var(--scale, 1)); cursor: ew-resize; }
/* 旋转连接线 - 反向缩放 */
.rotate-line {
  display: block;
  position: absolute;
  top: calc(-25px / var(--scale, 1));
  left: 50%;
  transform: translateX(-50%);
  width: calc(2px / var(--scale, 1));
  height: calc(20px / var(--scale, 1));
  background: var(--bubble-overlay-surface-inverse);
  pointer-events: none;
}
/* 旋转手柄 - 反向缩放保持屏幕上 12x12px */
.rotate-handle {
  display: block;
  position: absolute;
  top: calc(-35px / var(--scale, 1));
  left: 50%;
  transform: translateX(-50%);
  width: calc(12px / var(--scale, 1));
  height: calc(12px / var(--scale, 1));
  background: var(--bubble-overlay-surface-selected);
  border: calc(2px / var(--scale, 1)) solid var(--bubble-overlay-border-hover);
  border-radius: 50%;
  cursor: grab;
  pointer-events: auto;
  z-index: var(--z-local-overlay);
  box-shadow: 0 0 calc(6px / var(--scale, 1)) var(--bubble-overlay-shadow-strong);
  transition: transform 0.15s, box-shadow 0.15s;
}

.rotate-handle:hover {
  transform: translateX(-50%) scale(1.2);
  box-shadow: 0 0 calc(10px / var(--scale, 1)) var(--bubble-overlay-shadow-soft);
}

.rotate-handle:active {
  cursor: grabbing;
  background: var(--bubble-overlay-surface-overlay);
}
/* 笔刷模式下让事件穿透到下层 viewport。 */
.bubble-overlay.brush-mode .bubble-highlight-box,
.bubble-overlay.brush-mode .resize-handle,
.bubble-overlay.brush-mode .rotate-handle {
  pointer-events: none;
}
/* 绘制中的矩形 - 边框也反向缩放 */
.drawing-rect {
  position: absolute;
  border: calc(2px / var(--scale, 1)) dashed var(--bubble-overlay-border-muted);
  background: var(--bubble-overlay-surface-contrast);
  pointer-events: none;
}
</style>
