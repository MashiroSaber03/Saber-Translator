<!--
  气泡覆盖层组件
  在图片上显示气泡高亮框，支持选择、多选、拖拽移动、大小调整、绘制新气泡
-->
<template>
  <div
    class="bubble-overlay"
    :class="{ 'brush-mode': isBrushMode }"
    :style="{ '--scale': scale || 1 }"
    ref="overlayRef"
    @mousedown="handleOverlayMouseDown"
  >
    <!-- 已有气泡高亮框 -->
    <template v-for="(bubble, index) in bubbles" :key="index">
      <!-- 
        编辑模式下统一使用矩形渲染（忽略polygon数据）
        这样所有气泡都有调整手柄和旋转手柄
        原版edit_mode.js也是使用coords矩形进行编辑
      -->
      <!-- 矩形气泡 -->
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
        <!-- 调整手柄（选中时显示） -->
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
          <!-- 旋转连接线 -->
          <div class="rotate-line"></div>
          <!-- 旋转手柄 -->
          <div
            class="rotate-handle"
            title="拖拽旋转"
            :data-parent-index="index"
            @mousedown.stop="handleRotateStart(index, $event)"
          ></div>
        </template>
      </div>
    </template>

    <!-- 绘制中的临时矩形 -->
    <div
      v-if="drawingRect"
      class="drawing-rect"
      :style="getDrawingRectStyle()"
    ></div>
  </div>
</template>


<script setup lang="ts">
/**
 * 气泡覆盖层组件
 * 显示气泡高亮框，支持选择、多选、拖拽、调整大小、旋转
 * 使用bubbleStore共享拖动状态，实现多视口同步
 */
import { ref, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import { useBubbleStore } from '@/stores/bubbleStore'

// ============================================================
// Store
// ============================================================

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

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 气泡数组 */
  bubbles: BubbleState[]
  /** 当前选中的气泡索引 */
  selectedIndex: number
  /** 多选的气泡索引数组 */
  selectedIndices: number[]
  /** 当前缩放比例 */
  scale: number
  /** 是否处于绘制模式 */
  isDrawingMode: boolean
  /** 【修复问题2】是否处于笔刷模式（笔刷模式下禁用气泡框交互） */
  isBrushMode?: boolean
  /** 图片宽度（原始尺寸） */
  imageWidth?: number
  /** 图片高度（原始尺寸） */
  imageHeight?: number
}>()

const emit = defineEmits<{
  /** 选择气泡 */
  (e: 'select', index: number): void
  /** 多选气泡 */
  (e: 'multiSelect', index: number): void
  /** 开始拖拽 */
  (e: 'dragStart', index: number, event: MouseEvent): void
  /** 拖拽中 */
  (e: 'dragging', deltaX: number, deltaY: number): void
  /** 拖拽结束 */
  (e: 'dragEnd', index: number, newCoords: BubbleCoords): void
  /** 开始调整大小 */
  (e: 'resizeStart', index: number, handle: string, event: MouseEvent): void
  /** 调整大小中 */
  (e: 'resizing', newCoords: BubbleCoords): void
  /** 调整大小结束 */
  (e: 'resizeEnd', index: number, newCoords: BubbleCoords): void
  /** 开始旋转 */
  (e: 'rotateStart', index: number, event: MouseEvent): void
  /** 旋转中 */
  (e: 'rotating', angle: number): void
  /** 旋转结束 */
  (e: 'rotateEnd', index: number, angle: number): void
  /** 绘制新气泡 */
  (e: 'drawBubble', coords: BubbleCoords): void
  // 注意：clearMultiSelect 事件已移除，清除多选逻辑在 EditWorkspace.handleMouseDown 中实现
}>()

// ============================================================
// 状态定义（本地状态，拖动状态从store共享）
// ============================================================

const overlayRef = ref<HTMLElement | null>(null)

// 拖拽辅助状态（本地）
const dragStartX = ref(0)
const dragStartY = ref(0)

// 调整大小辅助状态（本地）
const resizeHandle = ref('')
const resizeStartX = ref(0)
const resizeStartY = ref(0)
const resizeInitialCoords = ref<BubbleCoords | null>(null)

// 旋转辅助状态（本地）
const rotateStartAngle = ref(0)
const rotateInitialAngle = ref(0)
const rotateCenterX = ref(0)
const rotateCenterY = ref(0)

// 绘制状态
const isDrawing = ref(false)
const drawStartX = ref(0)
const drawStartY = ref(0)
const drawingRect = ref<BubbleCoords | null>(null)

// 中键绘制状态
const isMiddleButtonDown = ref(false)

// ============================================================
// 样式计算
// ============================================================

/**
 * 获取气泡高亮框样式
 */
function getBubbleStyle(bubble: BubbleState, index: number): Record<string, string> {
  let x1: number, y1: number, x2: number, y2: number
  let rotation = bubble.rotationAngle || 0
  
  // 如果正在拖拽这个气泡，使用实时偏移（从store共享状态）
  if (isDragging.value && draggingIndex.value === index) {
    const [bx1, by1, bx2, by2] = bubble.coords
    x1 = dragInitialX.value + dragOffsetX.value
    y1 = dragInitialY.value + dragOffsetY.value
    x2 = x1 + (bx2 - bx1)
    y2 = y1 + (by2 - by1)
  }
  // 如果正在调整大小这个气泡，使用实时坐标
  else if (isResizing.value && resizingIndex.value === index && resizeCurrentCoords.value) {
    [x1, y1, x2, y2] = resizeCurrentCoords.value
  }
  // 如果正在旋转这个气泡，使用实时角度
  else if (isRotating.value && rotatingIndex.value === index) {
    [x1, y1, x2, y2] = bubble.coords
    rotation = rotateCurrentAngle.value
  }
  else {
    [x1, y1, x2, y2] = bubble.coords
  }

  const width = x2 - x1
  const height = y2 - y1

  const style: Record<string, string> = {
    left: `${x1}px`,
    top: `${y1}px`,
    width: `${width}px`,
    height: `${height}px`
  }

  // 如果有旋转角度
  if (rotation) {
    style.transformOrigin = 'center center'
    style.transform = `rotate(${rotation}deg)`
  }

  return style
}

/**
 * 获取绘制中矩形的样式
 */
function getDrawingRectStyle(): Record<string, string> {
  if (!drawingRect.value) return {}

  const [x1, y1, x2, y2] = drawingRect.value
  return {
    left: `${Math.min(x1, x2)}px`,
    top: `${Math.min(y1, y2)}px`,
    width: `${Math.abs(x2 - x1)}px`,
    height: `${Math.abs(y2 - y1)}px`
  }
}

// ============================================================
// 坐标转换
// ============================================================

/**
 * 获取鼠标在图片原生坐标系中的位置
 */
function getMousePositionInImage(event: MouseEvent): { x: number; y: number } | null {
  if (!overlayRef.value) return null

  const rect = overlayRef.value.getBoundingClientRect()
  const scale = props.scale || 1

  // 计算鼠标相对于 overlay 的位置，然后转换为图片原生坐标
  const x = (event.clientX - rect.left) / scale
  const y = (event.clientY - rect.top) / scale

  return { x, y }
}

// ============================================================
// 事件处理 - 点击选择
// ============================================================

/**
 * 处理点击事件
 * 注意：Shift+点击的多选逻辑已在 handleBubbleMouseDown 中处理，
 * 这里不再重复处理，避免 toggle 两次导致多选被取消
 */
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

/**
 * 处理覆盖层鼠标按下（用于绘制新气泡和清除选择）
 */
function handleOverlayMouseDown(event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  
  // 中键绘制
  if (event.button === 1) {
    event.preventDefault()
    isMiddleButtonDown.value = true
    document.body.classList.add('middle-button-drawing')
    startDrawing(event)
    return
  }

  // 左键
  if (event.button !== 0) return

  // 注意：清除多选的逻辑已移至 EditWorkspace.handleMouseDown
  // 因为 .bubble-overlay 是 pointer-events: none，此函数不会被空白处点击触发

  // 绘制模式下开始绘制
  if (props.isDrawingMode) {
    startDrawing(event)
  }
}

// ============================================================
// 事件处理 - 拖拽移动
// ============================================================

/**
 * 处理气泡鼠标按下（开始拖拽）
 */
function handleBubbleMouseDown(index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  
  if (event.button !== 0) return
  
  // 阻止默认行为（文本选择等）
  event.preventDefault()
  event.stopPropagation()

  // 【复刻原版】Shift+点击进行多选
  if (event.shiftKey) {
    emit('multiSelect', index)
    return
  }

  // 如果点击的不是当前选中的气泡，先选中它
  if (index !== props.selectedIndex) {
    emit('select', index)
    return
  }

  // 开始拖拽
  startDragging(index, event)
}

/**
 * 开始拖拽
 */
function startDragging(index: number, event: MouseEvent): void {
  // 先移除可能残留的事件监听器
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  
  // 更新store中的共享状态
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

  // 绑定全局事件
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}

/**
 * 更新拖拽位置（实时视觉反馈）
 * 使用Vue响应式更新，与图片拖动相同的模式
 */
function updateDragging(event: MouseEvent): void {
  const scale = props.scale || 1
  const deltaX = (event.clientX - dragStartX.value) / scale
  const deltaY = (event.clientY - dragStartY.value) / scale

  // 直接更新ref值，Vue会自动触发重新渲染
  dragOffsetX.value = deltaX
  dragOffsetY.value = deltaY
}

/**
 * 完成拖拽
 */
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

  // 边界检查
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000

  const safeWidth = Math.min(width, imgWidth)
  const safeHeight = Math.min(height, imgHeight)

  newX1 = Math.max(0, Math.min(newX1, imgWidth - safeWidth))
  newY1 = Math.max(0, Math.min(newY1, imgHeight - safeHeight))

  const newCoords: BubbleCoords = [newX1, newY1, newX1 + safeWidth, newY1 + safeHeight]
  emit('dragEnd', wasIndex, newCoords)
}

// ============================================================
// 事件处理 - 调整大小
// ============================================================

/**
 * 开始调整大小
 */
function handleResizeStart(handle: string, index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  
  if (event.button !== 0) return
  
  // 阻止默认行为
  event.preventDefault()
  event.stopPropagation()
  
  // 先移除可能残留的事件监听器
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)

  isResizing.value = true
  resizingIndex.value = index
  resizeHandle.value = handle
  resizeStartX.value = event.clientX
  resizeStartY.value = event.clientY

  const bubble = props.bubbles[index]
  if (bubble) {
    resizeInitialCoords.value = [...bubble.coords] as BubbleCoords
    resizeCurrentCoords.value = [...bubble.coords] as BubbleCoords
  }

  emit('resizeStart', index, handle, event)

  // 绑定全局事件
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}

/**
 * 更新调整大小（实时视觉反馈）
 * 使用Vue响应式更新
 */
function updateResizing(event: MouseEvent): void {
  if (!resizeInitialCoords.value) return

  const scale = props.scale || 1
  const deltaX = (event.clientX - resizeStartX.value) / scale
  const deltaY = (event.clientY - resizeStartY.value) / scale

  let [x1, y1, x2, y2] = resizeInitialCoords.value

  // 根据手柄类型调整坐标
  const handle = resizeHandle.value
  if (handle.includes('w')) x1 += deltaX
  if (handle.includes('e')) x2 += deltaX
  if (handle.includes('n')) y1 += deltaY
  if (handle.includes('s')) y2 += deltaY

  // 确保有效性（交换坐标如果反转）
  if (x1 > x2) [x1, x2] = [x2, x1]
  if (y1 > y2) [y1, y2] = [y2, y1]

  // 最小尺寸限制
  const minSize = 10
  if (x2 - x1 < minSize || y2 - y1 < minSize) return

  // 直接更新ref值，Vue会自动触发重新渲染
  resizeCurrentCoords.value = [x1, y1, x2, y2]
}

/**
 * 完成调整大小
 */
function finishResizing(event: MouseEvent): void {
  if (!resizeInitialCoords.value) return

  const scale = props.scale || 1
  const deltaX = (event.clientX - resizeStartX.value) / scale
  const deltaY = (event.clientY - resizeStartY.value) / scale

  let [x1, y1, x2, y2] = resizeInitialCoords.value

  // 根据手柄类型调整坐标
  const handle = resizeHandle.value
  if (handle.includes('w')) x1 += deltaX
  if (handle.includes('e')) x2 += deltaX
  if (handle.includes('n')) y1 += deltaY
  if (handle.includes('s')) y2 += deltaY

  // 确保有效性
  if (x1 > x2) [x1, x2] = [x2, x1]
  if (y1 > y2) [y1, y2] = [y2, y1]

  // 边界约束
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000

  x1 = Math.max(0, Math.round(x1))
  y1 = Math.max(0, Math.round(y1))
  x2 = Math.min(imgWidth, Math.round(x2))
  y2 = Math.min(imgHeight, Math.round(y2))

  // 最小尺寸检查
  const minSize = 10
  if (x2 - x1 < minSize || y2 - y1 < minSize) {
    console.warn('调整后尺寸过小，已撤销')
    isResizing.value = false
    resizingIndex.value = -1
    resizeInitialCoords.value = null
    return
  }

  emit('resizeEnd', resizingIndex.value, [x1, y1, x2, y2])

  isResizing.value = false
  resizingIndex.value = -1
  resizeInitialCoords.value = null
  resizeCurrentCoords.value = null
}

// ============================================================
// 事件处理 - 旋转
// ============================================================

/**
 * 开始旋转
 */
function handleRotateStart(index: number, event: MouseEvent): void {
  // 笔刷模式下禁用气泡框交互（防御性检查，CSS已设置pointer-events:none）
  if (props.isBrushMode) return
  
  if (event.button !== 0) return
  
  // 阻止默认行为
  event.preventDefault()
  event.stopPropagation()
  
  // 先移除可能残留的事件监听器
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

  // 计算框中心在屏幕上的位置
  rotateCenterX.value = overlayRect.left + ((x1 + x2) / 2) * scale
  rotateCenterY.value = overlayRect.top + ((y1 + y2) / 2) * scale

  // 计算鼠标当前位置相对于中心的角度
  const dx = event.clientX - rotateCenterX.value
  const dy = event.clientY - rotateCenterY.value
  rotateStartAngle.value = Math.atan2(dy, dx) * 180 / Math.PI

  // 获取当前旋转角度
  rotateInitialAngle.value = bubble.rotationAngle || 0
  rotateCurrentAngle.value = bubble.rotationAngle || 0

  emit('rotateStart', index, event)

  // 添加旋转光标
  document.body.classList.add('rotating-box')

  // 绑定全局事件
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)

  console.log(`开始旋转气泡 #${index + 1}，初始角度: ${rotateInitialAngle.value}°`)
}

/**
 * 更新旋转（实时视觉反馈）
 * 使用Vue响应式更新
 */
function updateRotating(event: MouseEvent): void {
  // 计算鼠标当前位置相对于中心的角度
  const dx = event.clientX - rotateCenterX.value
  const dy = event.clientY - rotateCenterY.value
  const currentAngle = Math.atan2(dy, dx) * 180 / Math.PI

  // 计算旋转差值
  const deltaAngle = currentAngle - rotateStartAngle.value

  // 计算新角度
  let newAngle = rotateInitialAngle.value + deltaAngle

  // 限制角度范围 -180 到 180
  while (newAngle > 180) newAngle -= 360
  while (newAngle < -180) newAngle += 360

  // 按住 Shift 键时吸附到 15° 的倍数
  if (event.shiftKey) {
    newAngle = Math.round(newAngle / 15) * 15
  }

  // 直接更新ref值，Vue会自动触发重新渲染
  rotateCurrentAngle.value = newAngle
}

/**
 * 完成旋转
 */
function finishRotating(_event: MouseEvent): void {
  document.body.classList.remove('rotating-box')

  // 使用当前旋转角度作为最终角度
  const index = rotatingIndex.value
  const finalAngle = rotateCurrentAngle.value

  emit('rotateEnd', index, finalAngle)

  isRotating.value = false
  rotatingIndex.value = -1

  console.log(`气泡 #${index + 1} 旋转完成，角度: ${finalAngle}°`)
}

// ============================================================
// 事件处理 - 绘制新气泡
// ============================================================

/**
 * 开始绘制
 */
function startDrawing(event: MouseEvent): void {
  const pos = getMousePositionInImage(event)
  if (!pos) return

  // 边界检查
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000
  if (pos.x < 0 || pos.x > imgWidth || pos.y < 0 || pos.y > imgHeight) return

  isDrawing.value = true
  drawStartX.value = pos.x
  drawStartY.value = pos.y
  drawingRect.value = [pos.x, pos.y, pos.x, pos.y]

  // 绑定全局事件
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)

  console.log('开始绘制新框:', pos.x.toFixed(1), pos.y.toFixed(1))
}

/**
 * 更新绘制
 */
function updateDrawing(event: MouseEvent): void {
  const pos = getMousePositionInImage(event)
  if (!pos || !drawingRect.value) return

  drawingRect.value = [drawStartX.value, drawStartY.value, pos.x, pos.y]
}

/**
 * 完成绘制
 */
function finishDrawing(event: MouseEvent): void {
  const pos = getMousePositionInImage(event)

  // 清理中键状态
  if (isMiddleButtonDown.value) {
    isMiddleButtonDown.value = false
    document.body.classList.remove('middle-button-drawing')
  }

  if (!pos || !drawingRect.value) {
    isDrawing.value = false
    drawingRect.value = null
    return
  }

  // 计算框的坐标并进行边界检查
  const imgWidth = props.imageWidth || 2000
  const imgHeight = props.imageHeight || 2000

  const x1 = Math.max(0, Math.round(Math.min(drawStartX.value, pos.x)))
  const y1 = Math.max(0, Math.round(Math.min(drawStartY.value, pos.y)))
  const x2 = Math.min(imgWidth, Math.round(Math.max(drawStartX.value, pos.x)))
  const y2 = Math.min(imgHeight, Math.round(Math.max(drawStartY.value, pos.y)))

  // 检查框大小是否有效
  const minSize = 10
  if (x2 - x1 < minSize || y2 - y1 < minSize) {
    console.log('绘制的框太小，已忽略')
    isDrawing.value = false
    drawingRect.value = null
    return
  }

  console.log('完成绘制新框:', [x1, y1, x2, y2])
  emit('drawBubble', [x1, y1, x2, y2])

  isDrawing.value = false
  drawingRect.value = null
}

// ============================================================
// 全局事件处理
// ============================================================

/**
 * 全局鼠标移动处理
 */
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

/**
 * 全局鼠标释放处理
 */
function handleMouseUp(event: MouseEvent): void {
  // 立即解绑全局事件，防止重复触发
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
  
  // 中键释放
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

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  // 组件挂载时的初始化
})

onUnmounted(() => {
  // 清理全局事件
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
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  user-select: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  /* 【优化大图渲染】启用 GPU 加速，减少重绘闪烁 */
  transform: translateZ(0);
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
  will-change: contents;
}

/* 【修复问题1】笔刷模式下禁用气泡框的鼠标事件，让事件穿透到下层viewport */
.bubble-overlay.brush-mode .bubble-highlight-box,
.bubble-overlay.brush-mode .resize-handle,
.bubble-overlay.brush-mode .rotate-handle {
  pointer-events: none !important;
}

/* 矩形气泡高亮框 - 使用反向缩放保持边框在屏幕上的固定宽度 */
.bubble-highlight-box {
  position: absolute;
  /* 边框宽度反向缩放：屏幕上始终显示为 2px */
  border: calc(2px / var(--scale, 1)) solid rgba(255, 200, 0, 0.8);
  background: rgba(255, 200, 0, 0.1);
  cursor: pointer;
  pointer-events: auto;
  box-sizing: border-box;
  overflow: visible;
  /* 【优化大图渲染】GPU 加速和优化重绘 */
  will-change: transform, left, top, width, height;
  contain: layout style;
}

.bubble-highlight-box:hover {
  border-color: #ff6b6b;
  background: rgba(255, 107, 107, 0.2);
}

.bubble-highlight-box.selected {
  /* 选中时边框稍粗：屏幕上始终显示为 3px */
  border: calc(3px / var(--scale, 1)) solid #00ff88;
  background: rgba(0, 255, 136, 0.15);
  box-shadow: 0 0 calc(15px / var(--scale, 1)) rgba(0, 255, 136, 0.5);
  z-index: 10;
  cursor: grab;
}

.bubble-highlight-box.selected:active {
  cursor: grabbing;
}

.bubble-highlight-box.multi-selected {
  border: calc(3px / var(--scale, 1)) solid #ff1744 !important;
  background: rgba(255, 23, 68, 0.25) !important;
  box-shadow: 0 0 calc(12px / var(--scale, 1)) rgba(255, 23, 68, 0.6);
}

/* 气泡索引标签 - 反向缩放保持屏幕上固定大小 */
.bubble-index {
  position: absolute;
  /* 位置也需要反向缩放 */
  top: calc(-20px / var(--scale, 1));
  left: 0;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  font-size: calc(11px / var(--scale, 1));
  padding: calc(2px / var(--scale, 1)) calc(6px / var(--scale, 1));
  border-radius: calc(3px / var(--scale, 1));
  pointer-events: none;
  /* 使用 transform-origin 确保从左上角缩放 */
  transform-origin: left top;
}

.bubble-highlight-box.selected .bubble-index {
  background: rgba(0, 255, 136, 0.9);
  color: #1a1a2e;
}

/* 调整手柄 - 反向缩放保持屏幕上 10x10px */
.resize-handle {
  display: block !important;
  position: absolute;
  /* 尺寸反向缩放 */
  width: calc(10px / var(--scale, 1));
  height: calc(10px / var(--scale, 1));
  background: #00ff88;
  border: calc(2px / var(--scale, 1)) solid #fff;
  border-radius: calc(3px / var(--scale, 1));
  pointer-events: auto !important;
  z-index: 20;
  box-sizing: border-box;
  box-shadow: 0 0 calc(3px / var(--scale, 1)) rgba(0, 0, 0, 0.3);
}

.resize-handle:hover {
  background: #00cc6a;
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
  display: block !important;
  position: absolute;
  top: calc(-25px / var(--scale, 1));
  left: 50%;
  transform: translateX(-50%);
  width: calc(2px / var(--scale, 1));
  height: calc(20px / var(--scale, 1));
  background: rgba(0, 255, 136, 0.6);
  pointer-events: none;
}

/* 旋转手柄 - 反向缩放保持屏幕上 12x12px */
.rotate-handle {
  display: block !important;
  position: absolute;
  top: calc(-35px / var(--scale, 1));
  left: 50%;
  transform: translateX(-50%);
  width: calc(12px / var(--scale, 1));
  height: calc(12px / var(--scale, 1));
  background: #00ff88;
  border: calc(2px / var(--scale, 1)) solid #fff;
  border-radius: 50%;
  cursor: grab;
  pointer-events: auto !important;
  z-index: 15;
  box-shadow: 0 0 calc(6px / var(--scale, 1)) rgba(0, 255, 136, 0.8);
  transition: transform 0.15s, box-shadow 0.15s;
}

.rotate-handle:hover {
  transform: translateX(-50%) scale(1.2);
  box-shadow: 0 0 calc(10px / var(--scale, 1)) rgba(0, 255, 136, 1);
}

.rotate-handle:active {
  cursor: grabbing;
  background: #00cc6a;
}

/* 绘制中的矩形 - 边框也反向缩放 */
.drawing-rect {
  position: absolute;
  border: calc(2px / var(--scale, 1)) dashed #00ff88;
  background: rgba(0, 255, 136, 0.1);
  pointer-events: none;
}

</style>
