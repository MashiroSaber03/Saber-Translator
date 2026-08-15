import { ref } from 'vue'
import type { BubbleCoords } from '@/types/bubble'
import type { ResizeHandle } from '@/utils/bubbleResize'

export function useBubbleOverlayInteractionState() {
  const isDragging = ref(false)
  const draggingIndex = ref(-1)
  const dragOffsetX = ref(0)
  const dragOffsetY = ref(0)
  const dragInitialX = ref(0)
  const dragInitialY = ref(0)
  const dragStartX = ref(0)
  const dragStartY = ref(0)

  const isResizing = ref(false)
  const resizingIndex = ref(-1)
  const resizeCurrentCoords = ref<BubbleCoords | null>(null)
  const resizeHandle = ref<ResizeHandle | ''>('')
  const resizeStartX = ref(0)
  const resizeStartY = ref(0)
  const resizeInitialCoords = ref<BubbleCoords | null>(null)

  const isRotating = ref(false)
  const rotatingIndex = ref(-1)
  const rotateCurrentAngle = ref(0)
  const rotateStartAngle = ref(0)
  const rotateInitialAngle = ref(0)
  const rotateCenterX = ref(0)
  const rotateCenterY = ref(0)

  function resetDragging(): void {
    isDragging.value = false
    draggingIndex.value = -1
    dragOffsetX.value = 0
    dragOffsetY.value = 0
  }

  function resetResizing(): void {
    isResizing.value = false
    resizingIndex.value = -1
    resizeCurrentCoords.value = null
    resizeInitialCoords.value = null
    resizeHandle.value = ''
  }

  function resetRotating(): void {
    isRotating.value = false
    rotatingIndex.value = -1
  }

  return {
    isDragging,
    draggingIndex,
    dragOffsetX,
    dragOffsetY,
    dragInitialX,
    dragInitialY,
    dragStartX,
    dragStartY,
    isResizing,
    resizingIndex,
    resizeCurrentCoords,
    resizeHandle,
    resizeStartX,
    resizeStartY,
    resizeInitialCoords,
    isRotating,
    rotatingIndex,
    rotateCurrentAngle,
    rotateStartAngle,
    rotateInitialAngle,
    rotateCenterX,
    rotateCenterY,
    resetDragging,
    resetResizing,
    resetRotating,
  }
}
