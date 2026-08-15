import type { BubbleCoords, BubbleState } from '@/types/bubble'

interface BubbleStyleState {
  bubble: BubbleState
  index: number
  isDragging: boolean
  draggingIndex: number
  dragInitialX: number
  dragInitialY: number
  dragOffsetX: number
  dragOffsetY: number
  isResizing: boolean
  resizingIndex: number
  resizeCurrentCoords: BubbleCoords | null
  isRotating: boolean
  rotatingIndex: number
  rotateCurrentAngle: number
}

export function buildBubbleOverlayStyle(state: BubbleStyleState): Record<string, string> {
  const { bubble, index } = state
  let x1: number, y1: number, x2: number, y2: number
  let rotation = bubble.rotationAngle || 0

  if (state.isDragging && state.draggingIndex === index) {
    const [bx1, by1, bx2, by2] = bubble.coords
    x1 = state.dragInitialX + state.dragOffsetX
    y1 = state.dragInitialY + state.dragOffsetY
    x2 = x1 + (bx2 - bx1)
    y2 = y1 + (by2 - by1)
  } else if (state.isResizing && state.resizingIndex === index && state.resizeCurrentCoords) {
    [x1, y1, x2, y2] = state.resizeCurrentCoords
  } else if (state.isRotating && state.rotatingIndex === index) {
    [x1, y1, x2, y2] = bubble.coords
    rotation = state.rotateCurrentAngle
  } else {
    [x1, y1, x2, y2] = bubble.coords
  }

  const style: Record<string, string> = {
    left: `${x1}px`,
    top: `${y1}px`,
    width: `${x2 - x1}px`,
    height: `${y2 - y1}px`
  }

  if (rotation) {
    style.transformOrigin = 'center center'
    style.transform = `rotate(${rotation}deg)`
  }

  return style
}

export function buildDrawingRectStyle(rect: BubbleCoords | null): Record<string, string> {
  if (!rect) return {}

  const [x1, y1, x2, y2] = rect
  return {
    position: 'absolute',
    left: `${Math.min(x1, x2)}px`,
    top: `${Math.min(y1, y2)}px`,
    width: `${Math.abs(x2 - x1)}px`,
    height: `${Math.abs(y2 - y1)}px`
  }
}
