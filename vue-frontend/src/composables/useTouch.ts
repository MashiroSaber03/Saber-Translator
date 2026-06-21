export type SwipeDirection = 'left' | 'right' | 'up' | 'down' | null

export function detectSwipe(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  threshold = 50
): SwipeDirection {
  const dx = endX - startX
  const dy = endY - startY
  const absDx = Math.abs(dx)
  const absDy = Math.abs(dy)

  if (absDx < threshold && absDy < threshold) {
    return null
  }

  if (absDx > absDy) {
    return dx > 0 ? 'right' : 'left'
  }

  return dy > 0 ? 'down' : 'up'
}

export function calculatePinchScale(initialDistance: number, currentDistance: number): number {
  if (initialDistance <= 0) {
    return 1
  }

  return currentDistance / initialDistance
}
