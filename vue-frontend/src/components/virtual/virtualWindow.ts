export interface VirtualWindow {
  end: number
  offset: number
  start: number
  totalSize: number
}

export function fixedVirtualWindow(
  itemCount: number,
  itemSize: number,
  scrollOffset: number,
  viewportSize: number,
  overscanItems = 4,
): VirtualWindow {
  if (itemCount <= 0 || itemSize <= 0) {
    return { start: 0, end: 0, offset: 0, totalSize: 0 }
  }
  const visibleStart = Math.floor(Math.max(0, scrollOffset) / itemSize)
  const visibleEnd = Math.ceil(
    (Math.max(0, scrollOffset) + Math.max(0, viewportSize)) / itemSize,
  )
  const start = Math.max(0, visibleStart - overscanItems)
  const end = Math.min(itemCount, visibleEnd + overscanItems)
  return {
    start,
    end,
    offset: start * itemSize,
    totalSize: itemCount * itemSize,
  }
}

export function variableVirtualWindow(
  itemSizes: readonly number[],
  scrollOffset: number,
  viewportSize: number,
  overscanPixels: number,
): VirtualWindow {
  const offsets = new Array<number>(itemSizes.length + 1)
  offsets[0] = 0
  for (let index = 0; index < itemSizes.length; index += 1) {
    offsets[index + 1] = offsets[index]! + Math.max(1, itemSizes[index]!)
  }
  const totalSize = offsets[offsets.length - 1] ?? 0
  const lowerBound = Math.max(0, scrollOffset - overscanPixels)
  const upperBound = Math.min(
    totalSize,
    Math.max(0, scrollOffset) + Math.max(0, viewportSize) + overscanPixels,
  )

  let start = 0
  while (start < itemSizes.length && offsets[start + 1]! < lowerBound) {
    start += 1
  }
  let end = start
  while (end < itemSizes.length && offsets[end]! <= upperBound) {
    end += 1
  }
  return {
    start,
    end,
    offset: offsets[start] ?? 0,
    totalSize,
  }
}
