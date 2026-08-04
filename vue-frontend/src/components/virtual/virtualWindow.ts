export interface VirtualWindow {
  end: number
  offset: number
  start: number
  totalSize: number
}

export function variableItemOffsets(itemSizes: readonly number[]): number[] {
  const offsets = new Array<number>(itemSizes.length + 1)
  offsets[0] = 0
  for (let index = 0; index < itemSizes.length; index += 1) {
    offsets[index + 1] = offsets[index]! + Math.max(1, itemSizes[index]!)
  }
  return offsets
}

function firstMatchingIndex(
  length: number,
  matches: (index: number) => boolean,
): number {
  let lower = 0
  let upper = length
  while (lower < upper) {
    const middle = lower + Math.floor((upper - lower) / 2)
    if (matches(middle)) upper = middle
    else lower = middle + 1
  }
  return lower
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
  cachedOffsets?: readonly number[],
): VirtualWindow {
  const offsets = cachedOffsets?.length === itemSizes.length + 1
    ? cachedOffsets
    : variableItemOffsets(itemSizes)
  const totalSize = offsets[offsets.length - 1] ?? 0
  const lowerBound = Math.max(0, scrollOffset - overscanPixels)
  const upperBound = Math.min(
    totalSize,
    Math.max(0, scrollOffset) + Math.max(0, viewportSize) + overscanPixels,
  )

  const start = firstMatchingIndex(
    itemSizes.length,
    index => offsets[index + 1]! >= lowerBound,
  )
  const end = firstMatchingIndex(
    itemSizes.length,
    index => offsets[index]! > upperBound,
  )
  return {
    start,
    end,
    offset: offsets[start] ?? 0,
    totalSize,
  }
}
