import type { ReaderFit } from './readerSettings'

export interface PageSize {
  id: string
  width: number | null
  height: number | null
}
export interface ReaderPosition {
  pageId: string
  index: number
  fraction: number
}

export function pageIndexAtOffset(offsets: readonly number[], scroll: number): number {
  let low = 0,
    high = offsets.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (offsets[middle]! <= scroll + 0.5) low = middle + 1
    else high = middle
  }
  return Math.max(0, Math.min(offsets.length - 2, low - 1))
}

export function pageGroups(
  pages: readonly PageSize[],
  double: boolean,
  offset: boolean
): number[][] {
  const groups: number[][] = []
  const wide = (index: number) => {
    const page = pages[index]
    return Boolean(page?.width && page.height && page.width > page.height)
  }
  for (let index = 0; index < pages.length;) {
    const pair =
      double &&
      !(offset && index === 0) &&
      !wide(index) &&
      index + 1 < pages.length &&
      !wide(index + 1)
    groups.push(pair ? [index, index + 1] : [index])
    index += pair ? 2 : 1
  }
  return groups
}

export function fitScale(
  width: number,
  height: number,
  availableWidth: number,
  availableHeight: number,
  fit: ReaderFit
): number {
  const x = Math.max(1, availableWidth) / Math.max(1, width)
  const y = Math.max(1, availableHeight) / Math.max(1, height)
  if (fit === 'original') return 1
  return Math.min(1, fit === 'width' ? x : fit === 'height' ? y : Math.min(x, y))
}

export function resolvePosition(pages: readonly PageSize[], position: ReaderPosition): number {
  const found = pages.findIndex(page => page.id === position.pageId)
  return found >= 0 ? found : Math.max(0, Math.min(pages.length - 1, position.index))
}
