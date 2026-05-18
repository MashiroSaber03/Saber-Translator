export function normalizePageSelection(selectedPages: number[]): number[] {
  return [...new Set(
    selectedPages
      .filter((page) => Number.isFinite(page))
      .map((page) => Math.floor(page))
      .filter((page) => page >= 1)
  )].sort((a, b) => a - b)
}

export function clampPageSelection(selectedPages: number[], totalImages: number): number[] {
  if (totalImages <= 0) return []
  return normalizePageSelection(selectedPages).filter((page) => page <= totalImages)
}

export function pageSelectionToPageIndexes(selectedPages: number[]): number[] {
  return normalizePageSelection(selectedPages).map((page) => page - 1)
}

export function pageIndexesToSelection(pageIndexes: number[]): number[] {
  return normalizePageSelection(pageIndexes.map((pageIndex) => pageIndex + 1))
}

export function createPageSelectionSummary(selectedPages: number[]): string {
  const normalized = normalizePageSelection(selectedPages)
  if (normalized.length === 0) {
    return '未选择页码'
  }
  if (normalized.length > 6) {
    return `已选 ${normalized.length} 页`
  }
  return `第 ${normalized.join('、')} 页`
}

