export function normalizePageSelection(selectedPages: number[]): number[] {
  return [...new Set(
    selectedPages
      .filter((page) => Number.isInteger(page))
      .filter((page) => page >= 1)
  )].sort((a, b) => a - b)
}

export interface ParsedPageSelection {
  pages: number[]
  error: string
}

export function parsePageSelectionText(
  input: string,
  totalImages: number,
): ParsedPageSelection {
  if (totalImages <= 0) {
    return { pages: [], error: '当前没有可选择的图片' }
  }

  const source = input.trim()
  if (!source) {
    return { pages: [], error: '请输入页码或页码范围' }
  }

  const normalized = source
    .replace(/\s*[-~～–—]\s*/g, '-')
    .replace(/[，、；;]/g, ',')
  const tokens = normalized.split(/[,\s]+/).filter(Boolean)
  const pages: number[] = []

  for (const token of tokens) {
    const singlePage = token.match(/^\d+$/)
    const pageRange = token.match(/^(\d+)-(\d+)$/)
    if (!singlePage && !pageRange) {
      return { pages: [], error: `无法识别页码“${token}”` }
    }

    const start = Number(pageRange?.[1] ?? token)
    const end = Number(pageRange?.[2] ?? token)
    if (start < 1 || end < 1) {
      return { pages: [], error: '页码必须从 1 开始' }
    }
    if (start > end) {
      return { pages: [], error: `页码范围“${token}”的起始页不能大于结束页` }
    }
    if (end > totalImages) {
      return { pages: [], error: `页码 ${end} 超出当前总页数 ${totalImages}` }
    }

    for (let page = start; page <= end; page += 1) {
      pages.push(page)
    }
  }

  return { pages: normalizePageSelection(pages), error: '' }
}

export function clampPageSelection(selectedPages: number[], totalImages: number): number[] {
  if (totalImages <= 0) return []
  return normalizePageSelection(selectedPages).filter((page) => page <= totalImages)
}

export function pageSelectionToPageIndexes(selectedPages: number[]): number[] {
  return normalizePageSelection(selectedPages).map((page) => page - 1)
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

