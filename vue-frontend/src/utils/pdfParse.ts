export interface DocumentParseBatchPlan {
  batchIndex: number
  startIndex: number
  count: number
  processedPages: number
  hasMore: boolean
}

function toWholePageCount(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }
  return Math.max(0, Math.trunc(value))
}

function toBatchSize(value: number): number {
  return Math.max(1, toWholePageCount(value))
}

export function buildDocumentParseBatches(
  totalPages: number,
  batchSize: number,
): DocumentParseBatchPlan[] {
  const total = toWholePageCount(totalPages)
  if (total === 0) {
    return []
  }

  const size = toBatchSize(batchSize)
  const batches: DocumentParseBatchPlan[] = []

  for (let startIndex = 0, batchIndex = 0; startIndex < total; startIndex += size, batchIndex += 1) {
    const count = Math.min(size, total - startIndex)
    const processedPages = startIndex + count
    batches.push({
      batchIndex,
      startIndex,
      count,
      processedPages,
      hasMore: processedPages < total,
    })
  }

  return batches
}

export function calculateDocumentParseProgress(processedPages: number, totalPages: number): number {
  const total = toWholePageCount(totalPages)
  if (total === 0) {
    return 100
  }

  const processed = Math.min(toWholePageCount(processedPages), total)
  return Math.round((processed / total) * 100)
}

export function createDocumentPageFileName(sourceFileName: string, pageNumber: number): string {
  const fileName = sourceFileName.trim().split(/[\\/]/).pop() || ''
  const baseName = fileName.replace(/\.[^./\\]+$/, '') || 'document'
  const page = Math.max(1, Math.trunc(Number.isFinite(pageNumber) ? pageNumber : 1))
  return `${baseName}_page_${String(page).padStart(3, '0')}.png`
}
