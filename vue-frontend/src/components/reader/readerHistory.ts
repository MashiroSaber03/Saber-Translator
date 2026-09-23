import type { ReaderPosition } from './readerLayout'

export interface ReaderRecord extends ReaderPosition {
  key: string
  offset: boolean
}
const STORAGE_KEY = 'readerHistory'

function records(): ReaderRecord[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    if (!Array.isArray(value)) return []
    return value
      .filter(
        (r): r is ReaderRecord =>
          r &&
          typeof r.key === 'string' &&
          typeof r.pageId === 'string' &&
          Number.isInteger(r.index) &&
          r.index >= 0 &&
          Number.isFinite(r.fraction) &&
          r.fraction >= 0 &&
          r.fraction <= 1 &&
          typeof r.offset === 'boolean'
      )
      .slice(0, 100)
  } catch {
    return []
  }
}

export function readerHistoryKey(user: string, book: string, chapter: string): string {
  return JSON.stringify([user, book, chapter])
}
export function loadReaderRecord(key: string): ReaderRecord | undefined {
  return records().find(record => record.key === key)
}
export function saveReaderRecord(record: ReaderRecord): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([record, ...records().filter(r => r.key !== record.key)].slice(0, 100))
    )
  } catch {
    /* Reading remains available when browser storage is unavailable. */
  }
}
