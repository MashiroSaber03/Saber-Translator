import type { NoteData, NoteType } from '@/types/insight'
import { toCamelCase } from '@/utils/insightConverters'
import type { NoteData as ApiNoteData } from '@/api/insight'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNoteType(value: unknown): value is NoteType {
  return value === 'text' || value === 'qa'
}

function hasOptionalString(value: Record<string, unknown>, key: string): boolean {
  return value[key] === undefined || typeof value[key] === 'string'
}

function hasOptionalNumber(value: Record<string, unknown>, key: string): boolean {
  return value[key] === undefined || (typeof value[key] === 'number' && Number.isFinite(value[key]))
}

function isCitation(value: unknown): value is { page: number; content: string } {
  return isRecord(value)
    && typeof value.page === 'number'
    && Number.isFinite(value.page)
    && typeof value.content === 'string'
}

export function isInsightNoteData(value: unknown): value is NoteData {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string' || !isNoteType(value.type) || typeof value.content !== 'string') {
    return false
  }
  if (!hasOptionalNumber(value, 'pageNum')) return false
  if (!hasOptionalString(value, 'createdAt')) return false
  if (!hasOptionalString(value, 'updatedAt')) return false
  if (!hasOptionalString(value, 'title')) return false
  if (!hasOptionalString(value, 'question')) return false
  if (!hasOptionalString(value, 'answer')) return false
  if (!hasOptionalString(value, 'comment')) return false
  if (value.tags !== undefined && (!Array.isArray(value.tags) || !value.tags.every(tag => typeof tag === 'string'))) {
    return false
  }
  if (value.citations !== undefined && (!Array.isArray(value.citations) || !value.citations.every(isCitation))) {
    return false
  }
  return true
}

export function filterValidInsightNotes(values: unknown[]): NoteData[] {
  return values.filter(isInsightNoteData)
}

export function mapInsightApiNote(note: ApiNoteData): NoteData {
  const converted = toCamelCase(note)
  if (!isInsightNoteData(converted)) {
    throw new Error('笔记响应格式无效')
  }
  return converted
}
