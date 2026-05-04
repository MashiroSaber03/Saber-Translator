import * as XLSX from 'xlsx'

export interface TranslationConstraintColumn {
  key: string
  label: string
}

type TableRow = Record<string, string>

export function exportRowsToJson(rows: TableRow[]): string {
  return JSON.stringify(rows, null, 2)
}

export function importRowsFromJson(
  json: string,
  columns: readonly TranslationConstraintColumn[],
): TableRow[] {
  const parsed = JSON.parse(json)
  if (!Array.isArray(parsed)) {
    throw new Error('JSON 内容必须是数组')
  }

  return parsed
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
    .map(item => normalizeImportedRow(item, columns))
}

export function exportRowsToXlsxBuffer(
  rows: TableRow[],
  columns: readonly TranslationConstraintColumn[],
): ArrayBuffer {
  const data = rows.map(row => {
    const record: Record<string, string> = {}
    for (const column of columns) {
      record[column.label] = row[column.key] ?? ''
    }
    return record
  })

  const worksheet = XLSX.utils.json_to_sheet(data)
  const workbook = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(workbook, worksheet, 'Sheet1')
  return XLSX.write(workbook, { type: 'array', bookType: 'xlsx' }) as ArrayBuffer
}

export function importRowsFromXlsxBuffer(
  buffer: ArrayBuffer,
  columns: readonly TranslationConstraintColumn[],
): TableRow[] {
  const workbook = XLSX.read(buffer, { type: 'array' })
  const firstSheetName = workbook.SheetNames[0]
  if (!firstSheetName) {
    return []
  }

  const worksheet = workbook.Sheets[firstSheetName]
  if (!worksheet) {
    return []
  }
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(worksheet, {
    defval: '',
  })
  return rows.map(row => normalizeImportedRow(row, columns))
}

export function normalizeImportedRow(
  row: Record<string, unknown>,
  columns: readonly TranslationConstraintColumn[],
): TableRow {
  const normalized: TableRow = {}
  for (const column of columns) {
    const direct = row[column.key]
    const labeled = row[column.label]
    normalized[column.key] = String(direct ?? labeled ?? '')
  }
  return normalized
}

export function validateRegexEntries(
  rows: Array<Record<string, string>>,
  options: {
    patternField: string
    matchModeField?: string
  },
): string | null {
  const { patternField, matchModeField = 'matchMode' } = options
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index] || {}
    if ((row[matchModeField] || 'text') !== 'regex') {
      continue
    }

    const pattern = row[patternField] || ''
    if (!pattern) {
      continue
    }

    try {
      new RegExp(pattern)
    } catch (error) {
      return `第 ${index + 1} 行正则无效: ${error instanceof Error ? error.message : '未知错误'}`
    }
  }
  return null
}
