import { describe, expect, it } from 'vitest'

import {
  exportRowsToJson,
  exportRowsToXlsxBuffer,
  importRowsFromJson,
  importRowsFromXlsxBuffer,
} from '@/utils/translationConstraintTable'

describe('translation constraint table utils', () => {
  const columns = [
    { key: 'source', label: '原文' },
    { key: 'target', label: '译文' },
    { key: 'note', label: '备注' },
    { key: 'matchMode', label: '匹配方式' },
  ] as const

  const rows = [
    { source: 'Alice', target: '爱丽丝', note: '主角', matchMode: 'text' },
    { source: '^dragon$', target: '巨龙', note: '', matchMode: 'regex' },
  ]

  it('round-trips constraint rows through JSON import/export', () => {
    const json = exportRowsToJson(rows)
    expect(importRowsFromJson(json, columns)).toEqual(rows)
  })

  it('round-trips constraint rows through XLSX import/export', () => {
    const buffer = exportRowsToXlsxBuffer(rows, columns)
    expect(importRowsFromXlsxBuffer(buffer, columns)).toEqual(rows)
  })
})
