import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const nullableNumberOwners = [
  'src/components/insight/AnalysisProgress.vue',
  'src/components/insight/notes/NoteEditorModal.vue',
]

describe('Insight nullable number fields', () => {
  it('uses UiNumberField nullable instead of raw number inputs for optional page numbers', () => {
    for (const filePath of nullableNumberOwners) {
      const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')

      expect(source, filePath).toContain('UiNumberField')
      expect(source, filePath).toContain('nullable')
      expect(source, filePath).not.toMatch(/<UiInput\b[^\n]*\btype="number"|<UiInput\b[^\n]*\bv-model\.number/)
    }
  })
})
