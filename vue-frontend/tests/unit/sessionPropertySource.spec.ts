import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const sessionPropertyFiles = [
  'tests/property/session-batch.property.ts',
  'tests/property/session-context.property.ts',
  'tests/property/session-data.property.ts',
  'tests/property/session-list.property.ts',
]

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('session property source contracts', () => {
  it('keeps session properties on current store contracts without migrated narration', () => {
    for (const file of sessionPropertyFiles) {
      const content = source(file)

      expect(content, file).toContain('function createSessionStore()')
      expect(content, file).not.toContain('beforeEach')
      expect(content, file).not.toContain('/**')
      expect(content, file).not.toContain('Property 34')
      expect(content, file).not.toMatch(/\/\/\s*(生成|验证|每次迭代)/)
    }
  })
})
