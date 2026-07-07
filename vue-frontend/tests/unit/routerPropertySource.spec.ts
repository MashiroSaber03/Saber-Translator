import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('router property source contracts', () => {
  it('keeps router property tests aligned with route constants and explicit assertions', () => {
    const propertyFile = source('tests/property/router.property.ts')

    expect(propertyFile).toContain("from '@/constants/routes'")
    expect(propertyFile).not.toContain('/**')
    expect(propertyFile).not.toContain('expectedName')
    expect(propertyFile).not.toMatch(/return\s+resolved\./)
    expect(propertyFile).not.toContain('redirect !== undefined ||')
  })

  it('keeps router unit path checks on the shared route constants', () => {
    const unitFile = source('tests/unit/router.test.ts')

    expect(unitFile).toContain("from '@/constants/routes'")
    expect(unitFile).not.toMatch(/toBe\('\/(translate|reader|insight)/)
    expect(unitFile).not.toContain("toBe('/')")
  })
})
