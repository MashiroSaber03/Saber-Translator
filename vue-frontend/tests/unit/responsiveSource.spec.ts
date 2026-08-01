import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const responsivePropertyFiles = [
  'tests/property/responsive-sidebar.property.ts',
]

describe('responsive property source contracts', () => {
  it('keeps responsive properties bound to current responsive owners', () => {
    for (const file of responsivePropertyFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('响应式布局')
      expect(source, file).not.toContain('/' + '**')
      expect(source, file).not.toContain('验证')
      expect(source, file).not.toContain('return true')
    }

    const sidebarSource = readFileSync(resolve(process.cwd(), 'tests/property/responsive-sidebar.property.ts'), 'utf8')
    expect(sidebarSource).toContain("from '@/components/ui/SidebarLayout.vue'")
    expect(sidebarSource).not.toContain('function shouldShowSidebar')
    expect(sidebarSource).not.toContain('function calculateSidebarWidth')
  })
})
