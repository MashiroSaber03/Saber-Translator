import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

describe('ImageGenerationPanel layout styles', () => {
  it('keeps a desktop two-column grid with a mobile fallback and fully visible images', () => {
    const filePath = resolve(process.cwd(), 'src/components/insight/continuation/ImageGenerationPanel.vue')
    const source = readFileSync(filePath, 'utf-8')

    expect(source).toContain('grid-template-columns: repeat(2, minmax(0, 1fr));')
    expect(source).toContain('@media (max-width: 1024px)')
    expect(source).toContain('grid-template-columns: 1fr;')
    expect(source).toContain('object-fit: contain;')
  })
})
