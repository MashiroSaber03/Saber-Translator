import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const continuationPanels = [
  'src/components/insight/ContinuationPanel.vue',
  'src/components/insight/continuation/ScriptGenerationPanel.vue',
  'src/components/insight/continuation/PageDetailsPanel.vue',
  'src/components/insight/continuation/ImageGenerationPanel.vue',
  'src/components/insight/continuation/ExportPanel.vue',
]

function readPanelSource(path: string): string {
  return readFileSync(
    resolve(process.cwd(), path),
    'utf8',
  )
}

function cssRuleBodies(source: string, className: string): string[] {
  return [...source.matchAll(new RegExp(`\\.${className}\\s*\\{([^{}]*)\\}`, 'g'))]
    .map(match => match[1] || '')
}

describe('Continuation section headers', () => {
  it('uses the shared product section header across continuation step panels', () => {
    for (const panelPath of continuationPanels) {
      const source = readPanelSource(panelPath)

      expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
      expect(source).toContain('<ProductSectionHeader')
      expect(source).not.toContain('<h3')
      expect(source).not.toMatch(/\.[\w-]+ h3\s*\{/)
    }
  })

  it('keeps the continuation step shell as the only outer padding owner', () => {
    const parentSource = readPanelSource('src/components/insight/ContinuationPanel.vue')

    expect(parentSource).toMatch(/\.continuation-panel__step-panel\s*\{[\s\S]*padding:\s*24px/)
    expect(parentSource).not.toMatch(/\.(?:analysis-sync-bar|step-content|step-panel)\b/)

    for (const panelPath of continuationPanels.slice(1)) {
      const source = readPanelSource(panelPath)
      const rootClass = panelPath.split('/').at(-1)!
        .replace(/Panel\.vue$/, '-panel')
        .replace(/([a-z])([A-Z])/g, '$1-$2')
        .toLowerCase()

      for (const body of cssRuleBodies(source, rootClass)) {
        expect(body).not.toMatch(/\bpadding(?:-[\w-]+)?\s*:/)
      }
    }
  })
})
