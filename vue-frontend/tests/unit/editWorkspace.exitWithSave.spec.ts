import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('EditWorkspace exit persistence', () => {
  it('flushes the current backend document before emitting exit', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/useEditWorkspace.ts'),
      'utf8',
    )
    const handler = source.slice(
      source.indexOf('async function handleExitToolbarAction'),
      source.indexOf('function handleBubbleUpdateWithSync'),
    )
    expect(handler).toMatch(/await persistCurrentDocument\(\)[\s\S]*?emit\('exit'\)/)
    expect(handler).toContain('showToast(')
  })

  it('does not render the deleted autosave/exit confirmation workflow', () => {
    const component = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditWorkspace.vue'),
      'utf8',
    )
    expect(component).not.toContain('EditExitSaveModal')
    expect(component).not.toContain('exitSaveDialog')
    expect(component).not.toContain('autoSaveInBookshelfMode')
  })
})
