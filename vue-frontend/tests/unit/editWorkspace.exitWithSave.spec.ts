import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('EditWorkspace exit persistence', () => {
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
