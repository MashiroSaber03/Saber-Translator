import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('EditWorkspace backend-owned navigation', () => {
  const source = readFileSync(
    resolve(process.cwd(), 'src/components/edit/useEditWorkspace.ts'),
    'utf8',
  )

  it('persists the authoritative page document before ordinary navigation', () => {
    expect(source).toContain('await prepareForNavigation()')
    expect(source).toContain('await queuePageDocumentSave(')
    expect(source).toContain('navigateAfterPersist(() => imageStore.goToNext())')
    expect(source).toContain('navigateAfterPersist(() => imageStore.goToPrevious())')
    expect(source).toContain('navigateAfterPersist(() => imageStore.setCurrentImageIndex(index))')
  })

  it('applies through backend rendering before advancing', () => {
    expect(source).toContain('const renderSucceeded = await reRenderFullImage()')
    expect(source).toMatch(/if \(!renderSucceeded\)[\s\S]*?return[\s\S]*?imageStore\.goToNext\(\)/)
  })

  it('contains no legacy chapter-session initialization or browser save steps', () => {
    expect(source).not.toContain('isBookshelfSessionInitialized')
    expect(source).not.toContain('forceInitializeBookshelfSession')
    expect(source).not.toContain('saveBookshelfPageProgress')
    expect(source).not.toContain('saveStep')
  })
})
