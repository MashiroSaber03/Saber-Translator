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
    expect(source).toMatch(/await Promise\.all\(\[\s*queuePageDocumentSave\([\s\S]*?flushPageDocument\(image\.id\),\s*\]\)/)
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

  it('waits for the authoritative page style before allowing edits', () => {
    expect(source).toContain('const isPageDocumentReady = ref(false)')
    expect(source).toContain('...document.pageStyleDefaults')
    expect(source).toContain('? { fontFamily: document.defaultFontId }')
    expect(source).toContain('!isPageDocumentReady.value')
    expect(source).toContain("{ flush: 'sync' }")
  })
})
