import { describe, expect, it } from 'vitest'

import type { V2PageDocument } from '@/api/v2/content'
import { pageDocumentToBubbles } from '@/adapters/v2ContentAdapter'

function documentWithFont(options: {
  defaultFontId: string | null
  fontId: string | null
  payloadFontFamily?: string
}): V2PageDocument {
  return {
    bubbles: [{
      bubbleId: 'bubble-1',
      fontId: options.fontId,
      ordinal: 1,
      payload: {
        coords: [0, 0, 100, 80],
        fontFamily: options.payloadFontFamily,
        translatedText: '译文',
      },
      updatedRevision: 1,
    }],
    chapterId: 'chapter-1',
    defaultFontId: options.defaultFontId,
    documentRevision: 1,
    pageId: 'page-1',
    pageStyleDefaults: {},
    pageStyleSchemaVersion: 1,
  }
}

describe('v2 content adapter', () => {
  it('uses the relational page font when a legacy payload contains an empty fontFamily', () => {
    const [bubble] = pageDocumentToBubbles(documentWithFont({
      defaultFontId: 'font-page-default',
      fontId: null,
      payloadFontFamily: '',
    }))

    expect(bubble?.fontFamily).toBe('font-page-default')
  })

  it('prefers a bubble font override over page and legacy payload fonts', () => {
    const [bubble] = pageDocumentToBubbles(documentWithFont({
      defaultFontId: 'font-page-default',
      fontId: 'font-bubble-override',
      payloadFontFamily: 'legacy-font',
    }))

    expect(bubble?.fontFamily).toBe('font-bubble-override')
  })

  it('does not accept a legacy payload font as a second source of truth', () => {
    expect(() => pageDocumentToBubbles(documentWithFont({
      defaultFontId: null,
      fontId: null,
      payloadFontFamily: 'fonts/legacy.ttf',
    }))).toThrow('缺少后端字体 ID')
  })
})
