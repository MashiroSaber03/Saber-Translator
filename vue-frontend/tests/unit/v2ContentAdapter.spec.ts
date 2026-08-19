import { describe, expect, it } from 'vitest'

import type { V2PageDocument, V2PageSummary } from '@/api/v2/content'
import { pageDocumentToBubbles, pageSummaryToImage } from '@/adapters/v2ContentAdapter'

function bubblePayload(
  overrides: Partial<V2PageDocument['bubbles'][number]['payload']> = {},
): V2PageDocument['bubbles'][number]['payload'] {
  return {
    originalText: '',
    translatedText: '',
    textboxText: '',
    coords: [0, 0, 100, 80],
    polygon: [],
    fontSize: 24,
    textDirection: 'vertical',
    autoTextDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#ffffff',
    rotationAngle: 0,
    position: { x: 0, y: 0 },
    strokeEnabled: false,
    strokeColor: '#ffffff',
    strokeWidth: 0,
    lineSpacing: 1.2,
    inlineAlign: 'center',
    blockAlign: 'end',
    inpaintMethod: 'solid',
    autoFgColor: null,
    autoBgColor: null,
    colorConfidence: 0,
    textlines: [],
    ocrResult: null,
    ...overrides,
  }
}

function pageSummary(overrides: Partial<V2PageSummary> = {}): V2PageSummary {
  return {
    id: 'page-1',
    chapterId: 'chapter-1',
    ordinal: 1,
    logicalSourcePath: 'page.png',
    sourceRevision: 1,
    documentRevision: 1,
    renderedRevision: null,
    renderStatus: 'not_rendered',
    detectionState: 'unprocessed',
    sourceUrl: '/source',
    thumbnailSourceUrl: '/thumbnail/source',
    cleanUrl: null,
    translatedUrl: null,
    width: null,
    height: null,
    ...overrides,
  }
}

function documentWithFont(options: {
  defaultFontId: string | null
  fontId: string | null
}): V2PageDocument {
  return {
    bubbles: [{
      bubbleId: 'bubble-1',
      fontId: options.fontId,
      ordinal: 1,
      payload: bubblePayload({
        coords: [0, 0, 100, 80],
        translatedText: '译文',
      }),
      updatedRevision: 1,
    }],
    chapterId: 'chapter-1',
    defaultFontId: options.defaultFontId,
    documentRevision: 1,
    pageId: 'page-1',
    pageStyleDefaults: {},
    pageStyleSchemaVersion: 2,
    renderStatus: 'not_rendered',
  }
}

describe('v2 content adapter', () => {
  it.each(['render_failed', 'repair_failed'] as const)(
    'maps the current %s render state to a failed image',
    (renderStatus) => {
      expect(pageSummaryToImage(pageSummary({ renderStatus })).translationStatus).toBe('failed')
    },
  )

  it.each(['rendering', 'awaiting_repair'] as const)(
    'maps the active %s render state to a processing image',
    (renderStatus) => {
      expect(pageSummaryToImage(pageSummary({ renderStatus })).translationStatus).toBe('processing')
    },
  )

  it('requires the current ready state and matching revision for a completed image', () => {
    expect(pageSummaryToImage(pageSummary({
      documentRevision: 3,
      renderedRevision: 3,
      renderStatus: 'ready',
      translatedUrl: '/translated',
    })).translationStatus).toBe('completed')
    expect(pageSummaryToImage(pageSummary({
      documentRevision: 3,
      renderedRevision: 3,
      renderStatus: 'stale',
      translatedUrl: '/translated',
    })).translationStatus).toBe('pending')
  })

  it('projects the canonical logical path into the file and folder fields used by navigation', () => {
    const image = pageSummaryToImage(pageSummary({
      logicalSourcePath: 'volume-1/chapter-a/page-001.png',
    }))

    expect(image).toMatchObject({
      fileName: 'page-001.png',
      folderPath: 'volume-1/chapter-a',
    })
    expect(image).not.toHaveProperty('relativePath')
    expect(image).not.toHaveProperty('sourceRevision')
  })

  it('keeps a root page at the folder-tree root', () => {
    expect(pageSummaryToImage(pageSummary({ logicalSourcePath: 'cover.png' }))).toMatchObject({
      fileName: 'cover.png',
    })
    expect(
      pageSummaryToImage(pageSummary({ logicalSourcePath: 'cover.png' })),
    ).not.toHaveProperty('folderPath')
  })

  it('uses the relational page font when the bubble has no override', () => {
    const [bubble] = pageDocumentToBubbles(documentWithFont({
      defaultFontId: 'font-page-default',
      fontId: null,
    }))

    expect(bubble?.fontFamily).toBe('font-page-default')
  })

  it('prefers a relational bubble font override over the page default', () => {
    const [bubble] = pageDocumentToBubbles(documentWithFont({
      defaultFontId: 'font-page-default',
      fontId: 'font-bubble-override',
    }))

    expect(bubble?.fontFamily).toBe('font-bubble-override')
  })

  it('rejects a document that has no relational font identity', () => {
    expect(() => pageDocumentToBubbles(documentWithFont({
      defaultFontId: null,
      fontId: null,
    }))).toThrow('缺少后端字体 ID')
  })

  it('rejects incomplete legacy bubble payloads instead of filling defaults', () => {
    const document = documentWithFont({
      defaultFontId: 'font-page-default',
      fontId: null,
    })
    document.bubbles[0]!.payload = {
      coords: [0, 0, 100, 80],
      translatedText: '旧数据',
    } as V2PageDocument['bubbles'][number]['payload']

    expect(() => pageDocumentToBubbles(document)).toThrow('不符合当前数据结构')
  })
})
