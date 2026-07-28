import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { createBubbleState } from '@/utils/bubbleFactory'

const { mutateMock } = vi.hoisted(() => ({
  mutateMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  mutatePageDocument: mutateMock,
}))

describe('page document persistence coordinator', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mutateMock.mockReset()
    vi.resetModules()
  })

  it('commits bubble and page style changes in one CAS without image payloads', async () => {
    mutateMock.mockResolvedValue({
      bubbles: [{
        bubbleId: 'bubble-1',
        fontId: 'font-2',
        ordinal: 1,
        payload: {
          coords: [0, 0, 100, 80],
          fontSize: 28,
          translatedText: '译文',
        },
        updatedRevision: 2,
      }],
      chapterId: 'chapter-1',
      defaultFontId: 'font-2',
      documentRevision: 2,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
    })
    const {
      queuePageDocumentMutation,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: 'font-1',
      documentRevision: 1,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
    })
    const bubble = createBubbleState({
      backendBubbleId: 'bubble-1',
      coords: [0, 0, 100, 80],
      fontFamily: 'font-2',
      fontSize: 28,
      polygon: [],
      translatedText: '译文',
    })

    await queuePageDocumentMutation('page-1', 1, [bubble], {
      defaultFontId: 'font-2',
      pageStyleDefaultsPatch: { fontFamily: 'font-2', fontSize: 28 },
      propagateStyleFields: ['fontFamily', 'fontSize'],
    })

    expect(mutateMock).toHaveBeenCalledWith('page-1', {
      baseRevision: 1,
      defaultFontId: 'font-2',
      mutations: [{
        bubbleId: 'bubble-1',
        fields: expect.objectContaining({
          fontId: 'font-2',
          fontSize: 28,
          translatedText: '译文',
        }),
        op: 'create',
      }],
      pageStyleDefaultsPatch: {
        fontFamily: 'font-2',
        fontSize: 28,
      },
      propagateStyleFields: ['fontFamily', 'fontSize'],
    })
    const request = mutateMock.mock.calls[0]?.[1]
    expect(JSON.stringify(request)).not.toContain('originalDataURL')
    expect(JSON.stringify(request)).not.toContain('base64')
  })
})
