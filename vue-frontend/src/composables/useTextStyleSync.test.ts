import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const persistenceMocks = vi.hoisted(() => ({
  queuePageDocumentMutation: vi.fn(),
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  queuePageDocumentMutation: persistenceMocks.queuePageDocumentMutation,
}))

vi.mock('@/api/v2/content', () => ({
  getPageSummary: vi.fn(),
}))

vi.mock('@/api/v2/translation', () => ({
  createChapterStyleApplyJob: vi.fn(),
}))

import { useTextStyleSync } from './useTextStyleSync'
import { useImageStore } from '@/stores/imageStore'

describe('useTextStyleSync page defaults', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    persistenceMocks.queuePageDocumentMutation.mockReset()
    persistenceMocks.queuePageDocumentMutation.mockResolvedValue(undefined)
  })

  it('persists inpaintMethod as a page default without propagating it to existing bubbles', async () => {
    const imageStore = useImageStore()
    imageStore.setImages([{
      id: '00000000-0000-0000-0000-000000000001',
      chapterId: '00000000-0000-0000-0000-000000000002',
      documentRevision: 3,
      fileName: 'page.png',
      sourceAssetUrl: '/source',
      translatedAssetUrl: null,
      cleanAssetUrl: null,
      bubbleStates: [],
      translationStatus: 'pending',
      translationFailed: false,
      hasUnsavedChanges: false,
    }])

    const sync = useTextStyleSync()
    await sync.handleTextStyleChanged('inpaintMethod', 'lama_mpe')

    expect(persistenceMocks.queuePageDocumentMutation).toHaveBeenCalledWith(
      '00000000-0000-0000-0000-000000000001',
      3,
      [],
      {
        pageStyleDefaultsPatch: { inpaintMethod: 'lama_mpe' },
        propagateStyleFields: [],
      },
    )
  })
})
