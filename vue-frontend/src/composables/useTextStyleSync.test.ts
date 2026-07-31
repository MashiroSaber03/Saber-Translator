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
import { useSettingsStore } from '@/stores/settings'

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

  it.each([
    ['fontSize', 31],
    ['layoutDirection', 'horizontal'],
    ['textColor', '#123456'],
    ['fillColor', '#abcdef'],
    ['strokeEnabled', false],
    ['strokeColor', '#654321'],
    ['strokeWidth', 5],
    ['lineSpacing', 1.6],
    ['textAlign', 'center'],
  ])('persists and propagates the %s page style through the backend command', async (
    settingKey,
    value,
  ) => {
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

    await useTextStyleSync().handleTextStyleChanged(settingKey, value)

    expect(persistenceMocks.queuePageDocumentMutation).toHaveBeenCalledWith(
      '00000000-0000-0000-0000-000000000001',
      3,
      [],
      {
        pageStyleDefaultsPatch: { [settingKey]: value },
        propagateStyleFields: [settingKey],
      },
    )
  })

  it('restores the manual page colors when automatic text color is disabled', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    settingsStore.updateTextStyle({
      textColor: '#123456',
      fillColor: '#abcdef',
      useAutoTextColor: true,
    })
    imageStore.setImages([{
      id: '00000000-0000-0000-0000-000000000001',
      chapterId: '00000000-0000-0000-0000-000000000002',
      documentRevision: 3,
      fileName: 'page.png',
      sourceAssetUrl: '/source',
      translatedAssetUrl: '/translated',
      cleanAssetUrl: '/clean',
      bubbleStates: [],
      translationStatus: 'completed',
      translationFailed: false,
      hasUnsavedChanges: false,
    }])

    const sync = useTextStyleSync()
    await sync.handleAutoTextColorChanged(false)

    expect(persistenceMocks.queuePageDocumentMutation).toHaveBeenCalledWith(
      '00000000-0000-0000-0000-000000000001',
      3,
      [],
      {
        pageStyleDefaultsPatch: {
          useAutoTextColor: false,
          textColor: '#123456',
          fillColor: '#abcdef',
        },
        propagateStyleFields: ['textColor', 'fillColor'],
      },
    )
  })
})
