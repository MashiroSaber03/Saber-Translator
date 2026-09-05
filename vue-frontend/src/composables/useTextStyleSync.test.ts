import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const persistenceMocks = vi.hoisted(() => ({
  flushPageDocument: vi.fn(),
  queuePageDocumentMutation: vi.fn(),
}))

const translationMocks = vi.hoisted(() => ({
  createChapterStyleApplyJob: vi.fn(),
}))

const taskCenterMocks = vi.hoisted(() => ({
  trackJob: vi.fn(),
}))

const contentMocks = vi.hoisted(() => ({
  getPageRenderStatus: vi.fn(),
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  flushPageDocument: persistenceMocks.flushPageDocument,
  queuePageDocumentMutation: persistenceMocks.queuePageDocumentMutation,
}))

vi.mock('@/api/v2/content', () => ({
  getPageRenderStatus: contentMocks.getPageRenderStatus,
}))

vi.mock('@/api/v2/translation', () => ({
  createChapterStyleApplyJob: translationMocks.createChapterStyleApplyJob,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => ({ trackJob: taskCenterMocks.trackJob }),
}))

import { useTextStyleSync } from './useTextStyleSync'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'

describe('useTextStyleSync page defaults', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    persistenceMocks.flushPageDocument.mockReset()
    persistenceMocks.flushPageDocument.mockResolvedValue(undefined)
    persistenceMocks.queuePageDocumentMutation.mockReset()
    persistenceMocks.queuePageDocumentMutation.mockResolvedValue(undefined)
    translationMocks.createChapterStyleApplyJob.mockReset()
    translationMocks.createChapterStyleApplyJob.mockResolvedValue({ jobIds: ['job-1'] })
    taskCenterMocks.trackJob.mockReset()
    contentMocks.getPageRenderStatus.mockReset()
    contentMocks.getPageRenderStatus.mockResolvedValue({
      pageId: '00000000-0000-0000-0000-000000000001',
      renderedRevision: 3,
      renderStatus: 'ready',
      translatedUrl: '/translated?revision=3',
    })
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
    ['strokeWidth', 1.2],
    ['lineSpacing', 1.6],
    ['inlineAlign', 'center'],
    ['blockAlign', 'end'],
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
      textColor: '#123456',
      fillColor: '#abcdef',
      useAutoTextColor: true,
      bubbleStates: [],
      translationStatus: 'completed',
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
    await vi.waitFor(() => {
      expect(imageStore.currentImage?.renderedRevision).toBe(3)
    })
  })

  it('flushes the source page style and freezes its committed revision for apply-to-all', async () => {
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
      hasUnsavedChanges: false,
    }])
    persistenceMocks.flushPageDocument.mockImplementation(async () => {
      imageStore.updateCurrentImage({ documentRevision: 4 })
    })

    await useTextStyleSync().handleApplyToAll({
      fillColor: false,
      fontFamily: false,
      fontSize: false,
      layoutDirection: false,
      lineSpacing: false,
      strokeColor: false,
      strokeEnabled: false,
      strokeWidth: false,
      inlineAlign: false,
      blockAlign: false,
      textColor: true,
    })

    expect(persistenceMocks.flushPageDocument).toHaveBeenCalledWith(
      '00000000-0000-0000-0000-000000000001',
    )
    expect(translationMocks.createChapterStyleApplyJob).toHaveBeenCalledWith(
      '00000000-0000-0000-0000-000000000002',
      {
        selectedFields: ['textColor'],
        sourceDocumentRevision: 4,
        sourcePageId: '00000000-0000-0000-0000-000000000001',
      },
    )
  })
})
