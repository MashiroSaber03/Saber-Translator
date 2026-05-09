import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'

const {
  executeAtomicStepMock,
  preSaveOriginalImagesMock,
  finalizeSaveMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  preSaveOriginalImagesMock: vi.fn(),
  finalizeSaveMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
  executeBatchAtomicStep: vi.fn(),
}))

vi.mock('@/composables/useValidation', () => ({
  useValidation: () => ({
    validateBeforeTranslation: () => true,
  }),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  }),
}))

vi.mock('@/composables/translation/core/saveStep', async () => {
  const actual = await vi.importActual<object>('@/composables/translation/core/saveStep')
  return {
    ...actual,
    shouldEnableAutoSave: () => true,
    preSaveOriginalImages: preSaveOriginalImagesMock,
    finalizeSave: finalizeSaveMock,
    resetSaveState: vi.fn(),
  }
})

describe('useSequentialPipeline completion projection', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executeAtomicStepMock.mockReset()
    preSaveOriginalImagesMock.mockReset()
    finalizeSaveMock.mockReset()

    preSaveOriginalImagesMock.mockResolvedValue(true)
    finalizeSaveMock.mockResolvedValue(undefined)

    executeAtomicStepMock.mockImplementation(async (step: string, task: any) => {
      switch (step) {
        case 'detection':
          return { ...task, bubbleCoords: [[0, 0, 10, 10]], bubbleAngles: [0], autoDirections: ['vertical'], bubbleStates: [], textlinesPerBubble: [] }
        case 'ocr':
          return { ...task, originalTexts: ['原文'], ocrResults: [] }
        case 'color':
          return { ...task, colors: [{ textColor: '#000000', bgColor: '#ffffff' }] }
        case 'translate':
          return { ...task, translatedTexts: ['译文'], textboxTexts: [''], warnings: [] }
        case 'inpaint':
          return { ...task, cleanImage: 'clean-image' }
        case 'render':
          return { ...task, finalImage: 'rendered-image', bubbleStates: [] }
        case 'save':
          return { ...task, persisted: true }
        default:
          return task
      }
    })
  })

  it('marks the image completed after save succeeds in sequential mode', async () => {
    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')

    const { useSequentialPipeline } = await import('@/composables/translation/core/SequentialPipeline')
    const pipeline = useSequentialPipeline()

    const result = await pipeline.execute({
      mode: 'standard',
      scope: 'current',
    })

    expect(result.success).toBe(true)
    expect(imageStore.images[0]?.translationStatus).toBe('completed')
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
  })
})
