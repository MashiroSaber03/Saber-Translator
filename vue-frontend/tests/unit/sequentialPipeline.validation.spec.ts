import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'

const {
  executeAtomicStepMock,
  validateBeforeTranslationMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  validateBeforeTranslationMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
  executeBatchAtomicStep: vi.fn(),
}))

vi.mock('@/composables/useValidation', () => ({
  useValidation: () => ({
    validateBeforeTranslation: validateBeforeTranslationMock,
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

describe('useSequentialPipeline validation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executeAtomicStepMock.mockReset()
    validateBeforeTranslationMock.mockReset()

    executeAtomicStepMock.mockImplementation(async (step: string, task: any) => {
      switch (step) {
        case 'detection':
          return {
            ...task,
            bubbleCoords: [[0, 0, 10, 10]],
            bubbleAngles: [0],
            bubblePolygons: [[]],
            autoDirections: ['vertical'],
            textlinesPerBubble: [[]],
            bubbleStates: [],
          }
        case 'ocr':
          return {
            ...task,
            originalTexts: ['原文'],
            ocrResults: [],
          }
        case 'inpaint':
          return {
            ...task,
            cleanImage: 'clean-image',
          }
        case 'render':
          return {
            ...task,
            finalImage: 'rendered-image',
            bubbleStates: [],
          }
        default:
          return task
      }
    })
  })

  it('skips OCR validation for removeText mode when removeTextWithOcr is disabled', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')
    settingsStore.settings.removeTextWithOcr = false

    validateBeforeTranslationMock.mockImplementation((type?: string) => type !== 'ocr')

    const { useSequentialPipeline } = await import('@/composables/translation/core/SequentialPipeline')
    const pipeline = useSequentialPipeline()
    const result = await pipeline.execute({
      mode: 'removeText',
      scope: 'current',
    })

    expect(result.success).toBe(true)
    expect(validateBeforeTranslationMock).not.toHaveBeenCalled()
  })

  it('requires OCR validation for removeText mode when removeTextWithOcr is enabled', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')
    settingsStore.settings.removeTextWithOcr = true

    validateBeforeTranslationMock.mockReturnValue(true)

    const { useSequentialPipeline } = await import('@/composables/translation/core/SequentialPipeline')
    const pipeline = useSequentialPipeline()
    const result = await pipeline.execute({
      mode: 'removeText',
      scope: 'current',
    })

    expect(result.success).toBe(true)
    expect(validateBeforeTranslationMock).toHaveBeenCalledWith('ocr')
  })
})
