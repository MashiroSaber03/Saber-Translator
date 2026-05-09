import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  executeAtomicStepMock,
  settingsStoreMock,
  imageStoreMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  settingsStoreMock: {
    settings: {
      removeTextWithOcr: false,
      autoSaveInBookshelfMode: false,
      textStyle: {
        fontSize: 16,
        autoFontSize: false,
        fontFamily: 'fonts/STSONG.TTF',
        layoutDirection: 'auto',
        textColor: '#000000',
        fillColor: '#ffffff',
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        inpaintMethod: 'solid',
        useAutoTextColor: false,
        lineSpacing: 1,
        textAlign: 'start',
      },
      hqTranslation: { batchSize: 1 },
      proofreading: { rounds: [{ batchSize: 1 }] },
    },
  },
  imageStoreMock: {
    images: [] as any[],
    setTranslationStatus: vi.fn(),
    updateImageByIndex: vi.fn(),
    currentImageIndex: 0,
  },
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
  executeBatchAtomicStep: vi.fn(),
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

vi.mock('@/stores/imageStore', () => ({
  useImageStore: () => imageStoreMock,
}))

describe('ParallelPipeline failure handling', () => {
  beforeEach(() => {
    executeAtomicStepMock.mockReset()
    imageStoreMock.images = []
    imageStoreMock.setTranslationStatus.mockReset()
    imageStoreMock.updateImageByIndex.mockReset()
  })

  it('resolves with a failed result when an early pool throws instead of hanging forever', async () => {
    executeAtomicStepMock.mockRejectedValueOnce(new Error('detect exploded'))

    const { ParallelPipeline } = await import('@/composables/translation/parallel/ParallelPipeline')
    const pipeline = new ParallelPipeline({
      enabled: true,
      deepLearningLockSize: 1,
    })

    const image = {
      originalDataURL: 'data:image/png;base64,abc',
      translatedDataURL: null,
      cleanImageData: null,
      bubbleStates: null,
      userMask: null,
    } as any
    imageStoreMock.images = [{ translationStatus: 'pending' }]

    const result = await Promise.race([
      pipeline.execute([image], 'standard'),
      new Promise(resolve => setTimeout(() => resolve('timeout'), 50)),
    ])

    expect(result).not.toBe('timeout')
    expect(result).toEqual({
      success: 0,
      failed: 1,
      errors: ['detect exploded'],
    })
    expect(imageStoreMock.setTranslationStatus).toHaveBeenNthCalledWith(1, 0, 'processing')
    expect(imageStoreMock.setTranslationStatus).toHaveBeenNthCalledWith(2, 0, 'failed', 'detect exploded')
  })

  it('resolves promptly when the parallel pipeline is cancelled mid-flight', async () => {
    executeAtomicStepMock.mockImplementationOnce(() => new Promise(() => {}))

    const { ParallelPipeline } = await import('@/composables/translation/parallel/ParallelPipeline')
    const pipeline = new ParallelPipeline({
      enabled: true,
      deepLearningLockSize: 1,
    })

    const image = {
      originalDataURL: 'data:image/png;base64,abc',
      translatedDataURL: null,
      cleanImageData: null,
      bubbleStates: null,
      userMask: null,
    } as any
    imageStoreMock.images = [{ translationStatus: 'processing' }]

    const execution = pipeline.execute([image], 'standard')
    await new Promise(resolve => setTimeout(resolve, 0))
    pipeline.cancel()

    const result = await Promise.race([
      execution,
      new Promise(resolve => setTimeout(() => resolve('timeout'), 50)),
    ])

    expect(result).not.toBe('timeout')
    expect(result).toEqual({
      success: 0,
      failed: 0,
      errors: [],
    })
    expect(imageStoreMock.setTranslationStatus).toHaveBeenNthCalledWith(1, 0, 'processing')
    expect(imageStoreMock.setTranslationStatus).toHaveBeenNthCalledWith(2, 0, 'pending')
  })
})
