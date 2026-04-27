import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  executeDetectionMock,
  settingsStoreMock,
  imageStoreMock,
} = vi.hoisted(() => ({
  executeDetectionMock: vi.fn(),
  settingsStoreMock: {
    settings: {
      removeTextWithOcr: false,
    },
  },
  imageStoreMock: {
    images: [] as any[],
    setTranslationStatus: vi.fn(),
    updateImageByIndex: vi.fn(),
    currentImageIndex: 0,
  },
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeDetection: executeDetectionMock,
  executeOcr: vi.fn(),
  executeColor: vi.fn(),
  executeTranslate: vi.fn(),
  executeAiTranslate: vi.fn(),
  executeInpaint: vi.fn(),
  executeRender: vi.fn(),
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

vi.mock('@/stores/imageStore', () => ({
  useImageStore: () => imageStoreMock,
}))

describe('ParallelPipeline failure handling', () => {
  beforeEach(() => {
    executeDetectionMock.mockReset()
    imageStoreMock.images = []
    imageStoreMock.setTranslationStatus.mockReset()
    imageStoreMock.updateImageByIndex.mockReset()
  })

  it('resolves with a failed result when an early pool throws instead of hanging forever', async () => {
    executeDetectionMock.mockRejectedValueOnce(new Error('detect exploded'))

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
    expect(imageStoreMock.setTranslationStatus).toHaveBeenCalledWith(0, 'failed', 'detect exploded')
  })

  it('resolves promptly when the parallel pipeline is cancelled mid-flight', async () => {
    executeDetectionMock.mockImplementationOnce(() => new Promise(() => {}))

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
