import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'

const {
  executeParallelMock,
  validateBeforeTranslationMock,
  notifyPipelineBeforeMock,
  notifyPipelineAfterMock,
} = vi.hoisted(() => ({
  executeParallelMock: vi.fn(),
  validateBeforeTranslationMock: vi.fn(),
  notifyPipelineBeforeMock: vi.fn(),
  notifyPipelineAfterMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/SequentialPipeline', () => ({
  useSequentialPipeline: () => ({
    progress: { value: { percentage: 0 } },
    isExecuting: { value: false },
    isTranslating: { value: false },
    progressPercent: { value: 0 },
    execute: vi.fn(),
    cancel: vi.fn(),
    STEP_CHAIN_CONFIGS: {},
  }),
}))

vi.mock('@/composables/translation/parallel', () => ({
  useParallelTranslation: () => ({
    isRunning: { value: false },
    progress: { value: { pools: [], totalCompleted: 0, totalFailed: 0, totalPages: 0, estimatedTimeRemaining: 0 } },
    executeParallel: executeParallelMock,
    cancel: vi.fn(),
    reset: vi.fn(),
    determineMode: vi.fn(),
  }),
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

vi.mock('@/api/pipeline', () => ({
  notifyPipelineBefore: notifyPipelineBeforeMock,
  notifyPipelineAfter: notifyPipelineAfterMock,
  PipelineCancelledError: class PipelineCancelledError extends Error {},
}))

describe('usePipeline validation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executeParallelMock.mockReset()
    validateBeforeTranslationMock.mockReset()
    notifyPipelineBeforeMock.mockReset()
    notifyPipelineAfterMock.mockReset()

    executeParallelMock.mockResolvedValue({
      success: 1,
      failed: 0,
      errors: [],
    })
    notifyPipelineBeforeMock.mockResolvedValue(undefined)
    notifyPipelineAfterMock.mockResolvedValue(undefined)
  })

  it('skips OCR validation for parallel removeText runs when removeTextWithOcr is disabled', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')
    imageStore.addImage('page-2.png', 'data:image/png;base64,orig2')
    settingsStore.settings.parallel.enabled = true
    settingsStore.settings.removeTextWithOcr = false

    validateBeforeTranslationMock.mockImplementation((type?: string) => type !== 'ocr')

    const { usePipeline } = await import('@/composables/translation/core/pipeline')
    const pipeline = usePipeline()
    const result = await pipeline.execute({
      mode: 'removeText',
      scope: 'all',
    })

    expect(result.success).toBe(true)
    expect(validateBeforeTranslationMock).not.toHaveBeenCalled()
    expect(executeParallelMock).toHaveBeenCalledTimes(1)
  })

  it('validates OCR config before parallel removeText runs when removeTextWithOcr is enabled', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')
    imageStore.addImage('page-2.png', 'data:image/png;base64,orig2')
    settingsStore.settings.parallel.enabled = true
    settingsStore.settings.removeTextWithOcr = true

    validateBeforeTranslationMock.mockReturnValue(false)

    const { usePipeline } = await import('@/composables/translation/core/pipeline')
    const pipeline = usePipeline()
    const result = await pipeline.execute({
      mode: 'removeText',
      scope: 'all',
    })

    expect(result.success).toBe(false)
    expect(result.errors).toEqual(['配置验证失败'])
    expect(validateBeforeTranslationMock).toHaveBeenCalledWith('ocr')
    expect(executeParallelMock).not.toHaveBeenCalled()
  })
})
