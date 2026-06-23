import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'

const {
  executePipelineMock,
  cancelPipelineMock,
} = vi.hoisted(() => ({
  executePipelineMock: vi.fn(),
  cancelPipelineMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/pipeline', () => ({
  usePipeline: () => ({
    progress: { value: {} },
    isExecuting: { value: false },
    isTranslating: { value: false },
    progressPercent: { value: 0 },
    execute: executePipelineMock,
    cancel: cancelPipelineMock,
  }),
}))

describe('useTranslationPipeline', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executePipelineMock.mockReset()
    cancelPipelineMock.mockReset()
  })

  it('restores the current image index when single-page translation throws', async () => {
    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,one')
    imageStore.addImage('page-2.png', 'data:image/png;base64,two')
    imageStore.setCurrentImageIndex(0)

    executePipelineMock.mockRejectedValueOnce(new Error('pipeline failed'))

    const { useTranslation } = await import('@/composables/useTranslationPipeline')
    const translation = useTranslation()

    await expect(
      translation.translatePages([1], 'standard')
    ).rejects.toThrow('pipeline failed')

    expect(imageStore.currentImageIndex).toBe(0)
  })
})
