import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { defineComponent } from 'vue'
import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { TaskContext } from '@/composables/translation/core/runtime'

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

    executeAtomicStepMock.mockImplementation(async (step: string, task: TaskContext) => {
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

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    let result
    try {
      result = await pipeline.execute({
        mode: 'standard',
        scope: 'current',
      })
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(result.success).toBe(true)
    expect(imageStore.images[0]?.translationStatus).toBe('completed')
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
  })

  it('keeps sequential pipeline task mocks typed to the current task contract', () => {
    for (const file of [
      'tests/unit/sequentialPipeline.status.spec.ts',
      'tests/unit/sequentialPipeline.validation.spec.ts',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
    }
  })

  it('does not let a previous delayed finish close progress for a new run', async () => {
    vi.useFakeTimers()
    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')

    const { useSequentialPipeline } = await import('@/composables/translation/core/SequentialPipeline')
    const pipeline = useSequentialPipeline()

    const firstResult = await pipeline.execute({
      mode: 'standard',
      scope: 'current',
    })
    expect(firstResult.success).toBe(true)
    expect(pipeline.progress.value.isInProgress).toBe(true)

    let resumeDetection: (() => void) | undefined
    executeAtomicStepMock.mockImplementation(async (step: string, task: TaskContext) => {
      if (step === 'detection') {
        await new Promise<void>((resolve) => {
          resumeDetection = resolve
        })
        return { ...task, bubbleCoords: [[0, 0, 10, 10]], bubbleAngles: [0], autoDirections: ['vertical'], bubbleStates: [], textlinesPerBubble: [] }
      }
      if (step === 'ocr') return { ...task, originalTexts: ['原文'], ocrResults: [] }
      if (step === 'color') return { ...task, colors: [{ textColor: '#000000', bgColor: '#ffffff' }] }
      if (step === 'translate') return { ...task, translatedTexts: ['译文'], textboxTexts: [''], warnings: [] }
      if (step === 'inpaint') return { ...task, cleanImage: 'clean-image' }
      if (step === 'render') return { ...task, finalImage: 'rendered-image', bubbleStates: [] }
      if (step === 'save') return { ...task, persisted: true }
      return task
    })

    const secondRun = pipeline.execute({
      mode: 'standard',
      scope: 'current',
    })
    await Promise.resolve()

    expect(pipeline.progress.value.isInProgress).toBe(true)
    vi.advanceTimersByTime(1000)
    expect(pipeline.progress.value.isInProgress).toBe(true)

    resumeDetection?.()
    await secondRun
    await vi.runOnlyPendingTimersAsync()
    vi.useRealTimers()
  })

  it('clears the delayed finish timer when the owner unmounts', async () => {
    vi.useFakeTimers()
    try {
      const imageStore = useImageStore()
      imageStore.addImage('page-1.png', 'data:image/png;base64,orig')

      const { useSequentialPipeline } = await import('@/composables/translation/core/SequentialPipeline')
      let pipeline: ReturnType<typeof useSequentialPipeline> | undefined
      const Host = defineComponent({
        setup() {
          pipeline = useSequentialPipeline()
          return () => null
        },
      })

      const wrapper = mount(Host)
      const result = await pipeline?.execute({
        mode: 'standard',
        scope: 'current',
      })

      expect(result?.success).toBe(true)
      expect(pipeline?.progress.value.isInProgress).toBe(true)

      wrapper.unmount()
      vi.advanceTimersByTime(1000)

      expect(pipeline?.progress.value.isInProgress).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })
})
