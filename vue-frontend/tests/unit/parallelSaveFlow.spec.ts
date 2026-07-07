import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import type { PipelineRuntime } from '@/composables/translation/core/runtime'
import { ParallelProgressTracker } from '@/composables/translation/parallel/ParallelProgressTracker'
import type { PipelineTask } from '@/composables/translation/parallel/types'
import type { ImageData } from '@/types/image'

const {
  executeAtomicStepMock,
  projectTaskContextMock,
  resultCollectorAddMock,
  imageStoreMock,
  bubbleStoreMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  projectTaskContextMock: vi.fn(),
  resultCollectorAddMock: vi.fn(),
  imageStoreMock: {
    currentImageIndex: 0,
  },
  bubbleStoreMock: {},
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
}))

vi.mock('@/composables/translation/core/taskProjector', () => ({
  projectTaskContext: projectTaskContextMock,
}))

vi.mock('@/stores/imageStore', () => ({
  useImageStore: () => imageStoreMock,
}))

vi.mock('@/stores/bubbleStore', () => ({
  useBubbleStore: () => bubbleStoreMock,
}))

describe('parallel save flow', () => {
  beforeEach(() => {
    executeAtomicStepMock.mockReset()
    projectTaskContextMock.mockReset()
    resultCollectorAddMock.mockReset()
  })

  function createRuntime(overrides: Partial<PipelineRuntime> = {}): PipelineRuntime {
    return {
      mode: 'standard',
      settingsSnapshot: createDefaultSettings(),
      bookTranslationConstraints: createEmptyBookTranslationConstraints(),
      savedTextStyles: null,
      autoSaveEnabled: true,
      isBookshelfMode: true,
      sessionPath: 'bookshelf/book-1/chapters/chapter-1/session',
      bookId: 'book-1',
      chapterId: 'chapter-1',
      ...overrides,
    }
  }

  function createSourceImage(overrides: Partial<ImageData> = {}): ImageData {
    return {
      id: 'img-1',
      fileName: 'page-1.png',
      originalDataURL: 'data:image/png;base64,original',
      translatedDataURL: null,
      cleanImageData: null,
      bubbleStates: null,
      translationStatus: 'processing',
      translationFailed: false,
      hasUnsavedChanges: true,
      fontSize: 16,
      autoFontSize: false,
      fontFamily: 'fonts/STSONG.TTF',
      layoutDirection: 'auto',
      textColor: '#000000',
      fillColor: '#ffffff',
      inpaintMethod: 'solid',
      strokeEnabled: false,
      strokeColor: '#000000',
      strokeWidth: 1,
      lineSpacing: 1,
      textAlign: 'start',
      useAutoTextColor: false,
      ...overrides,
    }
  }

  function createTask(overrides: Partial<PipelineTask> = {}): PipelineTask {
    return {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'standard',
      status: 'processing',
      persisted: false,
      runtime: createRuntime(),
      sourceImage: createSourceImage(),
      bubbleCoords: [],
      bubbleAngles: [],
      bubblePolygons: [],
      autoDirections: [],
      textlinesPerBubble: [],
      originalTexts: [],
      ocrResults: [],
      colors: [],
      translatedTexts: [],
      textboxTexts: [],
      warnings: [],
      autoGlossaryStats: {
        added: 0,
        duplicates: 0,
        failedPages: 0,
      },
      finalImage: 'latest-image',
      cleanImage: 'latest-clean',
      bubbleStates: [],
      ...overrides,
    }
  }

  it('keeps save-flow pool tests on current task contracts without private-member casts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/parallelSaveFlow.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('save pool marks tasks complete only after the shared save step succeeds', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, context: PipelineTask) => ({
      ...context,
      persisted: true,
      status: 'completed',
    }))

    const { SavePool } = await import('@/composables/translation/parallel/pools/SavePool')
    class TestSavePool extends SavePool {
      run(task: PipelineTask): Promise<PipelineTask> {
        return this.process(task)
      }
    }

    const progressTracker = new ParallelProgressTracker()
    progressTracker.progress.save = { completed: 0, total: 2 }
    const pool = new TestSavePool(
      progressTracker,
      { add: resultCollectorAddMock },
    )

    const task = createTask()

    const result = await pool.run(task)

    expect(result.persisted).toBe(true)
    expect(executeAtomicStepMock).toHaveBeenCalledWith('save', task, task.runtime)
    expect(projectTaskContextMock).toHaveBeenCalledWith(result, task.runtime)
    expect(resultCollectorAddMock).toHaveBeenCalledWith(result)
    expect(progressTracker.progress.save?.completed).toBe(1)
  })

  it('save pool does not report completion when the shared save step fails', async () => {
    executeAtomicStepMock.mockRejectedValue(new Error('disk full'))

    const { SavePool } = await import('@/composables/translation/parallel/pools/SavePool')
    class TestSavePool extends SavePool {
      run(task: PipelineTask): Promise<PipelineTask> {
        return this.process(task)
      }
    }

    const progressTracker = new ParallelProgressTracker()
    progressTracker.progress.save = { completed: 0, total: 2 }
    const pool = new TestSavePool(
      progressTracker,
      { add: resultCollectorAddMock },
    )

    const task = createTask()

    await expect(pool.run(task)).rejects.toThrow('disk full')
    expect(resultCollectorAddMock).not.toHaveBeenCalled()
    expect(progressTracker.progress.save?.completed).toBe(0)
  })
})
