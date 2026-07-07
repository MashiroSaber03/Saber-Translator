import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import type { PipelineRuntime } from '@/composables/translation/core/runtime'
import type { PipelineTask } from '@/composables/translation/parallel/types'
import type { ImageData } from '@/types/image'

const {
  executeAtomicStepMock,
  projectTaskContextMock,
  resultCollectorAddMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  projectTaskContextMock: vi.fn(),
  resultCollectorAddMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
}))

vi.mock('@/composables/translation/core/taskProjector', () => ({
  projectTaskContext: projectTaskContextMock,
}))

describe('RenderPool', () => {
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
      autoSaveEnabled: false,
      isBookshelfMode: false,
      sessionPath: null,
      bookId: null,
      chapterId: null,
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
      runtime: createRuntime(),
      status: 'processing',
      sourceImage: createSourceImage(),
      bubbleCoords: [[0, 0, 10, 10]],
      bubbleAngles: [0],
      bubblePolygons: [],
      autoDirections: ['vertical'],
      textlinesPerBubble: [],
      originalTexts: ['原文'],
      ocrResults: [],
      colors: [],
      translatedTexts: ['译文'],
      textboxTexts: [''],
      warnings: [],
      autoGlossaryStats: {
        added: 0,
        duplicates: 0,
        failedPages: 0,
      },
      cleanImage: 'clean-image',
      bubbleStates: [],
      persisted: false,
      ...overrides,
    }
  }

  it('keeps render pool tests on current task contracts without private-member casts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/renderPool.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('uses the shared render atomic step and completes immediately when save is disabled', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: PipelineTask) => ({
      ...task,
      finalImage: 'rendered-image',
      bubbleStates: [],
    }))

    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')
    class TestRenderPool extends RenderPool {
      run(task: PipelineTask): Promise<PipelineTask> {
        return this.process(task)
      }
    }

    const pool = new TestRenderPool(
      null,
      { incrementCompleted: vi.fn() },
      { add: resultCollectorAddMock },
    )

    const task = createTask()

    const result = await pool.run(task)

    expect(executeAtomicStepMock).toHaveBeenCalledWith('render', task, task.runtime)
    expect(projectTaskContextMock).toHaveBeenCalledWith(expect.objectContaining({
      finalImage: 'rendered-image',
      status: 'completed',
    }), task.runtime)
    expect(resultCollectorAddMock).toHaveBeenCalledWith(expect.objectContaining({
      finalImage: 'rendered-image',
      status: 'completed',
    }))
    expect(result.status).toBe('completed')
  })

  it('only projects preview and leaves completion to the save pool when save is enabled', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: PipelineTask) => ({
      ...task,
      finalImage: 'rendered-image',
      bubbleStates: [],
    }))

    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')
    class TestRenderPool extends RenderPool {
      run(task: PipelineTask): Promise<PipelineTask> {
        return this.process(task)
      }
    }

    const pool = new TestRenderPool(
      { enqueue: vi.fn() },
      { incrementCompleted: vi.fn() },
      { add: resultCollectorAddMock },
    )

    const task = createTask({
      id: 'task-2',
      imageIndex: 1,
      runtime: createRuntime({
        autoSaveEnabled: true,
        isBookshelfMode: true,
        sessionPath: 'bookshelf/book-1/chapters/chapter-1/session',
        bookId: 'book-1',
        chapterId: 'chapter-1',
      }),
      sourceImage: createSourceImage({ fileName: 'page-2.png' }),
    })

    const result = await pool.run(task)

    expect(projectTaskContextMock).toHaveBeenCalledTimes(1)
    expect(resultCollectorAddMock).not.toHaveBeenCalled()
    expect(result.status).toBe('processing')
  })

  it('does not update UI after the render pool has been cancelled', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: PipelineTask) => ({
      ...task,
      finalImage: 'rendered-image',
      bubbleStates: [],
    }))

    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')
    class TestRenderPool extends RenderPool {
      run(task: PipelineTask): Promise<PipelineTask> {
        return this.process(task)
      }
    }

    const pool = new TestRenderPool(
      null,
      { incrementCompleted: vi.fn() },
      { add: resultCollectorAddMock },
    )
    pool.cancel()

    const task = createTask({
      id: 'task-3',
      imageIndex: 2,
      sourceImage: createSourceImage({ fileName: 'page-3.png' }),
    })

    await pool.run(task)

    expect(projectTaskContextMock).not.toHaveBeenCalled()
    expect(resultCollectorAddMock).not.toHaveBeenCalled()
  })
})
