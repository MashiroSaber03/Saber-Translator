import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import { ParallelProgressTracker } from '@/composables/translation/parallel/ParallelProgressTracker'
import { TaskPool } from '@/composables/translation/parallel/TaskPool'
import type { PipelineRuntime } from '@/composables/translation/core/runtime'
import type { PipelineTask, ParallelTranslationMode } from '@/composables/translation/parallel/types'
import type { ImageData } from '@/types/image'

const {
  executeAtomicStepMock,
  executeBatchAtomicStepMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  executeBatchAtomicStepMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
  executeBatchAtomicStep: executeBatchAtomicStepMock,
}))

describe('TranslatePool', () => {
  beforeEach(() => {
    executeAtomicStepMock.mockReset()
    executeBatchAtomicStepMock.mockReset()
  })

  function createRuntime(
    mode: ParallelTranslationMode = 'standard',
    overrides: Partial<PipelineRuntime> = {},
  ): PipelineRuntime {
    const settings = createDefaultSettings()
    settings.hqTranslation.batchSize = 2
    settings.proofreading.rounds = [{
      name: 'Round 1',
      provider: settings.hqTranslation.provider,
      apiKey: settings.hqTranslation.apiKey,
      modelName: settings.hqTranslation.modelName,
      customBaseUrl: settings.hqTranslation.customBaseUrl,
      openaiOptions: settings.hqTranslation.openaiOptions,
      batchSize: 2,
      prompt: settings.hqTranslation.prompt,
    }]

    return {
      mode,
      settingsSnapshot: settings,
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
      originalDataURL: 'data:image/png;base64,abc',
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
      status: 'pending',
      sourceImage: createSourceImage(),
      bubbleCoords: [],
      bubbleAngles: [],
      bubblePolygons: [],
      autoDirections: [],
      textlinesPerBubble: [],
      originalTexts: ['原文'],
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
      persisted: false,
      ...overrides,
    }
  }

  class TestNextPool extends TaskPool {
    readonly enqueueMock = vi.fn()

    constructor() {
      super('Next', 'arrow-right', null, null, new ParallelProgressTracker())
    }

    override enqueue(task: PipelineTask): void {
      this.enqueueMock(task)
    }

    protected override async process(task: PipelineTask): Promise<PipelineTask> {
      return task
    }
  }

  async function createTestTranslatePool(
    nextPool: TaskPool | null,
    onTaskComplete?: (task: PipelineTask) => void,
  ) {
    const { TranslatePool } = await import('@/composables/translation/parallel/pools/TranslatePool')
    class TestTranslatePool extends TranslatePool {
      run(task: PipelineTask): Promise<PipelineTask> {
        return this.process(task)
      }
    }

    return new TestTranslatePool(nextPool, new ParallelProgressTracker(), onTaskComplete)
  }

  it('keeps translate pool tests on current task contracts without private-member casts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/translatePool.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('routes standard mode through the shared translate atomic step', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: PipelineTask) => ({
      ...task,
      translatedTexts: ['译文'],
      warnings: [],
    }))

    const pool = await createTestTranslatePool(null)
    pool.setMode('standard', 1, null)

    const task = createTask()

    const result = await pool.run(task)

    expect(executeAtomicStepMock).toHaveBeenCalledWith('translate', task, task.runtime)
    expect(result.translatedTexts).toEqual(['译文'])
  })

  it('buffers HQ tasks and flushes them through the shared aiTranslate batch step', async () => {
    executeBatchAtomicStepMock.mockImplementation(async (_step: string, tasks: PipelineTask[]) =>
      tasks.map((task) => ({
        ...task,
        translatedTexts: ['批量译文'],
        warnings: [],
      })),
    )

    const nextPool = new TestNextPool()
    const pool = await createTestTranslatePool(nextPool)
    pool.setMode('hq', 2, nextPool)

    const runtime = createRuntime('hq')
    const firstTask = createTask({
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'hq',
      runtime,
      originalTexts: ['原文1'],
      sourceImage: createSourceImage({
        id: 'img-1',
        fileName: 'page-1.png',
        originalDataURL: 'data:image/png;base64,abc1',
      }),
    })
    const secondTask = createTask({
      id: 'task-2',
      imageIndex: 1,
      translationMode: 'hq',
      runtime,
      originalTexts: ['原文2'],
      sourceImage: createSourceImage({
        id: 'img-2',
        fileName: 'page-2.png',
        originalDataURL: 'data:image/png;base64,abc2',
      }),
    })

    const buffered = await pool.run(firstTask)
    const flushed = await pool.run(secondTask)

    expect(buffered.status).toBe('buffered')
    expect(executeBatchAtomicStepMock).toHaveBeenCalledWith('aiTranslate', [firstTask, secondTask], runtime)
    expect(nextPool.enqueueMock).toHaveBeenCalledTimes(2)
    expect(flushed.status).toBe('buffered')
  })

  it('reports buffered HQ sibling tasks when a batch flush fails', async () => {
    executeBatchAtomicStepMock.mockRejectedValue(new Error('ai batch failed'))

    const nextPool = new TestNextPool()
    const onTaskComplete = vi.fn()
    const pool = await createTestTranslatePool(nextPool, onTaskComplete)
    pool.setMode('hq', 2, nextPool)

    const runtime = createRuntime('hq')
    const firstTask = createTask({
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'hq',
      runtime,
      originalTexts: ['原文1'],
      sourceImage: createSourceImage({
        id: 'img-1',
        fileName: 'page-1.png',
        originalDataURL: 'data:image/png;base64,abc1',
      }),
    })
    const secondTask = createTask({
      id: 'task-2',
      imageIndex: 1,
      translationMode: 'hq',
      runtime,
      originalTexts: ['原文2'],
      sourceImage: createSourceImage({
        id: 'img-2',
        fileName: 'page-2.png',
        originalDataURL: 'data:image/png;base64,abc2',
      }),
    })

    await pool.run(firstTask)
    await expect(pool.run(secondTask)).rejects.toThrow('ai batch failed')

    expect(onTaskComplete).toHaveBeenCalledWith(expect.objectContaining({
      id: 'task-1',
      status: 'failed',
      error: 'ai batch failed',
    }))
    expect(nextPool.enqueueMock).not.toHaveBeenCalled()
  })
})
