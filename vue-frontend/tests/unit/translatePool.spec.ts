import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  executeAtomicStepMock,
  executeBatchAtomicStepMock,
  settingsStoreMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  executeBatchAtomicStepMock: vi.fn(),
  settingsStoreMock: {
    settings: {
      hqTranslation: {
        batchSize: 2,
      },
      proofreading: {
        rounds: [{ batchSize: 2 }],
      },
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
        lineSpacing: 1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        useAutoTextColor: false,
      },
    },
  },
}))

vi.mock('@/composables/translation/core/atomicSteps', () => ({
  executeAtomicStep: executeAtomicStepMock,
  executeBatchAtomicStep: executeBatchAtomicStepMock,
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

describe('TranslatePool', () => {
  beforeEach(() => {
    executeAtomicStepMock.mockReset()
    executeBatchAtomicStepMock.mockReset()
  })

  it('routes standard mode through the shared translate atomic step', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: any) => ({
      ...task,
      translatedTexts: ['译文'],
      warnings: [],
    }))

    const { TranslatePool } = await import('@/composables/translation/parallel/pools/TranslatePool')
    const pool = new TranslatePool(null, { updatePool: vi.fn() } as any)
    pool.setMode('standard', 1, null)

    const task = {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'standard',
      runtime: { mode: 'standard', settingsSnapshot: settingsStoreMock.settings },
      status: 'pending',
      originalTexts: ['原文'],
      sourceImage: { originalDataURL: 'data:image/png;base64,abc' },
      bubbleCoords: [],
      bubbleAngles: [],
      bubblePolygons: [],
      autoDirections: [],
      textlinesPerBubble: [],
      ocrResults: [],
      colors: [],
      translatedTexts: [],
      textboxTexts: [],
      warnings: [],
      persisted: false,
    } as any

    const result = await (pool as any).process(task)

    expect(executeAtomicStepMock).toHaveBeenCalledWith('translate', task, task.runtime)
    expect(result.translatedTexts).toEqual(['译文'])
  })

  it('buffers HQ tasks and flushes them through the shared aiTranslate batch step', async () => {
    executeBatchAtomicStepMock.mockImplementation(async (_step: string, tasks: any[]) =>
      tasks.map((task) => ({
        ...task,
        translatedTexts: ['批量译文'],
        warnings: [],
      })),
    )

    const nextPool = { enqueue: vi.fn() }
    const { TranslatePool } = await import('@/composables/translation/parallel/pools/TranslatePool')
    const pool = new TranslatePool(nextPool as any, { updatePool: vi.fn() } as any)
    pool.setMode('hq', 2, nextPool as any)

    const runtime = { mode: 'hq', settingsSnapshot: settingsStoreMock.settings }
    const firstTask = {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'hq',
      runtime,
      status: 'pending',
      originalTexts: ['原文1'],
      sourceImage: { originalDataURL: 'data:image/png;base64,abc1' },
      bubbleCoords: [],
      bubbleAngles: [],
      bubblePolygons: [],
      autoDirections: [],
      textlinesPerBubble: [],
      ocrResults: [],
      colors: [],
      translatedTexts: [],
      textboxTexts: [],
      warnings: [],
      persisted: false,
    } as any
    const secondTask = {
      ...firstTask,
      id: 'task-2',
      imageIndex: 1,
      originalTexts: ['原文2'],
      sourceImage: { originalDataURL: 'data:image/png;base64,abc2' },
    }

    const buffered = await (pool as any).process(firstTask)
    const flushed = await (pool as any).process(secondTask)

    expect(buffered.status).toBe('buffered')
    expect(executeBatchAtomicStepMock).toHaveBeenCalledWith('aiTranslate', [firstTask, secondTask], runtime)
    expect(nextPool.enqueue).toHaveBeenCalledTimes(2)
    expect(flushed.status).toBe('buffered')
  })
})
