import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  executeAtomicStepMock,
  projectTaskContextMock,
  resultCollectorAddMock,
  settingsStoreMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  projectTaskContextMock: vi.fn(),
  resultCollectorAddMock: vi.fn(),
  settingsStoreMock: {
    settings: {
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
}))

vi.mock('@/composables/translation/core/taskProjector', () => ({
  projectTaskContext: projectTaskContextMock,
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

describe('RenderPool', () => {
  beforeEach(() => {
    executeAtomicStepMock.mockReset()
    projectTaskContextMock.mockReset()
    resultCollectorAddMock.mockReset()
  })

  it('uses the shared render atomic step and completes immediately when save is disabled', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: any) => ({
      ...task,
      finalImage: 'rendered-image',
      bubbleStates: [],
    }))

    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')

    const pool = new RenderPool(
      null,
      { incrementCompleted: vi.fn() } as any,
      { add: resultCollectorAddMock } as any,
    )

    const task = {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'standard',
      runtime: {
        mode: 'standard',
        settingsSnapshot: settingsStoreMock.settings,
        savedTextStyles: null,
        autoSaveEnabled: false,
        isBookshelfMode: false,
        sessionPath: null,
        bookId: null,
        chapterId: null,
      },
      status: 'processing',
      sourceImage: { fileName: 'page-1.png' },
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
      cleanImage: 'clean-image',
      bubbleStates: [],
      persisted: false,
    } as any

    const result = await (pool as any).process(task)

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
    executeAtomicStepMock.mockImplementation(async (_step: string, task: any) => ({
      ...task,
      finalImage: 'rendered-image',
      bubbleStates: [],
    }))

    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')

    const pool = new RenderPool(
      { enqueue: vi.fn() } as any,
      { incrementCompleted: vi.fn() } as any,
      { add: resultCollectorAddMock } as any,
    )

    const task = {
      id: 'task-2',
      imageIndex: 1,
      translationMode: 'standard',
      runtime: {
        mode: 'standard',
        settingsSnapshot: settingsStoreMock.settings,
        savedTextStyles: null,
        autoSaveEnabled: true,
        isBookshelfMode: true,
        sessionPath: 'bookshelf/book-1/chapters/chapter-1/session',
        bookId: 'book-1',
        chapterId: 'chapter-1',
      },
      status: 'processing',
      sourceImage: { fileName: 'page-2.png' },
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
      cleanImage: 'clean-image',
      bubbleStates: [],
      persisted: false,
    } as any

    const result = await (pool as any).process(task)

    expect(projectTaskContextMock).toHaveBeenCalledTimes(1)
    expect(resultCollectorAddMock).not.toHaveBeenCalled()
    expect(result.status).toBe('processing')
  })

  it('does not update UI after the render pool has been cancelled', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, task: any) => ({
      ...task,
      finalImage: 'rendered-image',
      bubbleStates: [],
    }))

    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')

    const pool = new RenderPool(
      null,
      { incrementCompleted: vi.fn() } as any,
      { add: resultCollectorAddMock } as any,
    )
    pool.cancel()

    const task = {
      id: 'task-3',
      imageIndex: 2,
      translationMode: 'standard',
      runtime: {
        mode: 'standard',
        settingsSnapshot: settingsStoreMock.settings,
        savedTextStyles: null,
        autoSaveEnabled: false,
        isBookshelfMode: false,
        sessionPath: null,
        bookId: null,
        chapterId: null,
      },
      status: 'processing',
      sourceImage: { fileName: 'page-3.png' },
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
      cleanImage: 'clean-image',
      bubbleStates: [],
      persisted: false,
    } as any

    await (pool as any).process(task)

    expect(projectTaskContextMock).not.toHaveBeenCalled()
    expect(resultCollectorAddMock).not.toHaveBeenCalled()
  })
})
