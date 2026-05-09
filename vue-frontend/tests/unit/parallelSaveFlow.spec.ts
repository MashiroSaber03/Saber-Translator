import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  executeAtomicStepMock,
  projectTaskContextMock,
  resultCollectorAddMock,
  progressSaveState,
  imageStoreMock,
  bubbleStoreMock,
  settingsStoreMock,
} = vi.hoisted(() => ({
  executeAtomicStepMock: vi.fn(),
  projectTaskContextMock: vi.fn(),
  resultCollectorAddMock: vi.fn(),
  progressSaveState: { value: { save: { completed: 0, total: 2 } } },
  imageStoreMock: {
    currentImageIndex: 0,
  },
  bubbleStoreMock: {},
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

vi.mock('@/stores/imageStore', () => ({
  useImageStore: () => imageStoreMock,
}))

vi.mock('@/stores/bubbleStore', () => ({
  useBubbleStore: () => bubbleStoreMock,
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

vi.mock('@/composables/translation/parallel/useParallelTranslation', () => ({
  useParallelTranslation: () => ({
    progress: progressSaveState,
  }),
}))

describe('parallel save flow', () => {
  beforeEach(() => {
    executeAtomicStepMock.mockReset()
    projectTaskContextMock.mockReset()
    resultCollectorAddMock.mockReset()
    progressSaveState.value.save.completed = 0
  })

  it('save pool marks tasks complete only after the shared save step succeeds', async () => {
    executeAtomicStepMock.mockImplementation(async (_step: string, context: any) => ({
      ...context,
      persisted: true,
      status: 'completed',
    }))

    const { SavePool } = await import('@/composables/translation/parallel/pools/SavePool')

    const pool = new SavePool(
      { incrementCompleted: vi.fn() } as any,
      { add: resultCollectorAddMock } as any,
    )

    const task = {
      id: 'task-1',
      imageIndex: 0,
      status: 'processing',
      persisted: false,
      runtime: { sessionPath: 'bookshelf/book-1/chapters/chapter-1/session' },
      sourceImage: { fileName: 'page-1.png' },
      finalImage: 'latest-image',
      cleanImage: 'latest-clean',
      bubbleStates: [],
    } as any

    const result = await (pool as any).process(task)

    expect(result.persisted).toBe(true)
    expect(executeAtomicStepMock).toHaveBeenCalledWith('save', task, task.runtime)
    expect(projectTaskContextMock).toHaveBeenCalledWith(result, task.runtime)
    expect(resultCollectorAddMock).toHaveBeenCalledWith(result)
    expect(progressSaveState.value.save.completed).toBe(1)
  })

  it('save pool does not report completion when the shared save step fails', async () => {
    executeAtomicStepMock.mockRejectedValue(new Error('disk full'))

    const { SavePool } = await import('@/composables/translation/parallel/pools/SavePool')

    const pool = new SavePool(
      { incrementCompleted: vi.fn() } as any,
      { add: resultCollectorAddMock } as any,
    )

    const task = {
      id: 'task-1',
      imageIndex: 0,
      status: 'processing',
      persisted: false,
      runtime: { sessionPath: 'bookshelf/book-1/chapters/chapter-1/session' },
      sourceImage: { fileName: 'page-1.png' },
      finalImage: 'latest-image',
      bubbleStates: [],
    } as any

    await expect((pool as any).process(task)).rejects.toThrow('disk full')
    expect(resultCollectorAddMock).not.toHaveBeenCalled()
    expect(progressSaveState.value.save.completed).toBe(0)
  })
})
