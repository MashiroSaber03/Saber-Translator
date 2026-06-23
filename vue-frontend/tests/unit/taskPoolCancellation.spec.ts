import { describe, expect, it, vi, type Mock } from 'vitest'

import { DeepLearningLock } from '@/composables/translation/parallel/DeepLearningLock'
import { ParallelProgressTracker } from '@/composables/translation/parallel/ParallelProgressTracker'
import { TaskPool } from '@/composables/translation/parallel/TaskPool'
import type { PipelineTask } from '@/composables/translation/parallel/types'

class LockedTestPool extends TaskPool {
  constructor(
    lock: DeepLearningLock,
    tracker: ParallelProgressTracker,
    private readonly processSpy: Mock<[PipelineTask], PipelineTask>,
  ) {
    super('检测', 'test', null, lock, tracker)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    return this.processSpy(task)
  }
}

function createTask(): PipelineTask {
  return {
    id: 'task-1',
    imageIndex: 0,
    translationMode: 'standard',
    sourceImage: {
      id: 'image-1',
      fileName: 'page-1.png',
      originalDataURL: 'data:image/png;base64,abc',
      translatedDataURL: null,
      cleanImageData: null,
      bubbleStates: null,
      translationStatus: 'pending',
      translationFailed: false,
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
      hasUnsavedChanges: false,
    },
    status: 'pending',
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
    bubbleStates: null,
    persisted: false,
  }
}

describe('TaskPool cancellation', () => {
  it('does not process a task that was cancelled while waiting for the deep-learning lock', async () => {
    const lock = new DeepLearningLock(1)
    const tracker = new ParallelProgressTracker()
    const processSpy = vi.fn((task: PipelineTask) => ({ ...task, status: 'completed' as const }))
    const pool = new LockedTestPool(lock, tracker, processSpy)

    await lock.acquire('external')
    pool.enqueue(createTask())
    await Promise.resolve()

    expect(processSpy).not.toHaveBeenCalled()

    pool.cancel()
    lock.reset()
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(processSpy).not.toHaveBeenCalled()
  })
})
