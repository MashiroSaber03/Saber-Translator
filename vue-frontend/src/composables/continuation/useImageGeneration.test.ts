import { ref, type Ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useImageGeneration } from './useImageGeneration'
import type { PageContent } from '@/api/continuation'
import type { ContinuationState } from './useContinuationState'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(nextResolve => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

const {
  generateAllPageImagesMock,
  regeneratePageImageMock,
  savePagesMock,
  setContinuationReferenceTokensMock,
  waitForJobMock,
} = vi.hoisted(() => ({
  generateAllPageImagesMock: vi.fn(),
  regeneratePageImageMock: vi.fn(),
  savePagesMock: vi.fn(),
  setContinuationReferenceTokensMock: vi.fn(),
  waitForJobMock: vi.fn(),
}))

vi.mock('@/api/continuation', () => ({
  generateAllPageImages: generateAllPageImagesMock,
  regeneratePageImage: regeneratePageImageMock,
  savePages: savePagesMock,
  setContinuationReferenceTokens: setContinuationReferenceTokensMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => ({ waitForJob: waitForJobMock }),
}))

function page(pageNumber: number, finalPrompt = `prompt-${pageNumber}`): PageContent {
  return {
    page_number: pageNumber,
    continuity_text: `continuity-${pageNumber}`,
    story_text: `story-${pageNumber}`,
    dialogue_text: `dialogue-${pageNumber}`,
    characters: [],
    final_prompt: finalPrompt,
    image_url: '',
    previous_url: '',
    status: 'pending',
  }
}

function createState(pages: Ref<PageContent[]>): ContinuationState {
  return {
    isLoading: ref(false),
    isDataReady: ref(true),
    isSyncingAnalysis: ref(false),
    currentStep: ref(0),
    messageType: ref(''),
    errorMessage: ref(''),
    successMessage: ref(''),
    lastAnalysisSyncAt: ref(''),
    pageCount: ref(10),
    styleRefPages: ref(2),
    continuationDirection: ref(''),
    characters: ref([]),
    chapterScript: ref(null),
    pages,
    imageRefreshKey: ref(0),
    isGeneratingPages: ref(false),
    hasMoreCharacterForms: ref(false),
    isLoadingMoreCharacterForms: ref(false),
    initializeData: vi.fn().mockResolvedValue(undefined),
    syncAnalysisData: vi.fn(),
    loadMoreCharacterForms: vi.fn().mockResolvedValue(undefined),
    resetState: vi.fn(),
    showMessage: vi.fn(),
    getCharacterImageUrl: vi.fn(),
  }
}

describe('useImageGeneration backend job ownership', () => {
  beforeEach(() => {
    generateAllPageImagesMock.mockReset().mockResolvedValue('job-1')
    regeneratePageImageMock.mockReset()
    savePagesMock.mockReset().mockResolvedValue(undefined)
    setContinuationReferenceTokensMock.mockReset().mockResolvedValue(undefined)
    waitForJobMock.mockReset().mockResolvedValue({ status: 'completed' })
  })

  it('submits one durable batch job instead of generating pages in the browser loop', async () => {
    const pages = ref([page(1), page(2)])
    const state = createState(pages)
    const composable = useImageGeneration(ref('book-1'), state)

    await composable.batchGenerateImages(pages.value, ['asset-9', 'asset-10'])

    expect(savePagesMock).toHaveBeenCalledOnce()
    expect(setContinuationReferenceTokensMock).toHaveBeenCalledWith('book-1', [
      'asset-9',
      'asset-10',
    ])
    expect(generateAllPageImagesMock).toHaveBeenCalledOnce()
    expect(generateAllPageImagesMock).toHaveBeenCalledWith('book-1', [1, 2])
    expect(waitForJobMock).toHaveBeenCalledWith('job-1', {
      onProgress: expect.any(Function),
    })
    expect(state.initializeData).toHaveBeenCalledOnce()
    expect(state.showMessage).toHaveBeenCalledWith(
      expect.stringContaining('关闭浏览器也会继续运行'),
      'info'
    )
  })

  it('accepts complete story content even when the optional final prompt is empty', async () => {
    const pages = ref([page(1, '')])
    const state = createState(pages)

    await useImageGeneration(ref('book-1'), state).batchGenerateImages(pages.value)

    expect(generateAllPageImagesMock).toHaveBeenCalledWith('book-1', [1])
  })

  it('keeps small non-zero progress visible for large image jobs', async () => {
    const pending = deferred<{ status: string }>()
    waitForJobMock.mockImplementationOnce((
      _jobId: string,
      options: { onProgress?: (progress: Record<string, unknown>) => void },
    ) => {
      options.onProgress?.({ completedItems: 8, totalItems: 2702 })
      return pending.promise
    })
    const pages = ref([page(1)])
    const state = createState(pages)
    const composable = useImageGeneration(ref('book-1'), state)

    const generation = composable.batchGenerateImages(pages.value)
    await vi.waitFor(() => {
      expect(composable.generationProgress.value).toBeCloseTo(8 / 2702 * 100)
    })
    expect(composable.generationProgress.value).toBeGreaterThan(0)

    pending.resolve({ status: 'completed' })
    await generation
  })

  it('does not refresh stale UI state after the selected book changes', async () => {
    const pending = deferred<{ status: string }>()
    waitForJobMock.mockReturnValueOnce(pending.promise)
    const pages = ref([page(1)])
    const state = createState(pages)
    const bookId = ref('book-1')
    const composable = useImageGeneration(bookId, state)

    const generation = composable.batchGenerateImages(pages.value)
    await vi.waitFor(() => expect(waitForJobMock).toHaveBeenCalled())
    bookId.value = 'book-2'
    pending.resolve({ status: 'completed' })
    await generation

    expect(state.initializeData).not.toHaveBeenCalled()
    expect(state.showMessage).not.toHaveBeenCalledWith(
      expect.stringContaining('图片生成完成'),
      'success'
    )
    expect(composable.isGenerating.value).toBe(false)
  })
})
