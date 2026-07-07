import { ref, type Ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useImageGeneration } from './useImageGeneration'
import type { PageContent } from '@/api/continuation'
import type * as ContinuationApi from '@/api/continuation'
import type { ContinuationState } from './useContinuationState'

type GeneratePageImageArgs = Parameters<typeof ContinuationApi.generatePageImage>

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

const {
  getAvailableImagesMock,
  generatePageImageMock,
  savePagesMock,
} = vi.hoisted(() => ({
  getAvailableImagesMock: vi.fn(),
  generatePageImageMock: vi.fn(),
  savePagesMock: vi.fn(),
}))

vi.mock('@/api/continuation', () => ({
  getAvailableImages: getAvailableImagesMock,
  generatePageImage: generatePageImageMock,
  savePages: savePagesMock,
  getStyleReferences: vi.fn(),
  regeneratePageImage: vi.fn(),
}))

describe('useImageGeneration', () => {
  beforeEach(() => {
    getAvailableImagesMock.mockReset()
    generatePageImageMock.mockReset()
    savePagesMock.mockReset()
  })

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
      initializeData: vi.fn(),
      syncAnalysisData: vi.fn(),
      resetState: vi.fn(),
      showMessage: vi.fn(),
      getCharacterImageUrl: vi.fn(),
      getFormImageUrl: vi.fn(),
      getGeneratedImageUrl: vi.fn(),
    }
  }

  it('maintains style reference tokens as the sliding window advances', async () => {
    getAvailableImagesMock.mockResolvedValue({
      success: true,
      original_images: [
        { page_number: 9, path: '/tmp/original-9.png', has_image: true, token: 'original:9' },
        { page_number: 10, path: '/tmp/original-10.png', has_image: true, token: 'original:10' },
      ],
      continuation_images: [],
      character_forms: [],
      total_original_pages: 10,
    })

    const styleRefSnapshots: string[][] = []
    generatePageImageMock.mockImplementation(async (...args: GeneratePageImageArgs) => {
      styleRefSnapshots.push([...(args[3] || [])])
      return {
        success: true,
        image_path: '/tmp/generated-page.png',
      }
    })
    savePagesMock.mockResolvedValue({ success: true })

    const pages = ref<PageContent[]>([
      {
        page_number: 1,
        continuity_text: '原作末页摘要',
        story_text: '第1页剧情',
        dialogue_text: '对白1',
        characters: [],
        final_prompt: 'prompt-1',
        image_url: '',
        previous_url: '',
        status: 'pending' as const,
      },
      {
        page_number: 2,
        continuity_text: '第1页剧情',
        story_text: '第2页剧情',
        dialogue_text: '对白2',
        characters: [],
        final_prompt: 'prompt-2',
        image_url: '',
        previous_url: '',
        status: 'pending' as const,
      },
    ])

    const state = createState(pages)

    const composable = useImageGeneration(ref('book-1'), state)
    await composable.batchGenerateImages(pages.value)

    expect(generatePageImageMock).toHaveBeenCalledTimes(2)
    expect(styleRefSnapshots[0] || []).toEqual(['original:9', 'original:10'])
    expect(styleRefSnapshots[1] || []).toEqual(['original:10', 'continuation:1'])
    expect(pages.value[0].image_url).toBe('/tmp/generated-page.png')
    expect(pages.value[1].image_url).toBe('/tmp/generated-page.png')
  })

  it('still generates pages when story content is complete even if final prompt is empty', async () => {
    getAvailableImagesMock.mockResolvedValue({
      success: true,
      original_images: [
        { page_number: 10, path: '/tmp/original-10.png', has_image: true, token: 'original:10' },
      ],
      continuation_images: [],
      character_forms: [],
      total_original_pages: 10,
    })
    generatePageImageMock.mockResolvedValue({
      success: true,
      image_path: '/tmp/generated-page.png',
    })
    savePagesMock.mockResolvedValue({ success: true })

    const pages = ref<PageContent[]>([
      {
        page_number: 1,
        continuity_text: '原作末页摘要',
        story_text: '第1页剧情',
        dialogue_text: '对白1',
        characters: [],
        final_prompt: '',
        image_url: '',
        previous_url: '',
        status: 'pending' as const,
      },
      {
        page_number: 2,
        continuity_text: '第1页剧情',
        story_text: '第2页剧情',
        dialogue_text: '对白2',
        characters: [],
        final_prompt: '上一页剧情：第1页剧情\n本页剧情：第2页剧情\n关键对白：对白2\n出场角色：男主\n风格约束：保持原作漫画线条、脸型、上色、页面密度和分镜节奏。',
        image_url: '',
        previous_url: '',
        status: 'pending' as const,
      },
    ])

    const state = createState(pages)

    const composable = useImageGeneration(ref('book-1'), state)
    await composable.batchGenerateImages(pages.value)

    expect(generatePageImageMock).toHaveBeenCalledTimes(2)
    expect(generatePageImageMock.mock.calls[0]?.[1]).toBe(1)
    expect(pages.value[1].image_url).toBe('/tmp/generated-page.png')
  })

  it('does not apply or save stale batch image results after the selected book changes', async () => {
    getAvailableImagesMock.mockResolvedValue({
      success: true,
      original_images: [
        { page_number: 10, path: '/tmp/original-10.png', has_image: true, token: 'original:10' },
      ],
      continuation_images: [],
      character_forms: [],
      total_original_pages: 10,
    })
    const pendingGeneration = deferred<{ success: boolean; image_path: string }>()
    generatePageImageMock.mockReturnValueOnce(pendingGeneration.promise)
    savePagesMock.mockResolvedValue({ success: true })

    const pages = ref<PageContent[]>([
      {
        page_number: 1,
        continuity_text: '原作末页摘要',
        story_text: '第1页剧情',
        dialogue_text: '对白1',
        characters: [],
        final_prompt: 'prompt-1',
        image_url: '',
        previous_url: '',
        status: 'pending' as const,
      },
    ])
    const bookId = ref('book-1')
    const state = createState(pages)

    const composable = useImageGeneration(bookId, state)
    const generation = composable.batchGenerateImages(pages.value)
    await vi.waitFor(() => {
      expect(generatePageImageMock).toHaveBeenCalled()
    })

    bookId.value = 'book-2'
    pendingGeneration.resolve({
      success: true,
      image_path: '/tmp/stale-generated-page.png',
    })
    await generation

    expect(pages.value[0].image_url).toBe('')
    expect(pages.value[0].status).toBe('pending')
    expect(savePagesMock).not.toHaveBeenCalled()
    expect(state.showMessage).not.toHaveBeenCalledWith(expect.stringContaining('图片生成完成'), 'success')
    expect(composable.isGenerating.value).toBe(false)
  })
})
