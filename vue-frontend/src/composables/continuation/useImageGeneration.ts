import { getCurrentInstance, onBeforeUnmount, ref, watch, type Ref } from 'vue'
import type { PageContent } from '@/api/continuation'
import * as continuationApi from '@/api/continuation'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { ContinuationState } from './useContinuationState'
import {
  hasUsableStoryContent,
  isUsableImagePrompt,
  normalizeImagePrompt,
} from './promptValidation'

interface ImageGenerationComposable {
  isGenerating: Ref<boolean>
  generationProgress: Ref<number>
  batchGenerateImages: (
    pages: PageContent[],
    initialStyleReferenceTokens?: string[]
  ) => Promise<void>
  regeneratePageImage: (pageNumber: number) => Promise<void>
}

function progressPercent(progress: Record<string, unknown>): number {
  const completed = Number(progress.completedItems ?? 0)
  const total = Number(progress.totalItems ?? 0)
  return total > 0 ? Math.min(100, completed / total * 100) : 5
}

export function useImageGeneration(
  bookId: Ref<string | undefined>,
  state: ContinuationState
): ImageGenerationComposable {
  const taskCenterStore = useTaskCenterStore()
  const isGenerating = ref(false)
  const generationProgress = ref(0)
  let requestId = 0
  let isMounted = true

  function isCurrent(id: number, activeBookId: string): boolean {
    return isMounted && requestId === id && bookId.value === activeBookId
  }

  async function batchGenerateImages(
    pages: PageContent[],
    initialStyleReferenceTokens?: string[]
  ): Promise<void> {
    const activeBookId = bookId.value
    if (!activeBookId || pages.length === 0 || isGenerating.value) return
    const id = ++requestId
    const pending = pages.filter(page => !page.image_url)
    if (pending.length === 0) {
      state.showMessage('所有页面图片均已生成', 'info')
      return
    }
    for (const page of pending) {
      page.final_prompt = normalizeImagePrompt(page.final_prompt)
      if (!hasUsableStoryContent(page) && !isUsableImagePrompt(page.final_prompt)) {
        state.showMessage(
          `第 ${page.page_number} 页剧情或最终提示词无效，请先完善后再生成`,
          'error'
        )
        return
      }
    }

    isGenerating.value = true
    generationProgress.value = 5
    try {
      await continuationApi.savePages(activeBookId, pages)
      if (initialStyleReferenceTokens?.length) {
        await continuationApi.setContinuationReferenceTokens(
          activeBookId,
          initialStyleReferenceTokens
        )
      }
      const jobId = await continuationApi.generateAllPageImages(
        activeBookId,
        pending.map(page => page.page_number)
      )
      state.showMessage('批量生图任务已进入任务中心，关闭浏览器也会继续运行', 'info')
      await taskCenterStore.waitForJob(jobId, {
        onProgress: progress => {
          if (isCurrent(id, activeBookId)) {
            generationProgress.value = progressPercent(progress as Record<string, unknown>)
          }
        },
      })
      if (!isCurrent(id, activeBookId)) return
      await state.initializeData()
      if (!isCurrent(id, activeBookId)) return
      state.showMessage(`图片生成完成 (${pending.length} 页)`, 'success')
    } catch (error) {
      if (isCurrent(id, activeBookId)) {
        state.showMessage(
          '批量生成失败: ' + (error instanceof Error ? error.message : '网络错误'),
          'error'
        )
      }
    } finally {
      if (isMounted && requestId === id) {
        isGenerating.value = false
        generationProgress.value = 0
      }
    }
  }

  async function regeneratePageImage(pageNumber: number): Promise<void> {
    const activeBookId = bookId.value
    const page = state.pages.value.find(item => item.page_number === pageNumber)
    if (!activeBookId || !page || isGenerating.value) return
    const id = ++requestId
    page.final_prompt = normalizeImagePrompt(page.final_prompt)
    if (!hasUsableStoryContent(page) && !isUsableImagePrompt(page.final_prompt)) {
      state.showMessage(`第 ${pageNumber} 页剧情或最终提示词无效`, 'error')
      return
    }
    isGenerating.value = true
    generationProgress.value = 5
    try {
      const jobId = await continuationApi.regeneratePageImage(activeBookId, pageNumber, page)
      state.showMessage('重新生图任务已进入任务中心', 'info')
      await taskCenterStore.waitForJob(jobId, {
        onProgress: progress => {
          if (isCurrent(id, activeBookId)) {
            generationProgress.value = progressPercent(progress as Record<string, unknown>)
          }
        },
      })
      if (!isCurrent(id, activeBookId)) return
      await state.initializeData()
      if (!isCurrent(id, activeBookId)) return
      state.showMessage(`第 ${pageNumber} 页图片已重新生成`, 'success')
    } catch (error) {
      if (isCurrent(id, activeBookId)) {
        state.showMessage(
          '重新生成失败: ' + (error instanceof Error ? error.message : '网络错误'),
          'error'
        )
      }
    } finally {
      if (isMounted && requestId === id) {
        isGenerating.value = false
        generationProgress.value = 0
      }
    }
  }

  watch(bookId, () => {
    requestId += 1
    isGenerating.value = false
    generationProgress.value = 0
  })

  if (getCurrentInstance()) {
    onBeforeUnmount(() => {
      isMounted = false
      requestId += 1
    })
  }

  return {
    isGenerating,
    generationProgress,
    batchGenerateImages,
    regeneratePageImage,
  }
}
