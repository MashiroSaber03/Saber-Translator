import { getCurrentInstance, onUnmounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { getPageSummary } from '@/api/v2/content'
import { pageSummaryToImage } from '@/adapters/v2ContentAdapter'
import {
  queuePageDocumentSave,
} from '@/services/pageDocumentPersistence'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'

export interface EditRenderCallbacks {
  onRenderStart?: () => void
  onRenderSuccess?: (translatedUrl: string) => void
  onRenderError?: (error: string) => void
  onRenderEnd?: () => void
}

export function useEditRender(callbacks?: EditRenderCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const { bubbles } = storeToRefs(bubbleStore)
  const { currentImage } = storeToRefs(imageStore)

  const isRendering = ref(false)
  const renderError = ref('')
  let currentRenderToken: symbol | null = null
  let isOwnerDisposed = false

  if (getCurrentInstance()) {
    onUnmounted(() => {
      isOwnerDisposed = true
      currentRenderToken = null
      isRendering.value = false
    })
  }

  function applyPageSummary(pageId: string, summary: ReturnType<typeof pageSummaryToImage>): void {
    const index = imageStore.images.findIndex(image => image.id === pageId)
    if (index < 0) return
    imageStore.updateImageByIndex(index, {
      cleanAssetUrl: summary.cleanAssetUrl,
      documentRevision: summary.documentRevision,
      renderedRevision: summary.renderedRevision,
      sourceAssetUrl: summary.sourceAssetUrl,
      sourceRevision: summary.sourceRevision,
      thumbnailSourceUrl: summary.thumbnailSourceUrl,
      thumbnailTranslatedUrl: summary.thumbnailTranslatedUrl,
      translatedAssetUrl: summary.translatedAssetUrl,
      translationFailed: summary.translationFailed,
      translationStatus: summary.translationStatus,
    })
  }

  async function refreshUntilRendered(
    pageId: string,
    token: symbol,
  ): Promise<string | null> {
    const deadline = Date.now() + 30_000
    while (!isOwnerDisposed && currentRenderToken === token) {
      const page = await getPageSummary(pageId)
      applyPageSummary(pageId, pageSummaryToImage(page))
      if (
        page.renderedRevision === page.documentRevision
        && page.renderStatus === 'ready'
      ) {
        return page.translatedUrl ?? null
      }
      if (page.renderStatus === 'render_failed') {
        throw new Error('后端渲染失败')
      }
      if (
        page.renderStatus === 'not_rendered'
        && !bubbles.value.some(bubble => bubble.translatedText?.trim())
      ) {
        return page.translatedUrl ?? null
      }
      if (Date.now() >= deadline) {
        throw new Error('后端渲染仍在继续，可稍后刷新查看结果')
      }
      await new Promise(resolve => setTimeout(resolve, 300))
    }
    return null
  }

  async function reRenderFullImage(silentMode = false): Promise<boolean> {
    const image = currentImage.value
    if (
      !image
      || !image.chapterId
      || image.documentRevision === undefined
    ) {
      return false
    }
    const token = Symbol('backend-render')
    currentRenderToken = token
    isRendering.value = true
    renderError.value = ''
    if (!silentMode) callbacks?.onRenderStart?.()

    try {
      await queuePageDocumentSave(
        image.id,
        image.documentRevision,
        bubbles.value,
      )
      const url = await refreshUntilRendered(image.id, token)
      if (currentRenderToken !== token || isOwnerDisposed) return false
      if (!silentMode) callbacks?.onRenderSuccess?.(url ?? image.sourceAssetUrl)
      return true
    } catch (error) {
      if (currentRenderToken !== token || isOwnerDisposed) return false
      const message = error instanceof Error ? error.message : '后端渲染请求失败'
      renderError.value = message
      if (!silentMode) callbacks?.onRenderError?.(message)
      return false
    } finally {
      if (currentRenderToken === token) {
        isRendering.value = false
        if (!silentMode) callbacks?.onRenderEnd?.()
      }
    }
  }

  function cancelRender(): void {
    currentRenderToken = null
    isRendering.value = false
  }

  return {
    isRendering,
    renderError,
    reRenderFullImage,
    cancelRender,
  }
}
