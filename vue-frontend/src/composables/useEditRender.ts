import { getCurrentInstance, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { getPageRenderStatus, type V2PageRenderStatus } from '@/api/v2/content'
import {
  flushPageDocument,
  queuePageDocumentSave,
} from '@/services/pageDocumentPersistence'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'

export interface EditRenderCallbacks {
  onRenderError?: (error: string) => void
}

export function useEditRender(callbacks?: EditRenderCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const { bubbles } = storeToRefs(bubbleStore)
  const { currentImage } = storeToRefs(imageStore)

  let currentRenderToken: symbol | null = null
  let currentRenderController: AbortController | null = null
  let isOwnerDisposed = false

  if (getCurrentInstance()) {
    onUnmounted(() => {
      isOwnerDisposed = true
      currentRenderController?.abort()
      currentRenderController = null
      currentRenderToken = null
    })
  }

  function applyRenderStatus(pageId: string, status: V2PageRenderStatus): void {
    const index = imageStore.images.findIndex(image => image.id === pageId)
    if (index < 0) return
    const failed = status.renderStatus === 'render_failed' || status.renderStatus === 'repair_failed'
    const translationStatus = failed
      ? 'failed'
      : status.renderStatus === 'ready'
        ? 'completed'
        : status.renderStatus === 'not_rendered'
          ? 'pending'
          : 'processing'
    imageStore.updateImageByIndex(index, {
      documentRevision: status.documentRevision,
      renderedRevision: status.renderedRevision,
      translatedAssetUrl: status.translatedUrl,
      translationStatus,
    })
  }

  async function refreshUntilRendered(
    pageId: string,
    minimumRevision: number,
    token: symbol,
    signal: AbortSignal,
  ): Promise<void> {
    const deadline = Date.now() + 30_000
    while (!signal.aborted && !isOwnerDisposed && currentRenderToken === token) {
      const status = await getPageRenderStatus(pageId, signal)
      if (isOwnerDisposed || currentRenderToken !== token) return
      if (status.pageId !== pageId) {
        throw new Error(`页面 ${pageId} 的渲染状态身份不匹配`)
      }
      applyRenderStatus(pageId, status)
      if (
        (status.renderedRevision ?? 0) >= minimumRevision
        && status.renderStatus === 'ready'
      ) {
        return
      }
      if (status.renderStatus === 'render_failed' || status.renderStatus === 'repair_failed') {
        throw new Error('后端渲染失败')
      }
      if (
        status.renderStatus === 'not_rendered'
        && !bubbles.value.some(bubble => bubble.translatedText.trim())
      ) {
        return
      }
      if (Date.now() >= deadline) {
        throw new Error('后端渲染仍在继续，可稍后刷新查看结果')
      }
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }

  async function reRenderFullImage(): Promise<boolean> {
    const image = currentImage.value
    if (
      !image
      || !image.chapterId
      || image.documentRevision === undefined
    ) {
      return false
    }
    const token = Symbol('backend-render')
    currentRenderController?.abort()
    const controller = new AbortController()
    currentRenderController = controller
    currentRenderToken = token

    try {
      await Promise.all([
        queuePageDocumentSave(image.id, image.documentRevision, bubbles.value),
        flushPageDocument(image.id),
      ])
      if (controller.signal.aborted || currentRenderToken !== token || isOwnerDisposed) return false
      const committed = imageStore.images.find(candidate => candidate.id === image.id)
      if (!committed || committed.documentRevision === undefined) {
        throw new Error('当前页文档版本不可用')
      }
      await refreshUntilRendered(
        image.id,
        committed.documentRevision,
        token,
        controller.signal,
      )
      if (currentRenderToken !== token || isOwnerDisposed) return false
      return true
    } catch (error) {
      if (currentRenderToken !== token || isOwnerDisposed) return false
      const message = error instanceof Error ? error.message : '后端渲染请求失败'
      callbacks?.onRenderError?.(message)
      return false
    } finally {
      if (currentRenderToken === token) {
        currentRenderController = null
        currentRenderToken = null
      }
    }
  }

  return {
    reRenderFullImage,
  }
}
