import { getCurrentInstance, onUnmounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { getPageRenderStatus, type V2PageRenderStatus } from '@/api/v2/content'
import {
  queuePageDocumentSave,
} from '@/services/pageDocumentPersistence'
import { useImageStore } from '@/stores/imageStore'

export interface EditRenderCallbacks {
  onRenderError?: (error: string) => void
}

export function applyPageRenderStatus(status: V2PageRenderStatus): void {
  const imageStore = useImageStore()
  const index = imageStore.images.findIndex(image => image.id === status.pageId)
  const image = imageStore.images[index]
  if (
    !image || (image.documentRevision ?? 0) > status.documentRevision
    || (image.renderedRevision === status.documentRevision
      && status.renderedRevision !== status.documentRevision)
  ) return
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
    cleanAssetUrl: status.cleanUrl,
    translatedAssetUrl: status.translatedUrl,
    translationStatus,
  })
}

/** Observe an already submitted document. All editor entry points use this result path. */
export async function waitForPageRender(pageId: string, minimumRevision: number, signal: AbortSignal): Promise<void> {
  const deadline = Date.now() + 30_000
  let readFailures = 0
  while (!signal.aborted) {
    let status: V2PageRenderStatus | undefined
    try {
      status = await getPageRenderStatus(pageId, signal)
      readFailures = 0
    } catch (error) {
      if (signal.aborted) return
      if (++readFailures >= 5) throw error
    }
    if (signal.aborted) return
    if (status) {
      if (status.pageId !== pageId) throw new Error(`页面 ${pageId} 的渲染状态身份不匹配`)
      applyPageRenderStatus(status)
      if (status.documentRevision >= minimumRevision) {
        if (status.renderStatus === 'render_failed' || status.renderStatus === 'repair_failed') throw new Error('后端渲染失败')
        if (status.renderStatus === 'ready' && (status.renderedRevision ?? 0) >= minimumRevision) return
        const image = useImageStore().images.find(image => image.id === pageId)
        if (status.renderStatus === 'not_rendered' && !image?.bubbleStates?.some(bubble => bubble.translatedText.trim())) return
      }
    }
    if (Date.now() >= deadline) throw new Error('后端渲染仍在继续，可稍后刷新查看结果')
    await new Promise<void>(resolve => {
      const finish = () => { clearTimeout(timer); signal.removeEventListener('abort', finish); resolve() }
      const timer = setTimeout(finish, 500)
      signal.addEventListener('abort', finish, { once: true })
    })
  }
}

export function useEditRender(callbacks?: EditRenderCallbacks) {
  const imageStore = useImageStore()
  const { currentImage } = storeToRefs(imageStore)

  let currentRenderToken: symbol | null = null
  let currentRenderController: AbortController | null = null
  let isOwnerDisposed = false
  const isRendering = ref(false)
  const renderError = ref('')

  watch(() => currentImage.value?.id, () => {
    currentRenderController?.abort()
    currentRenderToken = null
    isRendering.value = false
    renderError.value = ''
  })

  if (getCurrentInstance()) {
    onUnmounted(() => {
      isOwnerDisposed = true
      currentRenderController?.abort()
      currentRenderController = null
      currentRenderToken = null
    })
  }

  async function refresh(persist: boolean): Promise<boolean> {
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
    isRendering.value = true
    renderError.value = ''

    try {
      // The persistence queue owns the one debounce window. A native save or
      // sidebar submission has already committed; observing it must not save again.
      if (persist) await queuePageDocumentSave(image.id, image.documentRevision, image.bubbleStates ?? [])
      if (controller.signal.aborted || currentRenderToken !== token || isOwnerDisposed) return false
      const committed = imageStore.images.find(candidate => candidate.id === image.id)
      if (!committed || committed.documentRevision === undefined) {
        throw new Error('当前页文档版本不可用')
      }
      await waitForPageRender(
        image.id,
        committed.documentRevision,
        controller.signal,
      )
      if (controller.signal.aborted || currentRenderToken !== token || isOwnerDisposed) return false
      return true
    } catch (error) {
      if (currentRenderToken !== token || isOwnerDisposed) return false
      const message = error instanceof Error ? error.message : '后端渲染请求失败'
      renderError.value = message
      callbacks?.onRenderError?.(message)
      return false
    } finally {
      if (currentRenderToken === token) {
        isRendering.value = false
        currentRenderController = null
        currentRenderToken = null
      }
    }
  }

  return {
    reRenderFullImage: () => refresh(true),
    refreshRenderedImage: () => refresh(false),
    isRendering,
    renderError,
  }
}
