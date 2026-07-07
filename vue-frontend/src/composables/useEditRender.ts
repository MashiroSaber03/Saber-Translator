import { getCurrentInstance, onUnmounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { executeRender } from '@/composables/translation/core/steps'
import { buildEditRenderInput } from '@/composables/edit/editRenderRequest'

export interface EditRenderCallbacks {
  onRenderStart?: () => void
  onRenderSuccess?: (translatedDataURL: string) => void
  onRenderError?: (error: string) => void
  onRenderEnd?: () => void
}

export function useEditRender(callbacks?: EditRenderCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  const { bubbles } = storeToRefs(bubbleStore)
  const { currentImage } = storeToRefs(imageStore)

  const isRendering = ref(false)
  const renderError = ref('')

  let currentRenderToken: symbol | null = null
  let isOwnerDisposed = false

  function isSameCurrentImage(expectedImageId: string): boolean {
    return !isOwnerDisposed && currentImage.value?.id === expectedImageId
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      isOwnerDisposed = true
      currentRenderToken = null
      isRendering.value = false
    })
  }

  function getCleanImageBase64(): string | null {
    const image = currentImage.value
    if (!image) return null

    if (image.cleanImageData) {
      return image.cleanImageData
    }

    if (image.originalDataURL) {
      const base64Match = image.originalDataURL.match(/^data:image\/[^;]+;base64,(.+)$/)
      if (base64Match && base64Match[1]) {
        return base64Match[1]
      }
    }

    return null
  }

  async function reRenderFullImage(silentMode = false): Promise<boolean> {
    const image = currentImage.value
    if (!image) {
      return false
    }
    const expectedImageId = image.id

    if (bubbles.value.length === 0) {
      // 无气泡时仍展示当前干净背景图，保证笔刷处理结果可见。
      const cleanBase64 = getCleanImageBase64()
      if (cleanBase64) {
        if (!isSameCurrentImage(expectedImageId)) {
          return false
        }
        const translatedDataURL = `data:image/png;base64,${cleanBase64}`
        imageStore.updateCurrentImage({ translatedDataURL })
        if (!silentMode) callbacks?.onRenderSuccess?.(translatedDataURL)
      }
      return true
    }

    const cleanBase64 = getCleanImageBase64()

    if (!cleanBase64) {
      renderError.value = '缺少图像数据'
      if (!silentMode) callbacks?.onRenderError?.('缺少图像数据')
      return false
    }

    const renderToken = Symbol('render')
    currentRenderToken = renderToken

    isRendering.value = true
    renderError.value = ''
    if (!silentMode) callbacks?.onRenderStart?.()

    try {
      const bubbleStates = bubbles.value
      const response = await executeRender(buildEditRenderInput({
        imageIndex: imageStore.currentImageIndex,
        cleanImage: cleanBase64,
        bubbleStates,
        settings: settingsStore.settings,
      }))

      if (currentRenderToken !== renderToken) {
        return false
      }

      if (!isSameCurrentImage(expectedImageId)) {
        return false
      }

      if (response.finalImage) {
        const translatedDataURL = `data:image/png;base64,${response.finalImage}`
        imageStore.updateCurrentImage({
          translatedDataURL,
          bubbleStates: response.bubbleStates,
          hasUnsavedChanges: true,
        })

        if (!silentMode) callbacks?.onRenderSuccess?.(translatedDataURL)
        return true
      } else {
        const errorMsg = '渲染失败'
        renderError.value = errorMsg
        if (!silentMode) callbacks?.onRenderError?.(errorMsg)
        return false
      }
    } catch (error) {
      if (currentRenderToken !== renderToken) {
        return false
      }

      const errorMsg = error instanceof Error ? error.message : '渲染请求失败'
      renderError.value = errorMsg
      if (!silentMode) callbacks?.onRenderError?.(errorMsg)
      return false
    } finally {
      if (currentRenderToken === renderToken) {
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
    cancelRender
  }
}
