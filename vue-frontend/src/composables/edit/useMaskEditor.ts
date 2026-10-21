import { computed, onUnmounted, ref, shallowRef, watch } from 'vue'
import { getPageDocument, getPageRenderStatus, type V2PageDocument } from '@/api/v2/content'
import { getPageMasks, savePageUserMask, type PageMasks } from '@/api/v2/masks'
import { createMaskRepair, waitForOperation } from '@/api/v2/operations'
import { ApiClientError, isRequestCanceled } from '@/api/client'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import {
  flushPageDocument,
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { encodeGrayscaleMaskPng, paintMaskCircles } from '@/utils/binaryMaskPng'
import { showToast } from '@/utils/toast'
import type { BrushBounds, CurrentRepairSettings } from '@/composables/useBrush'
import { applyPageRenderStatus } from '@/composables/useEditRender'
import { maskOverlayClasses, maskOverlayPixels, maskVisibility, type MaskOverlay, type MaskStages } from '@/utils/maskPreview'

export type MaskTool = 'repair' | 'preserve' | 'clear'

function pngBlob(encoded: string): Blob {
  return new Blob([Uint8Array.from(atob(encoded), character => character.charCodeAt(0))], {
    type: 'image/png',
  })
}

async function decodeMask(encoded: string, width: number, height: number): Promise<Uint8Array> {
  const image = new Image()
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve()
    image.onerror = () => reject(new Error('修复范围加载失败'))
    image.src = `data:image/png;base64,${encoded}`
  })
  if (image.naturalWidth !== width || image.naturalHeight !== height)
    throw new Error('修复范围与原图尺寸不一致')
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) throw new Error('浏览器无法显示修复范围')
  context.drawImage(image, 0, 0)
  const rgba = context.getImageData(0, 0, width, height).data
  const pixels = new Uint8Array(width * height)
  for (let index = 0; index < pixels.length; index++) pixels[index] = rgba[index * 4]!
  return pixels
}

export function useMaskEditor(options: {
  getRepairSettings: () => CurrentRepairSettings
  onApplied: () => Promise<unknown>
}) {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const active = ref(false)
  const tool = ref<MaskTool>('repair')
  const visibility = ref(maskVisibility())
  const state = shallowRef<PageMasks | null>(null)
  const overlay = shallowRef<MaskOverlay | null>(null)
  const loading = ref(false)
  const saving = ref(false)
  const applying = ref(false)
  const dirty = ref(false)
  const error = ref('')
  const busy = computed(() => loading.value || saving.value || applying.value)
  let stages: MaskStages | null = null
  let manual: Uint8Array | null = null
  let overlayPixels: ImageData | null = null
  let overlayClasses: Uint8Array | null = null
  let controller = new AbortController()
  let requestSequence = 0

  function adopt(document: V2PageDocument): void {
    if (imageStore.currentImage?.id !== document.pageId) return
    const bubbles = registerPageDocument(document)
    imageStore.updateCurrentImage({
      documentRevision: document.documentRevision,
      bubbleStates: bubbles,
      hasUnsavedChanges: false,
    })
    bubbleStore.setBubbles(bubbles, true)
  }

  function drawOverlay(rebuild = false): void {
    if (!state.value || !stages || !manual) return
    const canvas = overlay.value?.canvas ?? document.createElement('canvas')
    if (canvas.width !== state.value.width) canvas.width = state.value.width
    if (canvas.height !== state.value.height) canvas.height = state.value.height
    const context = canvas.getContext('2d')
    if (!context) return
    if (!overlayPixels || overlayPixels.width !== canvas.width || overlayPixels.height !== canvas.height)
      overlayPixels = context.createImageData(canvas.width, canvas.height)
    if (rebuild || !overlayClasses)
      overlayClasses = maskOverlayClasses(canvas.width, canvas.height, stages, manual)
    maskOverlayPixels(overlayClasses, visibility.value, overlayPixels.data)
    context.putImageData(overlayPixels, 0, 0)
    // Updating the small wrapper notifies both viewports without copying/encoding the pixels.
    overlay.value = { canvas }
  }

  async function refresh(): Promise<void> {
    const image = imageStore.currentImage
    if (!active.value || !image || dirty.value) return
    if (state.value && state.value.pageId !== image.id) {
      state.value = null
      overlay.value = null
      stages = manual = null
    }
    const sequence = ++requestSequence
    loading.value = true
    error.value = ''
    try {
      await flushPageDocument(image.id)
      const result = await getPageMasks(
        image.id,
        options.getRepairSettings().inpaintMethod,
        controller.signal
      )
      if (!active.value || sequence !== requestSequence || imageStore.currentImage?.id !== image.id)
        return
      if (state.value && result.sourceRevision !== state.value.sourceRevision) {
        overlay.value = null
        throw new Error('原图已更换，请刷新页面后重新编辑修复范围')
      }
      const [raw, boxes, automatic, nextManual] = await Promise.all([
        result.preciseMask ? decodeMask(result.preciseMask, result.width, result.height) : null,
        decodeMask(result.boxMask, result.width, result.height),
        decodeMask(result.automaticMask, result.width, result.height),
        decodeMask(result.userMask, result.width, result.height),
      ])
      if (!active.value || sequence !== requestSequence || imageStore.currentImage?.id !== image.id)
        return
      const [document, renderStatus] = await Promise.all([
        imageStore.currentImage.documentRevision !== result.documentRevision
          ? getPageDocument(image.id, controller.signal) : null,
        getPageRenderStatus(image.id, controller.signal),
      ])
      if (!active.value || sequence !== requestSequence || imageStore.currentImage?.id !== image.id)
        return
      if (
        (document && (document.pageId !== image.id || document.documentRevision !== result.documentRevision))
        || renderStatus.pageId !== image.id
        || renderStatus.documentRevision !== result.documentRevision
        || (imageStore.currentImage.documentRevision ?? 0) > result.documentRevision
      ) {
        overlay.value = null
        throw new Error('页面已更新，请重新加载修复范围')
      }
      if (document) adopt(document)
      applyPageRenderStatus(renderStatus)
      state.value = result
      stages = { raw, boxes, automatic }
      manual = nextManual
      drawOverlay(true)
    } catch (cause) {
      if (sequence === requestSequence && !isRequestCanceled(cause)) {
        error.value = cause instanceof Error ? cause.message : '修复范围加载失败'
        showToast(error.value, 'error')
      }
    } finally {
      if (sequence === requestSequence) loading.value = false
    }
  }

  async function open(): Promise<void> {
    controller = new AbortController()
    active.value = true
    await refresh()
  }

  function close(): boolean {
    if (busy.value || dirty.value) {
      showToast('请先保存当前修复范围，或放弃未保存的修改', 'warning')
      return false
    }
    active.value = false
    controller.abort()
    requestSequence += 1
    state.value = null
    overlay.value = null
    overlayPixels = null
    overlayClasses = null
    stages = manual = null
    error.value = ''
    return true
  }

  async function save(): Promise<void> {
    const image = imageStore.currentImage
    if (!state.value || !manual || !image || image.documentRevision === undefined) return
    saving.value = true
    error.value = ''
    try {
      await queuePageDocumentSave(image.id, image.documentRevision, bubbleStore.bubbles)
      const revision = imageStore.currentImage?.documentRevision
      if (revision === undefined || imageStore.currentImage?.id !== image.id)
        throw new Error('当前页面已切换')
      const png = await encodeGrayscaleMaskPng(state.value.width, state.value.height, manual)
      const response = await savePageUserMask(image.id, revision, png, controller.signal)
      adopt(response.document)
      dirty.value = false
      await refresh()
    } catch (cause) {
      if (!isRequestCanceled(cause))
        error.value = cause instanceof Error ? cause.message : '修复范围保存失败'
      if (cause instanceof ApiClientError && cause.status === 409) {
        error.value = '页面已被其他操作更新，笔迹已保留。请重试保存，或放弃未保存修改。'
        // Keep the local stroke; the user's next retry uses the current revision.
        const latest = await getPageMasks(
          image.id,
          options.getRepairSettings().inpaintMethod,
          controller.signal
        )
        if (latest.sourceRevision !== state.value?.sourceRevision) {
          error.value = '原图已更换，请放弃未保存修改并刷新页面后重新绘制'
          overlay.value = null
          throw new Error(error.value)
        }
        adopt(await getPageDocument(image.id, controller.signal))
        throw new Error(error.value)
      }
      throw cause
    } finally {
      saving.value = false
    }
  }

  function stroke(bounds: BrushBounds): Promise<void> {
    if (!state.value || !manual || busy.value) return Promise.reject(new Error('修复范围尚未加载完成'))
    const changed = paintMaskCircles(
      manual,
      state.value.width,
      state.value.height,
      bounds.path.map(position => ({ x: position.x, y: position.y, radius: bounds.radius })),
      tool.value === 'repair' ? 255 : tool.value === 'preserve' ? 0 : 127
    )
    if (!changed) return dirty.value ? save() : Promise.resolve()
    dirty.value = true
    drawOverlay(true)
    return save()
  }

  async function retry(): Promise<void> {
    try {
      if (dirty.value) await save()
      else await refresh()
    } catch (cause) {
      if (!isRequestCanceled(cause)) showToast(error.value, 'error')
    }
  }

  async function discard(): Promise<void> {
    if (busy.value) return
    dirty.value = false
    await refresh()
  }

  async function apply(): Promise<void> {
    const image = imageStore.currentImage
    if (!image || !state.value || dirty.value || busy.value || error.value) return
    applying.value = true
    try {
      await flushPageDocument(image.id)
      if (imageStore.currentImage?.documentRevision !== state.value.documentRevision)
        await refresh()
      if (error.value || !state.value) throw new Error(error.value || '修复范围尚未加载完成')
      const settings = options.getRepairSettings()
      try {
        const accepted = await createMaskRepair(
          image.id,
          pngBlob(state.value.finalMask),
          settings.inpaintMethod === 'solid'
            ? {
                method: 'solid',
                fillColor: settings.fillColor,
                baseRevision: state.value.documentRevision,
                fromSource: true,
              }
            : {
                method: settings.inpaintMethod,
                baseRevision: state.value.documentRevision,
                fromSource: true,
              }
        )
        await waitForOperation(accepted.operationId, { signal: controller.signal })
      } finally {
        if (!controller.signal.aborted) {
          adopt(await getPageDocument(image.id, controller.signal))
          await refresh()
        }
      }
      await options.onApplied()
      showToast('已按当前范围完成背景修复', 'success')
    } catch (cause) {
      if (!isRequestCanceled(cause))
        showToast(cause instanceof Error ? cause.message : '背景修复失败', 'error')
    } finally {
      applying.value = false
    }
  }

  watch(visibility, () => drawOverlay(), { deep: true })
  watch(
    () => [
      imageStore.currentImage?.id,
      imageStore.currentImage?.documentRevision,
      options.getRepairSettings().inpaintMethod,
    ],
    (next, previous) => {
      const targetChanged = next[0] !== previous[0] || next[2] !== previous[2]
      if (
        active.value && !dirty.value
        && (targetChanged || next[1] !== state.value?.documentRevision)
        && (!busy.value || (loading.value && targetChanged))
      ) void refresh()
    }
  )
  onUnmounted(() => {
    controller.abort()
    requestSequence += 1
    stages = manual = null
    overlay.value = null
    overlayPixels = null
    overlayClasses = null
  })
  return {
    active,
    tool,
    visibility,
    state,
    overlay,
    busy,
    loading,
    saving,
    applying,
    dirty,
    error,
    open,
    close,
    stroke,
    apply,
    retry,
    discard,
  }
}
