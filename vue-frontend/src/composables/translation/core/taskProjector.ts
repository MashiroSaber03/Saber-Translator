import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import type { PipelineRuntime, TaskContext } from './runtime'

interface ProjectTaskContextOptions {
  imageStore?: ReturnType<typeof useImageStore>
  bubbleStore?: ReturnType<typeof useBubbleStore>
  syncBubbleStore?: boolean
}

function ensureDataUrl(data: string | null | undefined): string | null {
  if (!data || typeof data !== 'string') return null
  if (data.startsWith('data:') || data.startsWith('/api/')) {
    return data
  }
  return `data:image/png;base64,${data}`
}

function buildStyleProjection(context: TaskContext, runtime: PipelineRuntime): Record<string, unknown> {
  const image = context.sourceImage
  const saved = runtime.savedTextStyles
  const { textStyle } = runtime.settingsSnapshot

  return {
    fontSize: image.fontSize ?? saved?.fontSize ?? textStyle.fontSize,
    autoFontSize: image.autoFontSize ?? saved?.autoFontSize ?? textStyle.autoFontSize,
    fontFamily: image.fontFamily ?? saved?.fontFamily ?? textStyle.fontFamily,
    layoutDirection: image.layoutDirection ?? saved?.layoutDirection ?? textStyle.layoutDirection,
    textColor: image.textColor ?? saved?.textColor ?? textStyle.textColor,
    fillColor: image.fillColor ?? saved?.fillColor ?? textStyle.fillColor,
    strokeEnabled: image.strokeEnabled ?? saved?.strokeEnabled ?? textStyle.strokeEnabled,
    strokeColor: image.strokeColor ?? saved?.strokeColor ?? textStyle.strokeColor,
    strokeWidth: image.strokeWidth ?? saved?.strokeWidth ?? textStyle.strokeWidth,
    lineSpacing: image.lineSpacing ?? saved?.lineSpacing ?? textStyle.lineSpacing,
    textAlign: image.textAlign ?? saved?.textAlign ?? textStyle.textAlign,
    inpaintMethod: image.inpaintMethod ?? saved?.inpaintMethod ?? textStyle.inpaintMethod,
    useAutoTextColor: image.useAutoTextColor ?? saved?.useAutoTextColor ?? textStyle.useAutoTextColor,
  }
}

export function projectTaskContext(
  context: TaskContext,
  runtime: PipelineRuntime,
  options: ProjectTaskContextOptions = {}
): void {
  const imageStore = options.imageStore ?? useImageStore()
  const bubbleStore = options.bubbleStore ?? useBubbleStore()

  const translatedDataURL = context.finalImage
    ? ensureDataUrl(context.finalImage)
    : context.cleanImage
      ? ensureDataUrl(context.cleanImage)
      : (context.sourceImage.translatedDataURL ?? null)
  const translationStatus = context.status === 'failed'
    ? 'failed'
    : context.status === 'completed'
      ? 'completed'
      : 'processing'

  imageStore.updateImageByIndex(context.imageIndex, {
    translatedDataURL,
    cleanImageData: context.cleanImage ?? context.sourceImage.cleanImageData ?? null,
    bubbleStates: context.bubbleStates ?? null,
    bubbleCoords: context.bubbleCoords,
    bubbleAngles: context.bubbleAngles,
    originalTexts: context.originalTexts,
    bubbleTexts: context.translatedTexts,
    textboxTexts: context.textboxTexts,
    textlinesPerBubble: context.textlinesPerBubble,
    ocrResults: context.ocrResults,
    translationWarnings: context.warnings,
    textMask: context.textMask ?? context.sourceImage.textMask ?? null,
    userMask: context.sourceImage.userMask ?? null,
    translationStatus,
    translationFailed: context.status === 'failed',
    showOriginal: false,
    hasUnsavedChanges: !context.persisted,
    ...buildStyleProjection(context, runtime),
  })

  if (options.syncBubbleStore === false || imageStore.currentImageIndex !== context.imageIndex) {
    return
  }

  if (Array.isArray(context.bubbleStates)) {
    bubbleStore.setBubbles([...context.bubbleStates], true)
  } else {
    bubbleStore.clearBubblesLocal()
  }
}
