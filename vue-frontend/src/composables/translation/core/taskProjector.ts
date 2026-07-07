import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { toImageDataUrl } from '@/utils/dataUrl'
import type { PipelineRuntime, TaskContext } from './runtime'
import { resolveTaskStyleFields } from './taskStyleFields'

interface ProjectTaskContextOptions {
  imageStore?: ReturnType<typeof useImageStore>
  bubbleStore?: ReturnType<typeof useBubbleStore>
  syncBubbleStore?: boolean
}

export function projectTaskContext(
  context: TaskContext,
  runtime: PipelineRuntime,
  options: ProjectTaskContextOptions = {}
): void {
  const imageStore = options.imageStore ?? useImageStore()
  const bubbleStore = options.bubbleStore ?? useBubbleStore()

  const translatedDataURL = context.finalImage
    ? toImageDataUrl(context.finalImage)
    : context.cleanImage
      ? toImageDataUrl(context.cleanImage)
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
    ...resolveTaskStyleFields(context, runtime),
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
