import type { V2PageDocument, V2PageSummary } from '@/api/v2/content'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { BubbleState } from '@/types/bubble'
import type { ImageDataLoadInput, TranslationStatus } from '@/types/image'

function fileNameFromPath(path: string): string {
  const normalized = path.replaceAll('\\', '/')
  return normalized.split('/').at(-1) || path
}

function translationStatus(page: V2PageSummary): TranslationStatus {
  if (page.renderStatus === 'failed') return 'failed'
  if (page.translatedUrl && page.renderedRevision === page.documentRevision) return 'completed'
  if (page.renderStatus === 'rendering' || page.detectionState === 'processing') return 'processing'
  return 'pending'
}

export function pageSummaryToImage(page: V2PageSummary): ImageDataLoadInput {
  const status = translationStatus(page)
  return {
    id: page.id,
    chapterId: page.chapterId,
    sourceRevision: page.sourceRevision,
    documentRevision: page.documentRevision,
    renderedRevision: page.renderedRevision,
    fileName: fileNameFromPath(page.logicalSourcePath),
    relativePath: page.logicalSourcePath,
    width: page.width ?? 0,
    height: page.height ?? 0,
    // Compatibility fields now contain immutable backend URLs, never Data URLs.
    originalDataURL: page.sourceUrl,
    translatedDataURL: page.translatedUrl ?? null,
    cleanImageData: page.cleanUrl ?? null,
    sourceAssetUrl: page.sourceUrl,
    thumbnailSourceUrl: page.thumbnailSourceUrl,
    translatedAssetUrl: page.translatedUrl ?? null,
    thumbnailTranslatedUrl: page.thumbnailTranslatedUrl ?? null,
    bubbleStates: null,
    translationStatus: status,
    translationFailed: status === 'failed',
    hasUnsavedChanges: false,
  }
}

export function pageDocumentToBubbles(document: V2PageDocument): BubbleState[] {
  return document.bubbles.map(bubble => createBubbleState({
    ...(bubble.payload as Partial<BubbleState>),
    backendBubbleId: bubble.bubbleId,
    fontFamily: (
      typeof bubble.payload.fontFamily === 'string'
        ? bubble.payload.fontFamily
        : bubble.fontId || undefined
    ),
  }))
}
