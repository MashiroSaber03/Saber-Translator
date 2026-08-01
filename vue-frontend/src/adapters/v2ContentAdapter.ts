import type { V2PageDocument, V2PageSummary } from '@/api/v2/content'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { BubbleState } from '@/types/bubble'
import type { ImageDataLoadInput, TranslationStatus } from '@/types/image'

function fileNameFromPath(path: string): string {
  const normalized = path.replaceAll('\\', '/')
  return normalized.split('/').at(-1) || path
}

function translationStatus(page: V2PageSummary): TranslationStatus {
  if (page.renderStatus === 'render_failed' || page.renderStatus === 'repair_failed') {
    return 'failed'
  }
  if (
    page.renderStatus === 'ready'
    && page.translatedUrl
    && page.renderedRevision === page.documentRevision
  ) return 'completed'
  if (page.renderStatus === 'rendering' || page.renderStatus === 'awaiting_repair') {
    return 'processing'
  }
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
    sourceAssetUrl: page.sourceUrl,
    cleanAssetUrl: page.cleanUrl ?? null,
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
  return document.bubbles.map(bubble => {
    const fontId = bubble.fontId || document.defaultFontId
    if (!fontId) {
      throw new Error(`页面 ${document.pageId} 的气泡 ${bubble.bubbleId} 缺少后端字体 ID`)
    }
    return createBubbleState({
      ...(bubble.payload as Partial<BubbleState>),
      backendBubbleId: bubble.bubbleId,
      fontFamily: fontId,
    })
  })
}
