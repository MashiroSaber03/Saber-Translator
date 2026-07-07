import {
  loadSessionMeta,
  savePageImage,
  savePageMeta,
  saveSessionMeta,
} from '@/api/pageStorage'
import { apiClient } from '@/api/client'
import type { BubbleState, BubbleTextline } from '@/types/bubble'
import type { OcrResult } from '@/types/ocr'
import { extractBase64Payload, readBlobAsDataUrl } from '@/utils/dataUrl'
import type { PipelineRuntime, TaskContext } from './runtime'
import { resolveTaskStyleFields } from './taskStyleFields'

interface PersistPageOptions {
  includeOriginal?: boolean
  includeDerivedImagesFromSource?: boolean
}

interface PersistAllPagesOptions extends PersistPageOptions {
  onProgress?: (current: number, total: number) => void
  currentImageIndex?: number
}

function ensureSessionPath(runtime: PipelineRuntime): string {
  if (!runtime.sessionPath) {
    throw new Error('当前不在书架模式，无法保存到章节存档')
  }
  return runtime.sessionPath
}

async function dataUrlFromApiUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url)
    if (!response.ok) {
      return null
    }

    const blob = await response.blob()
    return await readBlobAsDataUrl(blob)
  } catch {
    return null
  }
}

async function resolvePersistableBase64(
  source: string | null | undefined,
  fetchApiUrl: boolean
): Promise<string | null> {
  if (!source || typeof source !== 'string') return null
  if (source.startsWith('/api/')) {
    if (!fetchApiUrl) {
      return null
    }
    const dataUrl = await dataUrlFromApiUrl(source)
    if (!dataUrl) {
      return null
    }
    return resolvePersistableBase64(dataUrl, false)
  }
  if (source.startsWith('data:')) {
    return extractBase64Payload(source) || null
  }
  return source
}

function buildUiSettings(runtime: PipelineRuntime): Record<string, unknown> {
  const { textStyle } = runtime.settingsSnapshot
  return {
    fontSize: textStyle.fontSize,
    autoFontSize: textStyle.autoFontSize,
    fontFamily: textStyle.fontFamily,
    layoutDirection: textStyle.layoutDirection,
    textColor: textStyle.textColor,
    useInpaintingMethod: textStyle.inpaintMethod,
    fillColor: textStyle.fillColor,
    strokeEnabled: textStyle.strokeEnabled,
    strokeColor: textStyle.strokeColor,
    strokeWidth: textStyle.strokeWidth,
    lineSpacing: textStyle.lineSpacing,
    textAlign: textStyle.textAlign,
    useAutoTextColor: textStyle.useAutoTextColor,
  }
}

function bubbleValueOrFallback<T>(
  bubbleStates: BubbleState[] | null | undefined,
  index: number,
  pick: (bubble: BubbleState) => T | undefined,
  fallback: T[]
): T | undefined {
  if (Array.isArray(bubbleStates)) {
    const bubble = bubbleStates[index]
    if (bubble) {
      const value = pick(bubble)
      if (value !== undefined) {
        return value
      }
    }
  }
  return fallback[index]
}

function buildPageMeta(context: TaskContext, runtime: PipelineRuntime): Record<string, unknown> {
  const image = context.sourceImage
  const bubbleStates: BubbleState[] | null | undefined = Array.isArray(context.bubbleStates) ? context.bubbleStates : image.bubbleStates
  const hasRenderedResult = Boolean(context.finalImage || context.cleanImage)
  const translationStatus = context.status === 'failed'
    ? 'failed'
    : hasRenderedResult
      ? 'completed'
      : (image.translationStatus || 'pending')
  const translationFailed = context.status === 'failed'
    ? true
    : hasRenderedResult
      ? false
      : Boolean(image.translationFailed)
  return {
    fileName: image.fileName,
    translationStatus,
    translationFailed,
    bubbleStates: bubbleStates ?? null,
    bubbleCoords: Array.from({ length: Math.max(context.bubbleCoords.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback(bubbleStates, index, (bubble) => bubble.coords, context.bubbleCoords),
    ),
    bubbleAngles: Array.from({ length: Math.max(context.bubbleAngles.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback<number>(bubbleStates, index, (bubble) => bubble.rotationAngle || 0, context.bubbleAngles) ?? 0,
    ),
    originalTexts: Array.from({ length: Math.max(context.originalTexts.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback<string>(bubbleStates, index, (bubble) => bubble.originalText || '', context.originalTexts) || '',
    ),
    bubbleTexts: Array.from({ length: Math.max(context.translatedTexts.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback<string>(bubbleStates, index, (bubble) => bubble.translatedText || '', context.translatedTexts) || '',
    ),
    textboxTexts: Array.from({ length: Math.max(context.textboxTexts.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback<string>(bubbleStates, index, (bubble) => bubble.textboxText || '', context.textboxTexts) || '',
    ),
    textlinesPerBubble: Array.from({ length: Math.max(context.textlinesPerBubble.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback<BubbleTextline[]>(bubbleStates, index, (bubble) => bubble.textlines || [], context.textlinesPerBubble) || [],
    ),
    ocrResults: Array.from({ length: Math.max(context.ocrResults.length, Array.isArray(bubbleStates) ? bubbleStates.length : 0) }, (_, index) =>
      bubbleValueOrFallback<OcrResult>(bubbleStates, index, (bubble) => bubble.ocrResult || {
          text: bubble?.originalText || '',
          confidence: null,
          confidenceSupported: false,
          engine: '',
          primaryEngine: '',
          fallbackUsed: false,
        }, context.ocrResults) || {
          text: '',
          confidence: null,
          confidenceSupported: false,
          engine: '',
          primaryEngine: '',
          fallbackUsed: false,
        },
    ),
    isManuallyAnnotated: image.isManuallyAnnotated,
    relativePath: image.relativePath,
    folderPath: image.folderPath,
    hasUnsavedChanges: false,
    textMask: context.textMask ?? image.textMask ?? null,
    userMask: image.userMask ?? null,
    ...resolveTaskStyleFields(context, runtime),
  }
}

export async function persistPage(
  context: TaskContext,
  runtime: PipelineRuntime,
  options: PersistPageOptions = {}
): Promise<TaskContext> {
  const sessionPath = ensureSessionPath(runtime)
  const { includeOriginal = false, includeDerivedImagesFromSource = false } = options

  const originalBase64 = includeOriginal
    ? await resolvePersistableBase64(context.sourceImage.originalDataURL, true)
    : null
  let translatedBase64 = context.finalImage
    ? await resolvePersistableBase64(context.finalImage, includeDerivedImagesFromSource)
    : null
  if (!translatedBase64 && includeDerivedImagesFromSource) {
    translatedBase64 = await resolvePersistableBase64(context.sourceImage.translatedDataURL, true)
  }

  let cleanBase64 = context.cleanImage
    ? await resolvePersistableBase64(context.cleanImage, includeDerivedImagesFromSource)
    : null
  if (!cleanBase64 && includeDerivedImagesFromSource) {
    cleanBase64 = await resolvePersistableBase64(context.sourceImage.cleanImageData, true)
  }

  if (includeOriginal && originalBase64) {
    const result = await savePageImage(sessionPath, context.imageIndex, 'original', originalBase64)
    if (!result.success) {
      throw new Error(result.error || '保存原图失败')
    }
  }
  if (translatedBase64) {
    const result = await savePageImage(sessionPath, context.imageIndex, 'translated', translatedBase64)
    if (!result.success) {
      throw new Error(result.error || '保存译图失败')
    }
  }
  if (cleanBase64) {
    const result = await savePageImage(sessionPath, context.imageIndex, 'clean', cleanBase64)
    if (!result.success) {
      throw new Error(result.error || '保存干净背景失败')
    }
  }

  const metaResult = await savePageMeta(sessionPath, context.imageIndex, buildPageMeta(context, runtime))
  if (!metaResult.success) {
    throw new Error(metaResult.error || '保存页面元数据失败')
  }

  return {
    ...context,
    persisted: true,
  }
}

export async function persistAllPages(
  contexts: TaskContext[],
  runtime: PipelineRuntime,
  options: PersistAllPagesOptions = {}
): Promise<TaskContext[]> {
  const { onProgress, currentImageIndex = 0, ...persistOptions } = options
  const persistedContexts: TaskContext[] = []
  const total = contexts.length

  for (let index = 0; index < contexts.length; index++) {
    const context = contexts[index]
    if (!context) continue
    const persistedContext = await persistPage(context, runtime, persistOptions)
    persistedContexts.push(persistedContext)
    onProgress?.(index + 1, total)
  }

  await persistSessionMeta(runtime, { totalPages: contexts.length, currentImageIndex })
  return persistedContexts
}

export async function persistSessionMeta(
  runtime: PipelineRuntime,
  options: { totalPages: number; currentImageIndex: number }
): Promise<void> {
  const sessionPath = ensureSessionPath(runtime)
  const result = await saveSessionMeta(sessionPath, {
    ui_settings: buildUiSettings(runtime),
    total_pages: options.totalPages,
    currentImageIndex: options.currentImageIndex,
  })
  if (!result.success) {
    throw new Error(result.error || '保存会话元数据失败')
  }

  if (runtime.bookId && runtime.chapterId) {
    try {
      await apiClient.put(`/api/bookshelf/books/${runtime.bookId}/chapters/${runtime.chapterId}/image-count`, {
        count: options.totalPages,
      })
    } catch {
      // Chapter image-count sync is best-effort after page/session data has been persisted.
    }
  }
}

export async function isSessionInitialized(runtime: PipelineRuntime): Promise<boolean> {
  if (!runtime.sessionPath) {
    return false
  }
  try {
    const result = await loadSessionMeta(runtime.sessionPath)
    return Boolean(result.success && result.data)
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'status' in error && (error as { status?: unknown }).status === 404) {
      return false
    }
    throw error
  }
}
