import { useSessionStore } from '@/stores/sessionStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import type { BubbleCoords, BubbleState, BubbleTextline } from '@/types/bubble'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import type { ImageData as AppImageData } from '@/types/image'
import type { OcrResult } from '@/types/ocr'
import type { TranslationSettings } from '@/types/settings'
import type { TranslationWarning } from '@/types/translationConstraints'
import type { SavedTextStyles, TranslationMode } from './types'

export type TaskExecutionStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'buffered'

export interface PipelineRuntime {
  mode: TranslationMode
  settingsSnapshot: TranslationSettings
  bookTranslationConstraints: BookTranslationConstraints
  savedTextStyles: SavedTextStyles | null
  autoSaveEnabled: boolean
  isBookshelfMode: boolean
  sessionPath: string | null
  bookId: string | null
  chapterId: string | null
}

export interface TaskContext {
  id: string
  imageIndex: number
  translationMode: TranslationMode
  sourceImage: AppImageData
  status: TaskExecutionStatus
  error?: string
  runtime?: PipelineRuntime

  bubbleCoords: BubbleCoords[]
  bubbleAngles: number[]
  bubblePolygons: number[][][]
  autoDirections: string[]
  textMask?: string
  textlinesPerBubble: BubbleTextline[][]

  originalTexts: string[]
  ocrResults: OcrResult[]

  colors: Array<{
    textColor: string
    bgColor: string
    autoFgColor?: [number, number, number] | null
    autoBgColor?: [number, number, number] | null
  }>

  translatedTexts: string[]
  textboxTexts: string[]
  warnings: TranslationWarning[]

  cleanImage?: string
  finalImage?: string
  bubbleStates?: BubbleState[] | null
  persisted: boolean
}

function cloneDeep<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

export function buildSavedTextStylesFromSettings(settings: TranslationSettings): SavedTextStyles {
  const { textStyle } = settings
  const layoutDirectionValue = textStyle.layoutDirection
  return {
    fontFamily: textStyle.fontFamily,
    fontSize: textStyle.fontSize,
    autoFontSize: textStyle.autoFontSize,
    autoTextDirection: layoutDirectionValue === 'auto',
    textDirection: layoutDirectionValue === 'auto' ? 'vertical' : layoutDirectionValue,
    layoutDirection: layoutDirectionValue,
    fillColor: textStyle.fillColor,
    textColor: textStyle.textColor,
    rotationAngle: 0,
    strokeEnabled: textStyle.strokeEnabled,
    strokeColor: textStyle.strokeColor,
    strokeWidth: textStyle.strokeWidth,
    lineSpacing: textStyle.lineSpacing,
    textAlign: textStyle.textAlign,
    useAutoTextColor: textStyle.useAutoTextColor,
    inpaintMethod: textStyle.inpaintMethod,
  }
}

export function getBookshelfSessionPath(bookId: string | null, chapterId: string | null): string | null {
  if (!bookId || !chapterId) {
    return null
  }
  return `bookshelf/${bookId}/chapters/${chapterId}/session`
}

export function createPipelineRuntime(
  mode: TranslationMode,
  options?: {
    settingsSnapshot?: TranslationSettings
    bookTranslationConstraints?: BookTranslationConstraints
    savedTextStyles?: SavedTextStyles | null
    autoSaveEnabled?: boolean
    sessionPath?: string | null
    bookId?: string | null
    chapterId?: string | null
  }
): PipelineRuntime {
  let sessionStore: ReturnType<typeof useSessionStore> | null = null
  let settingsStore: ReturnType<typeof useSettingsStore> | null = null
  let bookTranslationConstraintsStore: ReturnType<typeof useBookTranslationConstraintsStore> | null = null

  try {
    sessionStore = useSessionStore()
  } catch {
    sessionStore = null
  }

  try {
    settingsStore = useSettingsStore()
  } catch {
    settingsStore = null
  }

  try {
    bookTranslationConstraintsStore = useBookTranslationConstraintsStore()
  } catch {
    bookTranslationConstraintsStore = null
  }

  const sourceSettings = options?.settingsSnapshot ?? settingsStore?.settings
  if (!sourceSettings) {
    throw new Error('创建 PipelineRuntime 失败：缺少设置快照')
  }

  const settingsSnapshot = cloneDeep(sourceSettings)
  const bookId = options?.bookId ?? sessionStore?.currentBookId ?? null
  const chapterId = options?.chapterId ?? sessionStore?.currentChapterId ?? null
  const sessionPath = options?.sessionPath ?? getBookshelfSessionPath(bookId, chapterId)
  const isBookshelfMode = Boolean(bookId && chapterId)
  const autoSaveEnabled = options?.autoSaveEnabled ?? (
    settingsSnapshot.autoSaveInBookshelfMode && isBookshelfMode
  )
  const defaultConstraints: BookTranslationConstraints = {
    glossary: { enabled: false, entries: [] },
    non_translate: { enabled: false, entries: [] },
  }

  return {
    mode,
    settingsSnapshot,
    bookTranslationConstraints: cloneDeep(
      options?.bookTranslationConstraints
        ?? bookTranslationConstraintsStore?.constraints
        ?? defaultConstraints,
    ),
    savedTextStyles: options?.savedTextStyles ?? buildSavedTextStylesFromSettings(settingsSnapshot),
    autoSaveEnabled,
    isBookshelfMode,
    sessionPath,
    bookId,
    chapterId,
  }
}

export function createTaskContext(
  imageIndex: number,
  image: AppImageData,
  translationMode: TranslationMode,
  runtime?: PipelineRuntime
): TaskContext {
  const sourceImage = cloneDeep(image)
  return {
    id: `task-${imageIndex}`,
    imageIndex,
    translationMode,
    sourceImage,
    status: 'pending',
    runtime,
    bubbleCoords: [],
    bubbleAngles: [],
    bubblePolygons: [],
    autoDirections: [],
    textMask: sourceImage.textMask ?? undefined,
    textlinesPerBubble: [],
    originalTexts: [],
    ocrResults: [],
    colors: [],
    translatedTexts: [],
    textboxTexts: [],
    warnings: [],
    cleanImage: undefined,
    finalImage: undefined,
    bubbleStates: Array.isArray(sourceImage.bubbleStates) ? sourceImage.bubbleStates : sourceImage.bubbleStates ?? null,
    persisted: false,
  }
}

export function hydrateTaskContextFromImage(
  imageIndex: number,
  image: AppImageData,
  translationMode: TranslationMode,
  runtime?: PipelineRuntime
): TaskContext {
  const context = createTaskContext(imageIndex, image, translationMode, runtime)
  const hydratedImage = context.sourceImage
  const bubbleStates = Array.isArray(hydratedImage.bubbleStates) ? hydratedImage.bubbleStates : null

  context.bubbleStates = bubbleStates
  context.bubbleCoords = bubbleStates
    ? bubbleStates.map((bubble) => bubble.coords)
    : ((hydratedImage.bubbleCoords as BubbleCoords[] | undefined) || [])
  context.bubbleAngles = bubbleStates
    ? bubbleStates.map((bubble) => bubble.rotationAngle || 0)
    : ((hydratedImage.bubbleAngles as number[] | undefined) || [])
  context.autoDirections = bubbleStates
    ? bubbleStates.map((bubble) => bubble.autoTextDirection || bubble.textDirection || 'vertical')
    : []
  context.textMask = hydratedImage.textMask ?? undefined
  context.textlinesPerBubble = bubbleStates
    ? bubbleStates.map((bubble) => bubble.textlines || [])
    : ((hydratedImage.textlinesPerBubble as BubbleTextline[][] | undefined) || [])
  context.originalTexts = bubbleStates
    ? bubbleStates.map((bubble) => bubble.originalText || '')
    : ((hydratedImage.originalTexts as string[] | undefined) || [])
  context.ocrResults = bubbleStates
    ? bubbleStates.map((bubble) => bubble.ocrResult || {
        text: bubble.originalText || '',
        confidence: null,
        confidenceSupported: false,
        engine: '',
        primaryEngine: '',
        fallbackUsed: false,
      })
    : ((hydratedImage.ocrResults as OcrResult[] | undefined) || [])
  context.translatedTexts = bubbleStates
    ? bubbleStates.map((bubble) => bubble.translatedText || '')
    : ((hydratedImage.bubbleTexts as string[] | undefined) || [])
  context.textboxTexts = bubbleStates
    ? bubbleStates.map((bubble) => bubble.textboxText || '')
    : ((hydratedImage.textboxTexts as string[] | undefined) || [])
  context.cleanImage = typeof hydratedImage.cleanImageData === 'string' ? hydratedImage.cleanImageData : undefined
  context.finalImage = typeof hydratedImage.translatedDataURL === 'string' ? hydratedImage.translatedDataURL : undefined
  context.warnings = (hydratedImage.translationWarnings as TranslationWarning[] | undefined) || []
  return context
}
