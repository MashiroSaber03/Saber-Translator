import type {
  BubbleCoords,
  BubbleState,
  BubbleTextline,
  InpaintMethod,
  TextAlign,
  TextDirection,
} from './bubble'
import type { OcrResult } from './ocr'
import type { TranslationWarning } from './translationConstraints'

export type TranslationStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface ImageSourceFields {
  id: string
  chapterId?: string
  sourceRevision?: number
  documentRevision?: number
  renderedRevision?: number | null
  fileName: string
  width?: number
  height?: number
  originalDataURL: string
  translatedDataURL: string | null
  cleanImageData: string | null
  sourceAssetUrl?: string
  thumbnailSourceUrl?: string
  translatedAssetUrl?: string | null
  thumbnailTranslatedUrl?: string | null
}

export interface ImageDetectionFields {
  bubbleStates: BubbleState[] | null
  bubbleCoords?: BubbleCoords[]
  bubbleAngles?: number[]
  originalTexts?: string[]
  textlinesPerBubble?: BubbleTextline[][]
  ocrResults?: OcrResult[]
  bubbleTexts?: string[]
  textboxTexts?: string[]
}

export interface ImageMaskFields {
  textMask?: string | null
  userMask?: string | null
  isManuallyAnnotated?: boolean
}

export interface ImageWorkflowFields {
  translationStatus: TranslationStatus
  translationFailed: boolean
  errorMessage?: string
  translationWarnings?: TranslationWarning[]
}

export interface ImageTextStyleFields {
  fontSize: number
  autoFontSize: boolean
  fontFamily: string
  layoutDirection: TextDirection
  textColor: string
  fillColor: string
  inpaintMethod: InpaintMethod
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  lineSpacing?: number
  textAlign?: TextAlign
  useAutoTextColor?: boolean
}

export interface ImageUiFields {
  hasUnsavedChanges: boolean
  isManualAnnotation?: boolean
  showOriginal?: boolean
}

export interface ImageFolderFields {
  relativePath?: string
  folderPath?: string
}

export interface ImageData
  extends ImageSourceFields,
    ImageDetectionFields,
    ImageMaskFields,
    ImageWorkflowFields,
    ImageTextStyleFields,
    ImageUiFields,
    ImageFolderFields {}

export interface ImageDataLoadInput
  extends ImageSourceFields,
    ImageDetectionFields,
    ImageMaskFields,
    ImageWorkflowFields,
    Partial<ImageTextStyleFields>,
    ImageUiFields,
    ImageFolderFields {}

export type ImageDataOverrides = Partial<ImageData>

export type ImageDataUpdates = Partial<ImageData>

export interface ImageUploadResult {
  success: boolean
  images: ImageData[]
  errors?: string[]
}

export interface PdfParseSession {
  sessionId: string
  totalPages: number
  currentPage: number
}

export interface MobiParseSession {
  sessionId: string
  totalPages: number
  currentPage: number
}
