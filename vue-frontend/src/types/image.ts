import type {
  BubbleState,
  InpaintMethod,
  TextAlign,
  TextDirection,
} from './bubble'
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
  sourceAssetUrl: string
  translatedAssetUrl: string | null
  cleanAssetUrl: string | null
  thumbnailSourceUrl?: string
  thumbnailTranslatedUrl?: string | null
}

export interface ImageDetectionFields {
  bubbleStates: BubbleState[] | null
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
  showOriginal?: boolean
}

export interface ImageFolderFields {
  relativePath?: string
  folderPath?: string
}

export interface ImageData
  extends ImageSourceFields,
    ImageDetectionFields,
    ImageWorkflowFields,
    ImageTextStyleFields,
    ImageUiFields,
    ImageFolderFields {}

export interface ImageDataLoadInput
  extends ImageSourceFields,
    ImageDetectionFields,
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
