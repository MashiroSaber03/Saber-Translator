import type {
  BubbleState,
  InpaintMethod,
  LogicalAlign,
  TextDirection,
} from './bubble'

export type TranslationStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface ImageSourceFields {
  id: string
  chapterId?: string
  documentRevision?: number
  renderedRevision?: number | null
  fileName: string
  width?: number
  height?: number
  sourceAssetUrl: string
  translatedAssetUrl: string | null
  cleanAssetUrl: string | null
  thumbnailSourceUrl: string
}

export interface ImageDetectionFields {
  bubbleStates: BubbleState[] | null
  isManuallyAnnotated?: boolean
}

export interface ImageWorkflowFields {
  translationStatus: TranslationStatus
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
  lineSpacing: number
  inlineAlign: LogicalAlign
  blockAlign: LogicalAlign
  useAutoTextColor: boolean
}

export interface ImageUiFields {
  hasUnsavedChanges: boolean
  showOriginal?: boolean
}

export interface ImageFolderFields {
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

export type ImageDataUpdates = Partial<ImageData>
