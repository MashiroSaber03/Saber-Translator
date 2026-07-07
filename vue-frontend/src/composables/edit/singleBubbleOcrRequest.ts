import { normalizeProviderId } from '@/config/aiProviders'
import type { OcrSingleBubbleOptions } from '@/api/translate'
import type { BubbleCoords, BubbleState, BubbleTextline } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import type { TranslationSettings } from '@/types/settings'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'

export interface SingleBubbleOcrRequestParams {
  image: ImageData
  bubble: BubbleState
  bubbleIndex: number
  settings: TranslationSettings
}

export interface SingleBubbleOcrRequest {
  imageData: string
  bubbleCoords: BubbleCoords
  ocrEngine: string
  bubbleTextlines: BubbleTextline[]
  options: OcrSingleBubbleOptions
}

function resolveBubbleTextlines(
  image: ImageData,
  bubble: BubbleState,
  bubbleIndex: number,
): BubbleTextline[] {
  if (bubble.textlines?.length) {
    return bubble.textlines
  }

  return Array.isArray(image.textlinesPerBubble)
    ? image.textlinesPerBubble[bubbleIndex] || []
    : []
}

export function buildSingleBubbleOcrRequest({
  image,
  bubble,
  bubbleIndex,
  settings,
}: SingleBubbleOcrRequestParams): SingleBubbleOcrRequest {
  const bubbleTextlines = resolveBubbleTextlines(image, bubble, bubbleIndex)
  const ocrSourceLanguage = settings.ocrEngine === 'paddleocr_vl'
    ? settings.paddleOcrVl?.sourceLanguage || 'japanese'
    : settings.sourceLanguage

  return {
    imageData: image.originalDataURL.split(',')[1] || '',
    bubbleCoords: bubble.coords,
    ocrEngine: settings.ocrEngine || 'manga_ocr',
    bubbleTextlines,
    options: {
      source_language: ocrSourceLanguage,
      baidu_ocr_api_key: settings.baiduOcr.apiKey,
      baidu_ocr_secret_key: settings.baiduOcr.secretKey,
      baidu_version: settings.baiduOcr.version,
      baidu_source_language: settings.baiduOcr.sourceLanguage,
      ai_vision_provider: normalizeProviderId(settings.aiVisionOcr.provider),
      ai_vision_api_key: settings.aiVisionOcr.apiKey,
      ai_vision_model_name: settings.aiVisionOcr.modelName,
      ai_vision_ocr_prompt: settings.aiVisionOcr.prompt,
      ai_vision_prompt_mode: settings.aiVisionOcr.promptMode,
      custom_ai_vision_base_url: settings.aiVisionOcr.customBaseUrl,
      openai_options: serializeOpenAICompatibleOptionsForApi(settings.aiVisionOcr.openaiOptions),
      ai_vision_min_image_size: settings.aiVisionOcr.minImageSize,
      enable_hybrid_ocr: settings.hybridOcr.enabled,
      secondary_ocr_engine: settings.hybridOcr.secondaryEngine,
      hybrid_ocr_threshold: settings.hybridOcr.confidenceThreshold,
      bubble_textlines: bubbleTextlines,
      text_detector: settings.textDetector,
      enable_aux_yolo_detection: settings.enableAuxYoloDetection,
      aux_yolo_conf_threshold: settings.auxYoloConfThreshold,
      aux_yolo_overlap_threshold: settings.auxYoloOverlapThreshold,
      enable_saber_yolo_refine: settings.enableSaberYoloRefine,
      saber_yolo_refine_overlap_threshold: settings.saberYoloRefineOverlapThreshold,
    },
  }
}
