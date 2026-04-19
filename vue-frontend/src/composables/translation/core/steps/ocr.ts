/**
 * OCR 步骤
 * 提取自 SequentialPipeline.ts Line 289-325
 */
import { parallelOcr, type ParallelOcrResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import type { BubbleCoords } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import { getPureBase64FromImageSource } from '@/utils/imageBase64'

export interface OcrInput {
    imageIndex: number
    image: AppImageData
    bubbleCoords: BubbleCoords[]
    bubbleProbs: number[]
    textlinesPerBubble: any[]
}

export interface OcrOutput {
    originalTexts: string[]
}

export async function executeOcr(input: OcrInput): Promise<OcrOutput> {
    const { image, bubbleCoords, bubbleProbs, textlinesPerBubble } = input

    if (bubbleCoords.length === 0) {
        return { originalTexts: [] }
    }

    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings
    const base64 = await getPureBase64FromImageSource(image.originalDataURL)
    if (!base64) {
        throw new Error('无法读取图片数据')
    }

    // PaddleOCR-VL 使用独立的源语言设置
    const ocrSourceLanguage = settings.ocrEngine === 'paddleocr_vl'
        ? settings.paddleOcrVl?.sourceLanguage || 'japanese'
        : settings.sourceLanguage

    const response: ParallelOcrResponse = await parallelOcr({
        image: base64,
        bubble_coords: bubbleCoords,
        bubble_probs: bubbleProbs,
        source_language: ocrSourceLanguage,
        ocr_engine: settings.ocrEngine,
        baidu_api_key: settings.baiduOcr?.apiKey,
        baidu_secret_key: settings.baiduOcr?.secretKey,
        baidu_version: settings.baiduOcr?.version,
        baidu_ocr_language: settings.baiduOcr?.sourceLanguage,
        ai_vision_provider: settings.aiVisionOcr?.provider,
        ai_vision_api_key: settings.aiVisionOcr?.apiKey,
        ai_vision_model_name: settings.aiVisionOcr?.modelName,
        ai_vision_ocr_prompt: settings.aiVisionOcr?.prompt,
        custom_ai_vision_base_url: settings.aiVisionOcr?.customBaseUrl,
        ai_vision_min_image_size: settings.aiVisionOcr?.minImageSize,
        ai_vision_reasoning_effort: settings.aiVisionOcr?.reasoningEffort,
        ai_vision_image_detail: settings.aiVisionOcr?.imageDetail,
        textlines_per_bubble: textlinesPerBubble
    })

    if (!response.success) {
        throw new Error(response.error || 'OCR失败')
    }

    return {
        originalTexts: response.original_texts || []
    }
}
