/**
 * OCR 步骤
 * 提取自 SequentialPipeline.ts Line 289-325
 */
import { parallelOcr, type ParallelOcrResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import type { BubbleCoords, BubbleState } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import type { OcrResult } from '@/types/ocr'
import { getTextlinesPerBubbleFromStates } from '@/utils/bubbleFactory'

export interface OcrInput {
    imageIndex: number
    image: AppImageData
    bubbleCoords: BubbleCoords[]
    bubbleStates?: BubbleState[] | null
    textlinesPerBubble?: any[]
}

export interface OcrOutput {
    originalTexts: string[]
    ocrResults: OcrResult[]
}

export async function executeOcr(input: OcrInput): Promise<OcrOutput> {
    const { image, bubbleCoords, bubbleStates, textlinesPerBubble } = input

    if (bubbleCoords.length === 0) {
        return { originalTexts: [], ocrResults: [] }
    }

    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings
    const base64 = extractBase64(image.originalDataURL)

    // PaddleOCR-VL 使用独立的源语言设置
    const ocrSourceLanguage = settings.ocrEngine === 'paddleocr_vl'
        ? settings.paddleOcrVl?.sourceLanguage || 'japanese'
        : settings.sourceLanguage

    const bubbleStateTextlines = bubbleStates && bubbleStates.length > 0
        ? getTextlinesPerBubbleFromStates(bubbleStates)
        : []
    const preferredTextlines = bubbleCoords.map((_, index) => {
        const stateTextlines = bubbleStateTextlines[index]
        if (stateTextlines && stateTextlines.length > 0) {
            return stateTextlines
        }
        return textlinesPerBubble?.[index] || []
    })

    const response: ParallelOcrResponse = await parallelOcr({
        image: base64,
        bubble_coords: bubbleCoords,
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
        enable_hybrid_ocr: settings.hybridOcr?.enabled,
        secondary_ocr_engine: settings.hybridOcr?.secondaryEngine,
        hybrid_ocr_threshold: settings.hybridOcr?.confidenceThreshold,
        textlines_per_bubble: preferredTextlines
    })

    if (!response.success) {
        throw new Error(response.error || 'OCR失败')
    }

    return {
        originalTexts: response.original_texts || [],
        ocrResults: response.ocr_results || []
    }
}

function extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
        return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
}
