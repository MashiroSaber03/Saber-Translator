import { parallelInpaint, type ParallelInpaintResponse } from '@/api/parallelTranslate'
import type { BubbleCoords } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import type { TranslationSettings } from '@/types/settings'
import { extractBase64Payload } from '@/utils/dataUrl'

export interface InpaintInput {
    imageIndex: number
    image: AppImageData
    translationMode?: string
    bubbleCoords: BubbleCoords[]
    bubblePolygons: number[][][]
    textMask?: string
    userMask?: string
    settingsSnapshot: TranslationSettings
}

export interface InpaintOutput {
    cleanImage: string
}

export async function executeInpaint(input: InpaintInput): Promise<InpaintOutput> {
    const { image, bubbleCoords, bubblePolygons, textMask, userMask, translationMode = 'standard', settingsSnapshot } = input

    if (bubbleCoords.length === 0) {
        return { cleanImage: extractBase64Payload(image.originalDataURL) }
    }

    const settings = settingsSnapshot
    const { textStyle, preciseMask } = settings
    const base64 = extractBase64Payload(image.originalDataURL)

    const response: ParallelInpaintResponse = await parallelInpaint({
        image: base64,
        bubble_coords: bubbleCoords,
        translation_mode: translationMode,
        translation_scope: 'image',
        bubble_polygons: bubblePolygons,
        raw_mask: textMask || undefined,
        user_mask: userMask || undefined,
        method: textStyle.inpaintMethod === 'solid' ? 'solid' : 'lama',
        lama_model: textStyle.inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe',
        fill_color: textStyle.fillColor,
        mask_dilate_size: preciseMask.dilateSize,
        mask_box_expand_ratio: preciseMask.boxExpandRatio
    })

    if (!response.success) {
        throw new Error(response.error || '背景修复失败')
    }

    return { cleanImage: response.clean_image || '' }
}

