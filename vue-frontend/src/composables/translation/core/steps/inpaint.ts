/**
 * 修复步骤
 * 提取自 SequentialPipeline.ts Line 599-626
 */
import { parallelInpaint, type ParallelInpaintResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import type { BubbleCoords } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'

export interface InpaintInput {
    imageIndex: number
    image: AppImageData
    bubbleCoords: BubbleCoords[]
    bubblePolygons: number[][][]
    rawMask?: string
}

export interface InpaintOutput {
    cleanImage: string
}

export async function executeInpaint(input: InpaintInput): Promise<InpaintOutput> {
    const { image, bubbleCoords, bubblePolygons, rawMask } = input

    if (bubbleCoords.length === 0) {
        return { cleanImage: extractBase64(image.originalDataURL) }
    }

    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings
    const { textStyle, preciseMask } = settings
    const base64 = extractBase64(image.originalDataURL)

    const response: ParallelInpaintResponse = await parallelInpaint({
        image: base64,
        bubble_coords: bubbleCoords,
        bubble_polygons: bubblePolygons,
        raw_mask: rawMask,
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

function extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
        return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
}
