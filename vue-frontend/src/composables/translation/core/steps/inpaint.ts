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
    textMask?: string      // 文字检测掩膜
    userMask?: string      // 用户笔刷掩膜
}

export interface InpaintOutput {
    cleanImage: string
}

export async function executeInpaint(input: InpaintInput): Promise<InpaintOutput> {
    const { image, bubbleCoords, bubblePolygons, textMask, userMask } = input

    if (bubbleCoords.length === 0) {
        return { cleanImage: extractBase64(image.originalDataURL) }
    }

    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings
    const { textStyle, preciseMask } = settings
    const base64 = extractBase64(image.originalDataURL)

    // ✅ 分别发送 textMask 和 userMask，由后端合并处理
    console.log(`修复步骤 - textMask: ${textMask ? '✅' : '❌'}, userMask: ${userMask ? '✅' : '❌'}`)

    const response: ParallelInpaintResponse = await parallelInpaint({
        image: base64,
        bubble_coords: bubbleCoords,
        bubble_polygons: bubblePolygons,
        raw_mask: textMask || undefined,      // 文字检测掩膜
        user_mask: userMask || undefined,     // 用户笔刷掩膜（新增）
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

