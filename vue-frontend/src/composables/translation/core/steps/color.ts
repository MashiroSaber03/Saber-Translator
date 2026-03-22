/**
 * 颜色提取步骤
 * 提取自 SequentialPipeline.ts Line 327-346
 */
import { parallelColor, type ParallelColorResponse } from '@/api/parallelTranslate'
import type { BubbleCoords } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import { getPureBase64FromImageSource } from '@/utils/imageBase64'

export interface ColorInput {
    imageIndex: number
    image: AppImageData
    bubbleCoords: BubbleCoords[]
    textlinesPerBubble: any[]
}

export interface ColorOutput {
    colors: Array<{
        textColor: string
        bgColor: string
        autoFgColor?: [number, number, number] | null
        autoBgColor?: [number, number, number] | null
    }>
}

export async function executeColor(input: ColorInput): Promise<ColorOutput> {
    const { image, bubbleCoords, textlinesPerBubble } = input

    if (bubbleCoords.length === 0) {
        return { colors: [] }
    }

    const base64 = await getPureBase64FromImageSource(image.originalDataURL)
    if (!base64) {
        throw new Error('无法读取图片数据')
    }

    const response: ParallelColorResponse = await parallelColor({
        image: base64,
        bubble_coords: bubbleCoords,
        textlines_per_bubble: textlinesPerBubble
    })

    if (!response.success) {
        throw new Error(response.error || '颜色提取失败')
    }

    return {
        colors: response.colors || []
    }
}
