import { parallelColor, type ParallelColorResponse } from '@/api/parallelTranslate'
import type { BubbleCoords, BubbleState, BubbleTextline } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import { getTextlinesPerBubbleFromStates } from '@/utils/bubbleFactory'
import { extractBase64Payload } from '@/utils/dataUrl'

export interface ColorInput {
    imageIndex: number
    image: AppImageData
    translationMode?: string
    bubbleCoords: BubbleCoords[]
    bubbleStates?: BubbleState[] | null
    textlinesPerBubble?: BubbleTextline[][]
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
    const { image, bubbleCoords, bubbleStates, textlinesPerBubble, translationMode = 'standard' } = input

    if (bubbleCoords.length === 0) {
        return { colors: [] }
    }

    const base64 = extractBase64Payload(image.originalDataURL)

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

    const response: ParallelColorResponse = await parallelColor({
        image: base64,
        bubble_coords: bubbleCoords,
        translation_mode: translationMode,
        translation_scope: 'image',
        textlines_per_bubble: preferredTextlines
    })

    if (!response.success) {
        throw new Error(response.error || '颜色提取失败')
    }

    return {
        colors: response.colors || []
    }
}
