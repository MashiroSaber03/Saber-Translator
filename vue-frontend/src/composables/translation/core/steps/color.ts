/**
 * 颜色提取步骤
 * 提取自 SequentialPipeline.ts Line 327-346
 */
import { parallelColor, type ParallelColorResponse } from '@/api/parallelTranslate'
import type { BubbleCoords } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'

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

    // 复用已有气泡颜色：
    // 当图片已有 bubbleStates 且检测步骤被跳过时，textlinesPerBubble 会为空，
    // 48px 颜色提取会退化到简单裁剪模式，稳定性较差。
    // 这种情况下优先复用已缓存的自动颜色，避免无意义重算。
    if (!hasUsableTextlines(textlinesPerBubble, bubbleCoords.length)) {
        const cachedColors = getCachedColorsFromBubbleStates(image, bubbleCoords.length)
        if (cachedColors) {
            console.info(`[颜色提取] 复用已缓存颜色（页内已有气泡数据）`)
            return { colors: cachedColors }
        }
    }

    const base64 = extractBase64(image.originalDataURL)

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

function extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
        return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
}

function hasUsableTextlines(textlinesPerBubble: any[], expectedBubbleCount: number): boolean {
    if (!Array.isArray(textlinesPerBubble) || textlinesPerBubble.length !== expectedBubbleCount) {
        return false
    }
    return textlinesPerBubble.some(lines => Array.isArray(lines) && lines.length > 0)
}

function getCachedColorsFromBubbleStates(
    image: AppImageData,
    expectedBubbleCount: number
): ColorOutput['colors'] | null {
    const states = image.bubbleStates
    if (!states || states.length !== expectedBubbleCount) {
        return null
    }

    const colors = states.map(state => ({
        textColor: state.textColor || '#000000',
        bgColor: state.fillColor || '#FFFFFF',
        autoFgColor: state.autoFgColor || null,
        autoBgColor: state.autoBgColor || null
    }))

    const hasAnyAutoColor = colors.some(c => c.autoFgColor || c.autoBgColor)
    return hasAnyAutoColor ? colors : null
}
