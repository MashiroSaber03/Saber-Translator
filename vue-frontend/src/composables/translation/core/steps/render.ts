/**
 * 渲染步骤
 * 提取自 SequentialPipeline.ts Line 628-715
 * 
 * 注意：这是最复杂的步骤之一，需要处理文字方向、颜色、savedTextStyles等
 */
import { parallelRender, type ParallelRenderResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import type { SavedTextStyles } from '../types'

export interface RenderInput {
    imageIndex: number
    cleanImage: string
    bubbleCoords: BubbleCoords[]
    bubbleAngles: number[]
    autoDirections: string[]
    originalTexts: string[]
    translatedTexts: string[]
    textboxTexts: string[]
    colors: Array<{
        textColor: string
        bgColor: string
        autoFgColor?: [number, number, number] | null
        autoBgColor?: [number, number, number] | null
    }>
    savedTextStyles?: SavedTextStyles | null
    currentMode: string
}

export interface RenderOutput {
    finalImage: string
    bubbleStates: BubbleState[]
}

export async function executeRender(input: RenderInput): Promise<RenderOutput> {
    const {
        cleanImage,
        bubbleCoords,
        bubbleAngles,
        autoDirections,
        originalTexts,
        translatedTexts,
        textboxTexts,
        colors,
        savedTextStyles,
        currentMode
    } = input

    if (!cleanImage) {
        // 校对模式下，如果没有干净背景图，说明图片没有被翻译过
        if (currentMode === 'proofread') {
            throw new Error('此图片尚未翻译，请先翻译后再进行校对')
        }
        throw new Error('缺少干净背景图片')
    }

    const settingsStore = useSettingsStore()
    const { textStyle } = settingsStore.settings

    // 【简化设计】计算 textDirection：
    // - 如果全局设置是 'auto'，使用检测结果
    // - 否则使用全局设置的值
    const globalTextDir = savedTextStyles?.autoTextDirection
        ? 'auto'  // autoTextDirection 为 true 表示用户选择了 'auto'
        : (savedTextStyles?.textDirection || textStyle.layoutDirection)

    // 构建 bubbleStates
    const bubbleStates: BubbleState[] = bubbleCoords.map((coords, idx) => {
        const autoDir = autoDirections[idx] || 'vertical'
        // 将后端返回的 'v'/'h' 格式转换为 'vertical'/'horizontal'
        const mappedAutoDir: 'vertical' | 'horizontal' = autoDir === 'v' ? 'vertical'
            : autoDir === 'h' ? 'horizontal'
                : (autoDir === 'vertical' || autoDir === 'horizontal') ? autoDir : 'vertical'

        // 【简化设计】textDirection 直接使用具体方向值
        const textDirection =
            (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
                ? globalTextDir
                : mappedAutoDir

        // 【修复】颜色处理：根据 useAutoTextColor 设置决定是否使用自动提取的颜色
        const useAutoColor = savedTextStyles?.useAutoTextColor ?? textStyle.useAutoTextColor
        let finalTextColor = savedTextStyles?.textColor || textStyle.textColor
        let finalFillColor = savedTextStyles?.fillColor || textStyle.fillColor
        const colorInfo = colors[idx]

        if (useAutoColor && colorInfo) {
            if (colorInfo.textColor) finalTextColor = colorInfo.textColor
            if (colorInfo.bgColor) finalFillColor = colorInfo.bgColor
        }

        return {
            coords,
            polygon: [] as number[][],
            position: { x: 0, y: 0 },
            rotationAngle: bubbleAngles[idx] || 0,
            originalText: originalTexts[idx] || '',
            translatedText: translatedTexts[idx] || '',
            textboxText: textboxTexts[idx] || '',
            textDirection: textDirection as 'vertical' | 'horizontal',  // 渲染用的具体方向
            autoTextDirection: mappedAutoDir as 'vertical' | 'horizontal',  // 备份检测结果
            fontSize: savedTextStyles?.fontSize || textStyle.fontSize,
            fontFamily: savedTextStyles?.fontFamily || textStyle.fontFamily,
            autoFontSize: savedTextStyles?.autoFontSize ?? textStyle.autoFontSize,
            textColor: finalTextColor,
            fillColor: finalFillColor,
            strokeEnabled: savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
            strokeColor: savedTextStyles?.strokeColor || textStyle.strokeColor,
            strokeWidth: savedTextStyles?.strokeWidth || textStyle.strokeWidth,
            inpaintMethod: savedTextStyles?.inpaintMethod || textStyle.inpaintMethod,
            autoFgColor: colors[idx]?.autoFgColor || null,
            autoBgColor: colors[idx]?.autoBgColor || null
        }
    })

    const response: ParallelRenderResponse = await parallelRender({
        clean_image: cleanImage,
        bubble_states: bubbleStates,
        fontSize: savedTextStyles?.fontSize || textStyle.fontSize,
        fontFamily: savedTextStyles?.fontFamily || textStyle.fontFamily,
        textDirection: savedTextStyles?.textDirection || textStyle.layoutDirection,
        textColor: savedTextStyles?.textColor || textStyle.textColor,
        strokeEnabled: savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
        strokeColor: savedTextStyles?.strokeColor || textStyle.strokeColor,
        strokeWidth: savedTextStyles?.strokeWidth || textStyle.strokeWidth,
        autoFontSize: savedTextStyles?.autoFontSize ?? textStyle.autoFontSize,
        use_individual_styles: true
    })

    if (!response.success) {
        throw new Error(response.error || '渲染失败')
    }

    return {
        finalImage: response.final_image || '',
        bubbleStates: response.bubble_states || bubbleStates
    }
}
