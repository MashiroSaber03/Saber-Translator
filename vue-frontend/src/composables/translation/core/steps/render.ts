import { parallelRender, type ParallelRenderResponse } from '@/api/parallelTranslate'
import type { BubbleState, BubbleCoords, BubbleTextline } from '@/types/bubble'
import type { SavedTextStyles } from '../types'
import type { OcrResult } from '@/types/ocr'
import type { TranslationSettings } from '@/types/settings'
import { cloneBubbleStates } from '@/utils/bubbleFactory'

export interface RenderInput {
    imageIndex: number
    cleanImage: string
    bubbleCoords: BubbleCoords[]
    bubbleAngles: number[]
    autoDirections: string[]
    textlinesPerBubble?: BubbleTextline[][]
    existingBubbleStates?: BubbleState[] | null
    originalTexts: string[]
    ocrResults?: OcrResult[]
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
    settingsSnapshot: TranslationSettings
    renderStylePolicy: RenderStylePolicy
}

export interface RenderOutput {
    finalImage: string
    bubbleStates: BubbleState[]
}

export interface RenderStylePolicy {
    fontSize: 'preserve' | 'initialize_auto'
    color: 'preserve' | 'initialize_auto'
}

function hasOwn(object: object, key: string): boolean {
    return Object.prototype.hasOwnProperty.call(object, key)
}

function mergeRenderedBubbleStates(
    localBubbleStates: BubbleState[],
    renderedBubbleStates?: BubbleState[]
): BubbleState[] {
    if (!renderedBubbleStates || renderedBubbleStates.length === 0) {
        return localBubbleStates
    }

    return renderedBubbleStates.map((renderedState, index) => {
        const localState = localBubbleStates[index]
        if (!localState) {
            return renderedState
        }

        const mergedState: BubbleState = {
            ...localState,
            ...renderedState
        }

        if (!hasOwn(renderedState, 'textlines')) {
            mergedState.textlines = localState.textlines
        }
        if (!hasOwn(renderedState, 'ocrResult')) {
            mergedState.ocrResult = localState.ocrResult ?? null
        }
        if (!hasOwn(renderedState, 'colorConfidence')) {
            mergedState.colorConfidence = localState.colorConfidence
        }

        return mergedState
    })
}

export async function executeRender(input: RenderInput): Promise<RenderOutput> {
    const {
        cleanImage,
        bubbleCoords,
        bubbleAngles,
        autoDirections,
        textlinesPerBubble,
        existingBubbleStates,
        originalTexts,
        ocrResults,
        translatedTexts,
        textboxTexts,
        colors,
        savedTextStyles,
        currentMode,
        settingsSnapshot,
        renderStylePolicy,
    } = input

    if (!cleanImage) {
        if (currentMode === 'proofread') {
            throw new Error('此图片尚未翻译，请先翻译后再进行校对')
        }
        throw new Error('缺少干净背景图片')
    }

    const { textStyle } = settingsSnapshot
    const autoFontSizeEnabled = savedTextStyles?.autoFontSize ?? textStyle.autoFontSize
    const autoTextColorEnabled = savedTextStyles?.useAutoTextColor ?? textStyle.useAutoTextColor
    const shouldInitializeAutoFontSize = renderStylePolicy.fontSize === 'initialize_auto' && autoFontSizeEnabled
    const shouldInitializeAutoColor = renderStylePolicy.color === 'initialize_auto' && autoTextColorEnabled

    const globalTextDir = savedTextStyles?.autoTextDirection
        ? 'auto'
        : (savedTextStyles?.textDirection || textStyle.layoutDirection)

    const bubbleStatesSource = existingBubbleStates && existingBubbleStates.length === bubbleCoords.length
        ? cloneBubbleStates(existingBubbleStates)
        : null

    const bubbleStates: BubbleState[] = bubbleCoords.map((coords, idx) => {
        const autoDir = autoDirections[idx] || 'vertical'
        const mappedAutoDir: 'vertical' | 'horizontal' = autoDir === 'v' ? 'vertical'
            : autoDir === 'h' ? 'horizontal'
                : (autoDir === 'vertical' || autoDir === 'horizontal') ? autoDir : 'vertical'
        const baseState = bubbleStatesSource?.[idx]

        const textDirection =
            (baseState?.textDirection === 'vertical' || baseState?.textDirection === 'horizontal')
                ? baseState.textDirection
                : (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
                    ? globalTextDir
                    : mappedAutoDir

        let finalTextColor = baseState?.textColor ?? savedTextStyles?.textColor ?? textStyle.textColor
        let finalFillColor = baseState?.fillColor ?? savedTextStyles?.fillColor ?? textStyle.fillColor
        const colorInfo = colors[idx]

        if (shouldInitializeAutoColor && colorInfo) {
            if (colorInfo.textColor) finalTextColor = colorInfo.textColor
            if (colorInfo.bgColor) finalFillColor = colorInfo.bgColor
        }

        return {
            ...(baseState || {}),
            coords,
            polygon: baseState?.polygon || [] as number[][],
            position: baseState?.position || { x: 0, y: 0 },
            rotationAngle: bubbleAngles[idx] || 0,
            originalText: originalTexts[idx] || '',
            textlines: textlinesPerBubble?.[idx] || baseState?.textlines || [],
            ocrResult: ocrResults?.[idx] || null,
            translatedText: translatedTexts[idx] || '',
            textboxText: textboxTexts[idx] || '',
            textDirection: textDirection as 'vertical' | 'horizontal',
            autoTextDirection: mappedAutoDir as 'vertical' | 'horizontal',
            fontSize: baseState?.fontSize ?? savedTextStyles?.fontSize ?? textStyle.fontSize,
            fontFamily: baseState?.fontFamily ?? savedTextStyles?.fontFamily ?? textStyle.fontFamily,
            textColor: finalTextColor,
            fillColor: finalFillColor,
            strokeEnabled: baseState?.strokeEnabled ?? savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
            strokeColor: baseState?.strokeColor ?? savedTextStyles?.strokeColor ?? textStyle.strokeColor,
            strokeWidth: baseState?.strokeWidth ?? savedTextStyles?.strokeWidth ?? textStyle.strokeWidth,
            lineSpacing: baseState?.lineSpacing ?? savedTextStyles?.lineSpacing ?? textStyle.lineSpacing,
            textAlign: baseState?.textAlign ?? savedTextStyles?.textAlign ?? textStyle.textAlign,
            inpaintMethod: baseState?.inpaintMethod ?? savedTextStyles?.inpaintMethod ?? textStyle.inpaintMethod,
            autoFgColor: colors[idx]?.autoFgColor ?? baseState?.autoFgColor ?? null,
            autoBgColor: colors[idx]?.autoBgColor ?? baseState?.autoBgColor ?? null
        }
    })

    const response: ParallelRenderResponse = await parallelRender({
        clean_image: cleanImage,
        bubble_states: bubbleStates,
        translation_mode: currentMode,
        translation_scope: 'image',
        fontSize: bubbleStates[0]?.fontSize ?? savedTextStyles?.fontSize ?? textStyle.fontSize,
        fontFamily: bubbleStates[0]?.fontFamily ?? savedTextStyles?.fontFamily ?? textStyle.fontFamily,
        textDirection: bubbleStates[0]?.textDirection ?? savedTextStyles?.textDirection ?? textStyle.layoutDirection,
        textColor: bubbleStates[0]?.textColor ?? savedTextStyles?.textColor ?? textStyle.textColor,
        strokeEnabled: bubbleStates[0]?.strokeEnabled ?? savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
        strokeColor: bubbleStates[0]?.strokeColor ?? savedTextStyles?.strokeColor ?? textStyle.strokeColor,
        strokeWidth: bubbleStates[0]?.strokeWidth ?? savedTextStyles?.strokeWidth ?? textStyle.strokeWidth,
        lineSpacing: bubbleStates[0]?.lineSpacing ?? savedTextStyles?.lineSpacing ?? textStyle.lineSpacing,
        textAlign: bubbleStates[0]?.textAlign ?? savedTextStyles?.textAlign ?? textStyle.textAlign,
        autoFontSize: shouldInitializeAutoFontSize,
        use_individual_styles: true
    })

    if (!response.success) {
        throw new Error(response.error || '渲染失败')
    }

    const finalBubbleStates = mergeRenderedBubbleStates(bubbleStates, response.bubble_states)

    return {
        finalImage: response.final_image || '',
        bubbleStates: finalBubbleStates
    }
}
