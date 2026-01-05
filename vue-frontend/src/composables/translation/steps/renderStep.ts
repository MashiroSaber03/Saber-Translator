/**
 * 渲染步骤执行器
 * 
 * 负责将翻译结果渲染到图片上
 */

import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { apiClient } from '@/api/client'
import type { BubbleState, TextDirection } from '@/types/bubble'
import type {
    StepExecutor,
    RenderStepResult,
    RenderStepOptions,
    TranslationJsonData,
    SavedTextStyles
} from '../core/types'

/**
 * 导入翻译结果到图片数据（不渲染，仅更新数据）
 */
export async function importTranslationData(
    translationData: TranslationJsonData[],
    savedStyles?: SavedTextStyles
): Promise<void> {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const images = imageStore.images

    // 获取当前的全局设置作为默认值
    const currentFontSize = savedStyles?.fontSize || settingsStore.settings.textStyle.fontSize
    // autoFontSize 在此函数中未使用，由 renderSingleImage 处理
    const currentFontFamily = savedStyles?.fontFamily || settingsStore.settings.textStyle.fontFamily
    const rawTextDirection = savedStyles?.textDirection || settingsStore.settings.textStyle.layoutDirection
    const currentTextDirection = (rawTextDirection === 'auto') ? 'vertical' : rawTextDirection
    const currentTextColor = savedStyles?.textColor || settingsStore.settings.textStyle.textColor
    const currentFillColor = savedStyles?.fillColor || settingsStore.settings.textStyle.fillColor
    const currentStrokeEnabled = savedStyles?.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled
    const currentStrokeColor = savedStyles?.strokeColor || settingsStore.settings.textStyle.strokeColor
    const currentStrokeWidth = savedStyles?.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth

    for (const imageData of translationData) {
        const imageIndex = imageData.imageIndex
        if (imageIndex < 0 || imageIndex >= images.length) {
            console.warn(`跳过无效的图片索引: ${imageIndex}`)
            continue
        }

        const image = images[imageIndex]
        if (!image) continue

        let imageUpdated = false
        const bubbleTexts = image.bubbleTexts || []
        const bubbleCoords = image.bubbleCoords || []

        for (const bubbleData of imageData.bubbles || []) {
            const bubbleIndex = bubbleData.bubbleIndex
            if (bubbleIndex < 0 || bubbleIndex >= bubbleCoords.length) {
                console.warn(`图片 ${imageIndex}: 跳过无效的气泡索引 ${bubbleIndex}`)
                continue
            }

            const translatedText = bubbleData.translated
            let textDirection = bubbleData.textDirection
            if (textDirection === 'auto') {
                textDirection = currentTextDirection
            }

            bubbleTexts[bubbleIndex] = translatedText
            const effectiveTextDirection: TextDirection = (textDirection === 'vertical' || textDirection === 'horizontal')
                ? textDirection
                : (currentTextDirection as TextDirection)

            // 更新或创建 bubbleStates
            if (!image.bubbleStates || !Array.isArray(image.bubbleStates) || image.bubbleStates.length !== bubbleCoords.length) {
                // 创建新的气泡设置
                const detectedAngles = image.bubbleAngles || []
                const newSettings: BubbleState[] = []
                for (let i = 0; i < bubbleCoords.length; i++) {
                    const bubbleTextDirection: TextDirection = (i === bubbleIndex) ? effectiveTextDirection : (currentTextDirection as TextDirection)
                    const coords = bubbleCoords[i]
                    // 优先使用已有 bubbleStates 中的 autoTextDirection
                    let autoDir: TextDirection = bubbleTextDirection
                    const existingState = image.bubbleStates?.[i]
                    if (existingState?.autoTextDirection && existingState.autoTextDirection !== 'auto') {
                        autoDir = existingState.autoTextDirection
                    } else if (coords && coords.length >= 4) {
                        const [x1, y1, x2, y2] = coords
                        autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
                    }
                    newSettings.push({
                        translatedText: bubbleTexts[i] || '',
                        originalText: image.originalTexts?.[i] || '',
                        textboxText: '',
                        coords: coords as [number, number, number, number],
                        polygon: [],
                        fontSize: currentFontSize,
                        fontFamily: currentFontFamily,
                        textDirection: bubbleTextDirection,
                        autoTextDirection: autoDir,
                        position: { x: 0, y: 0 },
                        textColor: currentTextColor,
                        rotationAngle: detectedAngles[i] || 0,
                        fillColor: currentFillColor,
                        strokeEnabled: currentStrokeEnabled,
                        strokeColor: currentStrokeColor,
                        strokeWidth: currentStrokeWidth,
                        inpaintMethod: settingsStore.settings.textStyle.inpaintMethod
                    })
                }
                image.bubbleStates = newSettings
            } else if (image.bubbleStates[bubbleIndex]) {
                // 更新现有的 bubbleState
                image.bubbleStates[bubbleIndex].translatedText = translatedText
                if (textDirection && textDirection !== 'auto') {
                    image.bubbleStates[bubbleIndex].textDirection = effectiveTextDirection
                }
            }

            imageUpdated = true
        }

        if (imageUpdated && image.bubbleStates) {
            // 同步 bubbleTexts
            const newBubbleTexts = image.bubbleStates.map(bs => bs.translatedText || '')
            imageStore.updateImageByIndex(imageIndex, {
                bubbleStates: image.bubbleStates,
                bubbleTexts: newBubbleTexts,
                hasUnsavedChanges: true
            })
        }
    }
}

/**
 * 渲染单张图片
 */
export async function renderSingleImage(
    imageIndex: number,
    savedStyles?: SavedTextStyles
): Promise<RenderStepResult> {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const bubbleStore = useBubbleStore()
    const image = imageStore.images[imageIndex]

    if (!image || !image.bubbleStates || !image.cleanImageData) {
        return {
            success: false,
            error: '图片数据不完整，无法渲染'
        }
    }

    const { textStyle } = settingsStore.settings
    const currentFontSize = savedStyles?.fontSize || textStyle.fontSize
    const currentAutoFontSize = savedStyles?.autoFontSize ?? textStyle.autoFontSize
    const currentFontFamily = savedStyles?.fontFamily || textStyle.fontFamily
    const rawTextDirection = savedStyles?.textDirection || textStyle.layoutDirection
    const currentTextDirection = (rawTextDirection === 'auto') ? 'vertical' : rawTextDirection
    const currentTextColor = savedStyles?.textColor || textStyle.textColor
    const currentStrokeEnabled = savedStyles?.strokeEnabled ?? textStyle.strokeEnabled
    const currentStrokeColor = savedStyles?.strokeColor || textStyle.strokeColor
    const currentStrokeWidth = savedStyles?.strokeWidth ?? textStyle.strokeWidth

    const bubbleCoords = image.bubbleCoords || []
    const bubbleTexts = image.bubbleStates.map(bs => bs.translatedText || '')

    try {
        const bubbleStatesForApi = image.bubbleStates.map(bs => ({
            translatedText: bs.translatedText || '',
            coords: bs.coords,
            fontSize: bs.fontSize || currentFontSize,
            fontFamily: bs.fontFamily || currentFontFamily,
            textDirection: bs.textDirection || currentTextDirection,
            textColor: bs.textColor || currentTextColor,
            rotationAngle: bs.rotationAngle || 0,
            position: bs.position || { x: 0, y: 0 },
            strokeEnabled: bs.strokeEnabled ?? currentStrokeEnabled,
            strokeColor: bs.strokeColor || currentStrokeColor,
            strokeWidth: bs.strokeWidth ?? currentStrokeWidth
        }))

        let cleanImageBase64 = image.cleanImageData
        if (cleanImageBase64.includes('base64,')) {
            cleanImageBase64 = cleanImageBase64.split('base64,')[1] || ''
        }

        const renderResponse = await apiClient.post<{ rendered_image?: string; error?: string }>(
            '/api/re_render_image',
            {
                clean_image: cleanImageBase64,
                bubble_texts: bubbleTexts,
                bubble_coords: bubbleCoords,
                fontSize: currentAutoFontSize ? 'auto' : currentFontSize,
                autoFontSize: currentAutoFontSize,
                fontFamily: currentFontFamily,
                textDirection: currentTextDirection,
                textColor: currentTextColor,
                bubble_states: bubbleStatesForApi,
                use_individual_styles: true,
                use_inpainting: false,
                use_lama: false,
                fillColor: null,
                is_font_style_change: false,
                strokeEnabled: currentStrokeEnabled,
                strokeColor: currentStrokeColor,
                strokeWidth: currentStrokeWidth
            }
        )

        if (renderResponse.rendered_image) {
            imageStore.updateImageByIndex(imageIndex, {
                translatedDataURL: `data:image/png;base64,${renderResponse.rendered_image}`,
                bubbleStates: image.bubbleStates,
                bubbleTexts: bubbleTexts,
                hasUnsavedChanges: true
            })

            // 如果是当前图片，同步更新 bubbleStore
            if (imageIndex === imageStore.currentImageIndex) {
                bubbleStore.setBubbles(image.bubbleStates)
            }

            return {
                success: true,
                renderedImage: renderResponse.rendered_image,
                bubbleStates: image.bubbleStates
            }
        } else {
            return {
                success: false,
                error: renderResponse.error || '渲染失败'
            }
        }
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : '渲染失败'
        console.error(`渲染图片 ${imageIndex} 失败:`, error)
        return {
            success: false,
            error: errorMessage
        }
    }
}

/**
 * 渲染所有需要渲染的图片
 */
export async function renderAllImages(
    translationData: TranslationJsonData[],
    savedStyles?: SavedTextStyles,
    onProgress?: (current: number, total: number) => void
): Promise<{ success: number; failed: number }> {
    const imageStore = useImageStore()
    const images = imageStore.images

    // 找出需要渲染的图片索引
    const imagesToRender: number[] = translationData
        .map(d => d.imageIndex)
        .filter(idx => {
            const image = images[idx]
            return image && image.cleanImageData && image.bubbleStates && image.bubbleStates.length > 0
        })

    let successCount = 0
    let failCount = 0

    for (let i = 0; i < imagesToRender.length; i++) {
        const imageIndex = imagesToRender[i]
        if (imageIndex === undefined) continue

        if (onProgress) {
            onProgress(i + 1, imagesToRender.length)
        }

        const result = await renderSingleImage(imageIndex, savedStyles)
        if (result.success) {
            successCount++
        } else {
            failCount++
        }
    }

    return { success: successCount, failed: failCount }
}

/**
 * 渲染步骤执行器（用于单张图片模式）
 */
export const renderStepExecutor: StepExecutor<RenderStepResult> = {
    name: 'render',
    type: 'render',

    async execute(
        context,
        _options?: RenderStepOptions
    ): Promise<RenderStepResult> {
        const { imageIndex } = context
        return renderSingleImage(imageIndex)
    }
}

/**
 * 获取渲染步骤执行器
 */
export function getRenderStepExecutor(): StepExecutor<RenderStepResult> {
    return renderStepExecutor
}
