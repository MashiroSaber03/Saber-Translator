/**
 * 准备步骤执行器
 * 
 * 负责图片的检测、OCR、颜色提取、背景修复等准备工作
 * 调用后端 /api/translate_image API
 */

import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import {
    translateImage as translateImageApi,
    type TranslateImageParams
} from '@/api/translate'
import { extractExistingBubbleData, buildTranslateParams, createBubbleStatesFromApiResponse } from '../utils'
import type {
    StepExecutor,
    PrepareStepResult,
    PrepareStepOptions,
    ImageExecutionContext
} from '../core/types'

/**
 * 准备步骤执行器
 */
export const prepareStepExecutor: StepExecutor<PrepareStepResult> = {
    name: 'prepare',
    type: 'prepare',

    async execute(
        context: ImageExecutionContext,
        options?: PrepareStepOptions
    ): Promise<PrepareStepResult> {
        const { imageIndex, image, progress } = context
        const imageStore = useImageStore()
        const bubbleStore = useBubbleStore()

        const skipTranslation = options?.skipTranslation ?? false

        // 设置处理中状态
        imageStore.setTranslationStatus(imageIndex, 'processing')

        try {
            // 提取已有气泡数据
            const existingData = extractExistingBubbleData(image)

            // 如果用户清空了文本框，跳过处理
            if (existingData?.isEmpty) {
                console.log(`准备步骤[${imageIndex}]: 文本框已被用户清空，跳过`)
                imageStore.setTranslationStatus(imageIndex, 'pending')
                return {
                    success: true,
                    bubbleCoords: [],
                    originalTexts: [],
                    bubbleStates: []
                }
            }

            // 构建请求参数
            const params = buildTranslateParams(image.originalDataURL, {
                removeTextOnly: skipTranslation,
                existingBubbleCoords: existingData?.coords,
                existingBubbleAngles: existingData?.angles,
                useExistingBubbles: !!existingData
            })

            // 调用后端 API
            const response = await translateImageApi(params as unknown as TranslateImageParams)

            // 处理响应
            if (response.error || !response.translated_image) {
                imageStore.setTranslationStatus(imageIndex, 'failed', response.error || '处理失败')
                return {
                    success: false,
                    error: response.error || '处理失败'
                }
            }

            // 创建气泡状态
            const bubbleStates = createBubbleStatesFromApiResponse(response)

            // 更新图片数据
            const translatedDataURL = `data:image/png;base64,${response.translated_image}`

            imageStore.updateImageByIndex(imageIndex, {
                translatedDataURL,
                cleanImageData: response.clean_image || null,
                bubbleStates,
                bubbleCoords: response.bubble_coords,
                bubbleAngles: response.bubble_angles || [],
                originalTexts: response.original_texts,
                textboxTexts: response.textbox_texts || [],
                bubbleTexts: bubbleStates?.map(s => s.translatedText || '') || [],
                translationStatus: 'completed',
                translationFailed: false,
                showOriginal: false,
                hasUnsavedChanges: true
            })

            // 如果是当前图片，同步更新 bubbleStore
            if (imageIndex === imageStore.currentImageIndex && bubbleStates) {
                bubbleStore.setBubbles(bubbleStates)
            }

            progress.incrementCompleted()

            return {
                success: true,
                translatedImage: response.translated_image,
                cleanImage: response.clean_image,
                bubbleCoords: response.bubble_coords,
                bubbleAngles: response.bubble_angles,
                originalTexts: response.original_texts,
                bubbleTexts: response.bubble_texts,
                textboxTexts: response.textbox_texts,
                bubbleStates: bubbleStates || undefined
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '处理失败'
            imageStore.setTranslationStatus(imageIndex, 'failed', errorMessage)
            progress.incrementFailed()

            return {
                success: false,
                error: errorMessage
            }
        }
    }
}

/**
 * 获取准备步骤执行器
 */
export function getPrepareStepExecutor(): StepExecutor<PrepareStepResult> {
    return prepareStepExecutor
}
