/**
 * 检测步骤
 * 提取自 SequentialPipeline.ts Line 234-287
 */
import { parallelDetect, type ParallelDetectResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleCoords, BubbleState, BubbleTextline } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import { createBubbleState } from '@/utils/bubbleFactory'

export interface DetectionInput {
    imageIndex: number
    image: AppImageData
    forceDetect?: boolean
}

export interface DetectionOutput {
    bubbleCoords: BubbleCoords[]
    bubbleAngles: number[]
    bubblePolygons: number[][][]
    autoDirections: string[]
    textMask?: string  // 文字检测掩膜
    textlinesPerBubble: BubbleTextline[][]
    originalTexts?: string[]
    bubbleStates: BubbleState[]
}

function createBubbleStatesFromDetection(
    image: AppImageData,
    result: {
        bubbleCoords: BubbleCoords[]
        bubbleAngles: number[]
        autoDirections: string[]
        textlinesPerBubble: BubbleTextline[][]
    }
): BubbleState[] {
    const settingsStore = useSettingsStore()
    const { textStyle } = settingsStore.settings

    return result.bubbleCoords.map((coords, index) => {
        const autoDirection = result.autoDirections[index] === 'h'
            ? 'horizontal'
            : result.autoDirections[index] === 'v'
                ? 'vertical'
                : (coords[3] - coords[1]) > (coords[2] - coords[0])
                    ? 'vertical'
                    : 'horizontal'

        const textDirection = textStyle.layoutDirection === 'vertical' || textStyle.layoutDirection === 'horizontal'
            ? textStyle.layoutDirection
            : autoDirection

        return createBubbleState({
            coords,
            polygon: [],
            originalText: image.originalTexts?.[index] || '',
            translatedText: image.bubbleTexts?.[index] || '',
            textboxText: image.textboxTexts?.[index] || '',
            rotationAngle: result.bubbleAngles[index] || 0,
            textDirection,
            autoTextDirection: autoDirection,
            textlines: result.textlinesPerBubble[index] || [],
            fontSize: textStyle.fontSize,
            fontFamily: textStyle.fontFamily,
            textColor: textStyle.textColor,
            fillColor: textStyle.fillColor,
            strokeEnabled: textStyle.strokeEnabled,
            strokeColor: textStyle.strokeColor,
            strokeWidth: textStyle.strokeWidth,
            lineSpacing: textStyle.lineSpacing,
            textAlign: textStyle.textAlign,
            inpaintMethod: textStyle.inpaintMethod
        })
    })
}

export async function executeDetection(input: DetectionInput): Promise<DetectionOutput> {
    const { imageIndex, image, forceDetect = false } = input
    const settingsStore = useSettingsStore()

    // 如果图片已有 bubbleStates 数据（包括空数组），跳过检测
    // - bubbleStates === null/undefined: 从未处理过，需要自动检测
    // - bubbleStates === []: 用户主动清空，跳过检测（避免"框复活"）
    // - bubbleStates.length > 0: 有气泡数据，复用已有数据
    const existingBubbles = image.bubbleStates
    if (!forceDetect && existingBubbles !== null && existingBubbles !== undefined) {
        if (existingBubbles.length > 0) {
            console.log(`图片 ${imageIndex + 1} 已有 ${existingBubbles.length} 个气泡，跳过检测`)
            // 坐标需要转换为整数，后端 numpy 切片需要整数索引
            return {
                bubbleCoords: existingBubbles.map(s =>
                    s.coords.map(c => Math.round(c)) as BubbleCoords
                ),
                bubbleAngles: existingBubbles.map(s => s.rotationAngle || 0),
                bubblePolygons: existingBubbles.map(s => s.polygon || []),
                autoDirections: existingBubbles.map(s => s.autoTextDirection || s.textDirection || 'vertical'),
                textMask: image.textMask ?? undefined,  // 从持久化数据中获取掩膜
                textlinesPerBubble: existingBubbles.map((bubble, index) =>
                    bubble.textlines && bubble.textlines.length > 0
                        ? bubble.textlines
                        : image.textlinesPerBubble?.[index] || []
                ),
                originalTexts: existingBubbles.map(s => s.originalText || ''),
                bubbleStates: existingBubbles
            }
        } else {
            console.log(`图片 ${imageIndex + 1} 气泡已被清空，跳过检测`)
            return {
                bubbleCoords: [],
                bubbleAngles: [],
                bubblePolygons: [],
                autoDirections: [],
                textMask: undefined,
                textlinesPerBubble: [],
                originalTexts: [],
                bubbleStates: []
            }
        }
    }

    const settings = settingsStore.settings
    const base64 = extractBase64(image.originalDataURL)

    // 步骤1: 使用用户选择的检测器进行检测（获取文本框）
    const response: ParallelDetectResponse = await parallelDetect({
        image: base64,
        detector_type: settings.textDetector,
        enable_aux_yolo_detection: settings.enableAuxYoloDetection,
        aux_yolo_conf_threshold: settings.auxYoloConfThreshold,
        aux_yolo_overlap_threshold: settings.auxYoloOverlapThreshold,
        enable_saber_yolo_refine: settings.enableSaberYoloRefine,
        saber_yolo_refine_overlap_threshold: settings.saberYoloRefineOverlapThreshold,
        box_expand_ratio: settings.boxExpand.ratio,
        box_expand_top: settings.boxExpand.top,
        box_expand_bottom: settings.boxExpand.bottom,
        box_expand_left: settings.boxExpand.left,
        box_expand_right: settings.boxExpand.right
    })

    if (!response.success) {
        throw new Error(response.error || '检测失败')
    }

    // 步骤2: 固定使用 Default 检测器生成精确文字掩膜
    // 无论用户选择哪个检测器，都统一使用 Default 生成掩膜
    // 这样所有检测器都能享受精确掩膜的好处
    let textMaskData: string | undefined = undefined

    console.log(`使用 Default 检测器生成精确文字掩膜...`)
    try {
        const maskResponse: ParallelDetectResponse = await parallelDetect({
            image: base64,
            detector_type: 'default',  // 固定使用 Default 检测器生成掩膜
            enable_aux_yolo_detection: false,  // 掩膜路径不需要辅助检测
            enable_saber_yolo_refine: false,  // 掩膜路径不需要二阶段纠错
            box_expand_ratio: 0,       // 掩膜生成不需要扩展
            box_expand_top: 0,
            box_expand_bottom: 0,
            box_expand_left: 0,
            box_expand_right: 0
        })

        if (maskResponse.success && maskResponse.raw_mask) {
            textMaskData = maskResponse.raw_mask
            console.log(`✅ 精确文字掩膜生成成功`)
        } else {
            console.warn(`⚠️ Default 检测器未能生成掩膜`)
        }
    } catch (error) {
        console.error(`❌ 生成精确文字掩膜失败:`, error)
        // 掩膜生成失败不影响主流程，继续使用检测结果
    }

    const detectionResult = {
        bubbleCoords: (response.bubble_coords || []) as BubbleCoords[],
        bubbleAngles: response.bubble_angles || [],
        bubblePolygons: response.bubble_polygons || [],
        autoDirections: response.auto_directions || [],
        textMask: textMaskData,  // 返回生成的精确掩膜
        textlinesPerBubble: response.textlines_per_bubble || []
    }
    return {
        ...detectionResult,
        bubbleStates: createBubbleStatesFromDetection(image, detectionResult)
    }
}

function extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
        return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
}

/**
 * 统一保存检测结果到 ImageData
 * 确保所有检测结果字段都被正确保存，避免遗漏
 */
export function saveDetectionResultToImage(
    imageIndex: number,
    result: DetectionOutput,
    options?: {
        /** 是否同时更新 bubbleStates（编辑模式需要） */
        updateBubbleStates?: boolean
        /** bubbleStates 数据（如果需要更新） */
        bubbleStates?: any[]
    }
): void {
    const imageStore = useImageStore()

    const updateData: Record<string, any> = {
        bubbleCoords: result.bubbleCoords,
        bubbleAngles: result.bubbleAngles,
        textMask: result.textMask || null,  // 精确文字掩膜
        textlinesPerBubble: result.textlinesPerBubble || [],
    }

    // 如果提供了 bubbleStates，一起更新
    if (options?.updateBubbleStates && options.bubbleStates) {
        updateData.bubbleStates = options.bubbleStates.map((state, index) => ({
            ...state,
            textlines: state.textlines && state.textlines.length > 0
                ? state.textlines
                : (result.textlinesPerBubble[index] || [])
        }))
    } else if (result.bubbleStates.length > 0) {
        updateData.bubbleStates = result.bubbleStates
    }

    imageStore.updateImageByIndex(imageIndex, updateData)
}
