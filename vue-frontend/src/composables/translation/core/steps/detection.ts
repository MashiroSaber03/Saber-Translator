/**
 * 检测步骤
 * 提取自 SequentialPipeline.ts Line 234-287
 */
import { parallelDetect, type ParallelDetectResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleCoords } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'

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
    textlinesPerBubble: any[]
    originalTexts?: string[]
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
                textlinesPerBubble: [],
                originalTexts: existingBubbles.map(s => s.originalText || '')
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
                originalTexts: []
            }
        }
    }

    const settings = settingsStore.settings
    const base64 = extractBase64(image.originalDataURL)

    // 步骤1: 使用用户选择的检测器进行检测（获取文本框）
    const response: ParallelDetectResponse = await parallelDetect({
        image: base64,
        detector_type: settings.textDetector,
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

    return {
        bubbleCoords: (response.bubble_coords || []) as BubbleCoords[],
        bubbleAngles: response.bubble_angles || [],
        bubblePolygons: response.bubble_polygons || [],
        autoDirections: response.auto_directions || [],
        textMask: textMaskData,  // 返回生成的精确掩膜
        textlinesPerBubble: response.textlines_per_bubble || []
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
    }

    // 如果提供了 bubbleStates，一起更新
    if (options?.updateBubbleStates && options.bubbleStates) {
        updateData.bubbleStates = options.bubbleStates
    }

    imageStore.updateImageByIndex(imageIndex, updateData)
}
