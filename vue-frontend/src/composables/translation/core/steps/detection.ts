/**
 * 检测步骤
 * 提取自 SequentialPipeline.ts Line 234-287
 */
import { parallelDetect, type ParallelDetectResponse } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
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
    rawMask?: string
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
                textlinesPerBubble: [],
                originalTexts: []
            }
        }
    }

    const settings = settingsStore.settings
    const base64 = extractBase64(image.originalDataURL)

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

    return {
        bubbleCoords: (response.bubble_coords || []) as BubbleCoords[],
        bubbleAngles: response.bubble_angles || [],
        bubblePolygons: response.bubble_polygons || [],
        autoDirections: response.auto_directions || [],
        rawMask: response.raw_mask,
        textlinesPerBubble: response.textlines_per_bubble || []
    }
}

function extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
        return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
}
