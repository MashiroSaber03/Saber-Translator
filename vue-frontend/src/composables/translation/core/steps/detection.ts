import { parallelDetect, type ParallelDetectResponse } from '@/api/parallelTranslate'
import type {
    BubbleApiResponse,
    BubbleCoords,
    BubbleGlobalDefaults,
    BubbleState,
    BubbleTextline,
} from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'
import type { TranslationSettings } from '@/types/settings'
import { createBubbleStatesFromResponse, detectTextDirection } from '@/utils/bubbleFactory'
import { extractBase64Payload } from '@/utils/dataUrl'

export interface DetectionInput {
    imageIndex: number
    image: AppImageData
    translationMode?: string
    forceDetect?: boolean
    settingsSnapshot: TranslationSettings
}

export interface DetectionOutput {
    bubbleCoords: BubbleCoords[]
    bubbleAngles: number[]
    bubblePolygons: number[][][]
    autoDirections: string[]
    textMask?: string
    textlinesPerBubble: BubbleTextline[][]
    originalTexts?: string[]
    bubbleStates: BubbleState[]
}

function normalizeAutoDirection(direction: string | undefined, coords: BubbleCoords): 'v' | 'h' {
    if (direction === 'v' || direction === 'h') {
        return direction
    }
    return detectTextDirection(coords) === 'vertical' ? 'v' : 'h'
}

function buildDetectionBubbleResponse(
    image: AppImageData,
    result: {
        bubbleCoords: BubbleCoords[]
        bubbleAngles: number[]
        autoDirections: string[]
        textlinesPerBubble: BubbleTextline[][]
    },
): BubbleApiResponse {
    return {
        bubble_coords: result.bubbleCoords,
        bubble_angles: result.bubbleAngles,
        auto_directions: result.bubbleCoords.map((coords, index) =>
            normalizeAutoDirection(result.autoDirections[index], coords)
        ),
        textlines_per_bubble: result.textlinesPerBubble,
        original_texts: image.originalTexts || [],
        bubble_texts: image.bubbleTexts || [],
        textbox_texts: image.textboxTexts || [],
    }
}

function buildDetectionBubbleDefaults(settingsSnapshot: TranslationSettings): BubbleGlobalDefaults {
    const textStyle = settingsSnapshot.textStyle

    return {
        fontSize: textStyle.fontSize,
        fontFamily: textStyle.fontFamily,
        textDirection: textStyle.layoutDirection,
        textColor: textStyle.textColor,
        fillColor: textStyle.fillColor,
        strokeEnabled: textStyle.strokeEnabled,
        strokeColor: textStyle.strokeColor,
        strokeWidth: textStyle.strokeWidth,
        lineSpacing: textStyle.lineSpacing,
        textAlign: textStyle.textAlign,
        inpaintMethod: textStyle.inpaintMethod,
    }
}

export async function executeDetection(input: DetectionInput): Promise<DetectionOutput> {
    const { image, translationMode = 'standard', forceDetect = false, settingsSnapshot } = input

    // A persisted empty array is a user-cleared state, not a missing detection result.
    const existingBubbles = image.bubbleStates
    if (!forceDetect && existingBubbles !== null && existingBubbles !== undefined) {
        if (existingBubbles.length > 0) {
            return {
                bubbleCoords: existingBubbles.map(s =>
                    s.coords.map(c => Math.round(c)) as BubbleCoords
                ),
                bubbleAngles: existingBubbles.map(s => s.rotationAngle || 0),
                bubblePolygons: existingBubbles.map(s => s.polygon || []),
                autoDirections: existingBubbles.map(s => s.autoTextDirection || s.textDirection || 'vertical'),
                textMask: image.textMask ?? undefined,
                textlinesPerBubble: existingBubbles.map((bubble, index) =>
                    bubble.textlines && bubble.textlines.length > 0
                        ? bubble.textlines
                        : image.textlinesPerBubble?.[index] || []
                ),
                originalTexts: existingBubbles.map(s => s.originalText || ''),
                bubbleStates: existingBubbles
            }
        } else {
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

    const settings = settingsSnapshot
    const base64 = extractBase64Payload(image.originalDataURL)

    const response: ParallelDetectResponse = await parallelDetect({
        image: base64,
        translation_mode: translationMode,
        translation_scope: 'image',
        detector_type: settings.textDetector,
        min_text_block_area_percent: settings.minTextBlockAreaPercent,
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

    let textMaskData: string | undefined = undefined

    try {
        const maskResponse: ParallelDetectResponse = await parallelDetect({
            image: base64,
            translation_mode: translationMode,
            translation_scope: 'image',
            detector_type: 'default',
            enable_aux_yolo_detection: false,
            enable_saber_yolo_refine: false,
            box_expand_ratio: 0,
            box_expand_top: 0,
            box_expand_bottom: 0,
            box_expand_left: 0,
            box_expand_right: 0
        })

        if (maskResponse.success && maskResponse.raw_mask) {
            textMaskData = maskResponse.raw_mask
        }
    } catch {
        // Mask generation is best-effort; detection results remain usable without it.
    }

    const detectionResult = {
        bubbleCoords: (response.bubble_coords || []) as BubbleCoords[],
        bubbleAngles: response.bubble_angles || [],
        bubblePolygons: response.bubble_polygons || [],
        autoDirections: response.auto_directions || [],
        textMask: textMaskData,
        textlinesPerBubble: response.textlines_per_bubble || []
    }
    return {
        ...detectionResult,
        bubbleStates: createBubbleStatesFromResponse(
            buildDetectionBubbleResponse(image, detectionResult),
            buildDetectionBubbleDefaults(settingsSnapshot)
        )
    }
}
