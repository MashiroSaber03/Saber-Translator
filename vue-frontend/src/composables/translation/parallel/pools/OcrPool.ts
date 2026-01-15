/**
 * OCR池
 * 
 * 负责调用后端OCR API，识别气泡中的文字
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { parallelOcr } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'
import { DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE } from '@/constants'

export class OcrPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('OCR', '📖', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const { imageData, detectionResult } = task
    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings

    if (!detectionResult || detectionResult.bubbleCoords.length === 0) {
      // 没有检测到气泡，直接跳过
      task.ocrResult = {
        originalTexts: [],
        textlinesPerBubble: []
      }
      task.status = 'processing'
      return task
    }

    const base64 = this.extractBase64(imageData.originalDataURL)

    // 调用后端OCR API
    // PaddleOCR-VL 使用独立的源语言设置
    const ocrSourceLanguage = settings.ocrEngine === 'paddleocr_vl' 
      ? settings.paddleOcrVl?.sourceLanguage || 'japanese'
      : settings.sourceLanguage
    
    const response = await parallelOcr({
      image: base64,
      bubble_coords: detectionResult.bubbleCoords,
      source_language: ocrSourceLanguage,
      ocr_engine: settings.ocrEngine,
      baidu_api_key: settings.baiduOcr?.apiKey,
      baidu_secret_key: settings.baiduOcr?.secretKey,
      baidu_version: settings.baiduOcr?.version,
      baidu_ocr_language: settings.baiduOcr?.sourceLanguage,
      ai_vision_provider: settings.aiVisionOcr?.provider,
      ai_vision_api_key: settings.aiVisionOcr?.apiKey,
      ai_vision_model_name: settings.aiVisionOcr?.modelName,
      ai_vision_ocr_prompt: settings.aiVisionOcr?.prompt,
      custom_ai_vision_base_url: settings.aiVisionOcr?.customBaseUrl,
      ai_vision_min_image_size: settings.aiVisionOcr?.minImageSize ?? DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE,
      textlines_per_bubble: detectionResult.textlinesPerBubble
    })

    if (!response.success) {
      throw new Error(response.error || 'OCR失败')
    }

    task.ocrResult = {
      originalTexts: response.original_texts || [],
      textlinesPerBubble: response.textlines_per_bubble || []
    }

    task.status = 'processing'
    return task
  }

  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
  }
}
