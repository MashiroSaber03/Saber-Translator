/**
 * 气泡操作组合式函数
 * 管理气泡的选择、拖拽、调整大小、旋转、OCR识别、背景修复等操作
 * 便于后续频繁修改气泡逻辑
 */

import { onUnmounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { normalizeProviderId } from '@/config/aiProviders'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'
import { ocrSingleBubble as ocrSingleBubbleApi, inpaintSingleBubble as inpaintSingleBubbleApi } from '@/api/translate'
import { showToast } from '@/utils/toast'
import type { BubbleState, BubbleCoords } from '@/types/bubble'

// ============================================================
// 类型定义
// ============================================================

export interface BubbleActionCallbacks {
  /** 触发重新渲染 */
  onReRender?: () => void | Promise<unknown>
  /** 触发延迟渲染预览，用于防抖后的实时预览。 */
  onDelayedPreview?: () => void | Promise<unknown>
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 确保坐标为整数格式（后端 OpenCV 需要整数坐标）
 * 后端渲染接口要求整数坐标，这里作为防御性转换
 */
function normalizeCoords(coords: BubbleCoords): BubbleCoords {
  return [
    Math.round(coords[0]),
    Math.round(coords[1]),
    Math.round(coords[2]),
    Math.round(coords[3])
  ]
}

// ============================================================
// 组合式函数
// ============================================================

export function useBubbleActions(callbacks?: BubbleActionCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  const {
    bubbles,
    selectedIndex: selectedBubbleIndex,
    hasSelection
  } = storeToRefs(bubbleStore)

  const { currentImage } = storeToRefs(imageStore)

  // ============================================================
  // 绘制模式状态
  // ============================================================

  /** 是否处于绘制模式 */
  const isDrawingMode = ref(false)

  /** 是否正在绘制新框 */
  const isDrawingBox = ref(false)

  /** 当前绘制的临时矩形 */
  const currentDrawingRect = ref<BubbleCoords | null>(null)

  /** 是否中键按下 */
  const isMiddleButtonDown = ref(false)

  // ============================================================
  // 气泡选择操作
  // ============================================================

  /** 处理气泡选择 */
  function handleBubbleSelect(index: number): void {
    bubbleStore.selectBubble(index)
  }

  /** 处理气泡多选 */
  function handleBubbleMultiSelect(index: number): void {
    bubbleStore.toggleMultiSelect(index)
  }

  /** 处理清除多选 */
  function handleClearMultiSelect(): void {
    bubbleStore.clearMultiSelect()
  }

  // ============================================================
  // 气泡拖拽操作
  // ============================================================

  /** 处理气泡拖动开始 */
  function handleBubbleDragStart(index: number, _event: MouseEvent): void {
    void index
  }

  /** 处理气泡拖动结束 */
  function handleBubbleDragEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    // 拖动结束后触发重新渲染
    triggerDelayedPreview()
  }

  // ============================================================
  // 气泡调整大小操作
  // ============================================================

  /** 处理气泡调整大小开始 */
  function handleBubbleResizeStart(index: number, handle: string, _event: MouseEvent): void {
    void index
    void handle
  }

  /** 处理气泡调整大小结束 */
  function handleBubbleResizeEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    // 调整大小结束后触发重新渲染
    triggerDelayedPreview()
  }

  // ============================================================
  // 气泡旋转操作
  // ============================================================

  /** 处理气泡旋转开始 */
  function handleBubbleRotateStart(index: number, _event: MouseEvent): void {
    void index
  }

  /** 处理气泡旋转结束 */
  function handleBubbleRotateEnd(index: number, angle: number): void {
    bubbleStore.updateBubble(index, { rotationAngle: angle })
    // 旋转结束后触发重新渲染
    triggerDelayedPreview()
  }

  // ============================================================
  // 气泡绘制操作
  // ============================================================

  /** 切换绘制模式 */
  function toggleDrawingMode(): void {
    isDrawingMode.value = !isDrawingMode.value
  }

  /** 处理绘制新气泡 */
  function handleDrawBubble(coords: BubbleCoords): void {
    bubbleStore.addBubble(coords)
    bubbleStore.selectBubble(bubbleStore.bubbleCount - 1)
    // 添加新气泡后触发重新渲染
    callbacks?.onReRender?.()
  }

  /** 获取绘制框样式 */
  function getDrawingRectStyle(): Record<string, string> {
    if (!currentDrawingRect.value) return {}
    const [x1, y1, x2, y2] = currentDrawingRect.value
    return {
      position: 'absolute',
      left: `${Math.min(x1, x2)}px`,
      top: `${Math.min(y1, y2)}px`,
      width: `${Math.abs(x2 - x1)}px`,
      height: `${Math.abs(y2 - y1)}px`
    }
  }

  // ============================================================
  // 延迟渲染机制
  // ============================================================

  /** 延迟渲染计时器 */
  let previewTimer: ReturnType<typeof setTimeout> | null = null
  /** 渲染状态锁，防止竞态条件 */
  let isRenderingPreview = false
  /** 渲染期间是否又收到了新的预览请求 */
  let previewRequestedWhileRendering = false
  /** 延迟时间（毫秒） */
  const PREVIEW_DELAY = 150

  /**
   * 触发延迟渲染预览。
   * 使用防抖机制避免频繁渲染，等待渲染 Promise 完成后才解锁。
   */
  function triggerDelayedPreview(): void {
    if (previewTimer) {
      clearTimeout(previewTimer)
    }
    previewTimer = setTimeout(async () => {
      previewTimer = null
      if (isRenderingPreview) {
        previewRequestedWhileRendering = true
        return
      }
      isRenderingPreview = true

      try {
        // 优先使用延迟预览回调，否则使用重新渲染回调。
        // 等待 Promise 完成后才释放锁。
        if (callbacks?.onDelayedPreview) {
          await callbacks.onDelayedPreview()
        } else if (callbacks?.onReRender) {
          await callbacks.onReRender()
        }
      } catch {
        showToast('预览渲染失败', 'error')
      } finally {
        // 渲染完成后才重置状态。
        isRenderingPreview = false
        if (previewRequestedWhileRendering) {
          previewRequestedWhileRendering = false
          triggerDelayedPreview()
        }
      }
    }, PREVIEW_DELAY)
  }

  function isSameCurrentImage(expectedImageId: string): boolean {
    return currentImage.value?.id === expectedImageId
  }

  function isSameBubbleTarget(expectedImageId: string, index: number, expectedBubble: BubbleState): boolean {
    return isSameCurrentImage(expectedImageId) && bubbles.value[index] === expectedBubble
  }

  function updateCurrentImageIfStillCurrent(
    expectedImageId: string,
    updates: Parameters<typeof imageStore.updateCurrentImage>[0]
  ): boolean {
    if (!isSameCurrentImage(expectedImageId)) {
      return false
    }
    imageStore.updateCurrentImage(updates)
    return true
  }

  onUnmounted(() => {
    if (previewTimer) {
      clearTimeout(previewTimer)
      previewTimer = null
    }
    isRenderingPreview = false
    previewRequestedWhileRendering = false
  })

  // ============================================================
  // 气泡编辑操作
  // ============================================================

  /** 处理气泡更新（带延迟渲染） */
  function handleBubbleUpdate(updates: Partial<BubbleState>): void {
    bubbleStore.updateSelectedBubble(updates)
    // 触发延迟渲染预览。
    triggerDelayedPreview()
  }

  /** 删除选中的气泡 */
  function deleteSelectedBubbles(): void {
    if (hasSelection.value) {
      bubbleStore.deleteSelected()
      // 删除后触发重新渲染。
      callbacks?.onReRender?.()
    }
  }

  /** 修复选中的气泡（支持LAMA或纯色填充） */
  async function repairSelectedBubble(): Promise<void> {
    const index = selectedBubbleIndex.value
    if (index < 0) {
      showToast('请先选中要修复的气泡框', 'warning')
      return
    }

    const bubble = bubbles.value[index]
    const image = currentImage.value
    if (!bubble || !image?.originalDataURL) {
      showToast('无法修复背景：缺少气泡或图片数据', 'warning')
      return
    }
    const expectedImageId = image.id

    // 获取修复方法和填充颜色
    const inpaintMethod = bubble.inpaintMethod || 'solid'
    const fillColor = bubble.fillColor || '#FFFFFF'
    const rotationAngle = bubble.rotationAngle || 0

    try {
      // 获取基础图像数据（优先使用cleanImageData保留之前的修复效果）
      let baseImageData: string
      if (image.cleanImageData) {
        baseImageData = image.cleanImageData
      } else {
        const match = image.originalDataURL.match(/^data:image\/[^;]+;base64,(.+)$/)
        baseImageData = match && match[1] ? match[1] : ''
        if (!baseImageData) {
          showToast('无法解析图像数据', 'error')
          return
        }
      }

      const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama'

      if (isLamaMethod) {
        // 根据当前修复方式确定 LAMA 模型类型。
        const lamaModel = inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe'

        // 确保坐标为整数（后端 OpenCV 需要）
        const coords = normalizeCoords(bubble.coords)

        // 使用LAMA修复（传递完整参数）
        const response = await inpaintSingleBubbleApi(
          baseImageData,
          coords,
          {
            bubbleAngle: rotationAngle,
            method: 'lama',
            lamaModel: lamaModel
          }
        )

        if (response.success && response.inpainted_image) {
          if (!updateCurrentImageIfStillCurrent(expectedImageId, { cleanImageData: response.inpainted_image })) {
            return
          }
          triggerDelayedPreview()
        } else {
          showToast('LAMA 修复失败，已使用纯色填充', 'warning')
          const applied = await fillBubbleWithColor(bubble.coords, fillColor, rotationAngle, expectedImageId)
          if (applied) {
            triggerDelayedPreview()
          }
        }
      } else {
        // 使用纯色填充
        const applied = await fillBubbleWithColor(bubble.coords, fillColor, rotationAngle, expectedImageId)
        if (applied) {
          triggerDelayedPreview()
        }
      }
    } catch (error) {
      if (!isSameCurrentImage(expectedImageId)) {
        return
      }
      const errorMessage = error instanceof Error ? error.message : '背景修复失败'
      showToast(errorMessage, 'error')
    }
  }

  /** 使用纯色填充气泡区域 */
  async function fillBubbleWithColor(
    coords: [number, number, number, number],
    fillColor: string,
    rotationAngle: number = 0,
    expectedImageId?: string
  ): Promise<boolean> {
    const image = currentImage.value
    if (!image) return false

    const [x1, y1, x2, y2] = coords

    // 获取基础图像
    let baseSrc: string
    if (image.cleanImageData) {
      baseSrc = 'data:image/png;base64,' + image.cleanImageData
    } else if (image.originalDataURL) {
      baseSrc = image.originalDataURL
    } else {
      showToast('无法找到基础图像用于填充', 'error')
      return false
    }

    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          resolve(false)
          return
        }

        // 绘制基础图像
        ctx.drawImage(img, 0, 0)

        // 用填充色填充指定气泡区域
        ctx.fillStyle = fillColor

        if (Math.abs(rotationAngle) < 0.1) {
          // 无旋转，使用简单矩形
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
        } else {
          // 有旋转，绘制旋转后的多边形
          const cx = (x1 + x2) / 2
          const cy = (y1 + y2) / 2
          const hw = (x2 - x1) / 2
          const hh = (y2 - y1) / 2
          const rad = rotationAngle * Math.PI / 180
          const cos_a = Math.cos(rad)
          const sin_a = Math.sin(rad)

          // 计算旋转后的四个角点
          const corners: [number, number][] = [
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
          ].map(([dx, dy]): [number, number] => [
            cx + (dx as number) * cos_a - (dy as number) * sin_a,
            cy + (dx as number) * sin_a + (dy as number) * cos_a
          ])

          ctx.beginPath()
          const firstCorner = corners[0]
          if (firstCorner) {
            ctx.moveTo(firstCorner[0], firstCorner[1])
            for (let i = 1; i < corners.length; i++) {
              const corner = corners[i]
              if (corner) {
                ctx.lineTo(corner[0], corner[1])
              }
            }
          }
          ctx.closePath()
          ctx.fill()
        }

        // 更新cleanImageData
        const newCleanData = canvas.toDataURL('image/png').split(',')[1]
        if (expectedImageId) {
          if (!updateCurrentImageIfStillCurrent(expectedImageId, { cleanImageData: newCleanData })) {
            resolve(false)
            return
          }
        } else {
          imageStore.updateCurrentImage({ cleanImageData: newCleanData })
        }

        resolve(true)
      }
      img.onerror = () => {
        showToast('加载基础图像失败', 'error')
        resolve(false)
      }
      img.src = baseSrc
    })
  }

  // ============================================================
  // OCR 识别操作
  // ============================================================

  /** 处理单气泡 OCR 重新识别 */
  async function handleOcrRecognize(index: number): Promise<void> {
    const bubble = bubbles.value[index]
    const image = currentImage.value
    if (!bubble || !image?.originalDataURL) {
      showToast('无法进行 OCR 识别：缺少气泡或图片数据', 'warning')
      return
    }
    const expectedImageId = image.id
    const expectedBubble = bubble

    try {
      const imageData = image.originalDataURL.split(',')[1] || ''
      const settings = settingsStore.settings
      const bubbleTextlines = bubble.textlines?.length
        ? bubble.textlines
        : (Array.isArray(image.textlinesPerBubble) ? image.textlinesPerBubble[index] || [] : [])
      // PaddleOCR-VL 使用独立的源语言设置
      const ocrSourceLanguage = settings.ocrEngine === 'paddleocr_vl'
        ? settings.paddleOcrVl?.sourceLanguage || 'japanese'
        : settings.sourceLanguage
      const response = await ocrSingleBubbleApi(
        imageData,
        bubble.coords,
        settings.ocrEngine || 'manga_ocr',
        {
          source_language: ocrSourceLanguage,
          // 百度 OCR 请求参数
          baidu_ocr_api_key: settings.baiduOcr.apiKey,
          baidu_ocr_secret_key: settings.baiduOcr.secretKey,
          baidu_version: settings.baiduOcr.version,
          baidu_source_language: settings.baiduOcr.sourceLanguage,
          // AI 视觉 OCR 请求参数
          ai_vision_provider: normalizeProviderId(settings.aiVisionOcr.provider),
          ai_vision_api_key: settings.aiVisionOcr.apiKey,
          ai_vision_model_name: settings.aiVisionOcr.modelName,
          ai_vision_ocr_prompt: settings.aiVisionOcr.prompt,
          ai_vision_prompt_mode: settings.aiVisionOcr.promptMode,
          custom_ai_vision_base_url: settings.aiVisionOcr.customBaseUrl,
          openai_options: serializeOpenAICompatibleOptionsForApi(settings.aiVisionOcr.openaiOptions),
          ai_vision_min_image_size: settings.aiVisionOcr.minImageSize,
          enable_hybrid_ocr: settings.hybridOcr.enabled,
          secondary_ocr_engine: settings.hybridOcr.secondaryEngine,
          hybrid_ocr_threshold: settings.hybridOcr.confidenceThreshold,
          bubble_textlines: bubbleTextlines,
          text_detector: settings.textDetector,
          enable_aux_yolo_detection: settings.enableAuxYoloDetection,
          aux_yolo_conf_threshold: settings.auxYoloConfThreshold,
          aux_yolo_overlap_threshold: settings.auxYoloOverlapThreshold,
          enable_saber_yolo_refine: settings.enableSaberYoloRefine,
          saber_yolo_refine_overlap_threshold: settings.saberYoloRefineOverlapThreshold
        }
      )

      if (response.success && response.text !== undefined) {
        if (!isSameBubbleTarget(expectedImageId, index, expectedBubble)) {
          return
        }
        bubbleStore.updateBubble(index, {
          originalText: response.text,
          textlines: response.textlines || bubbleTextlines,
          ocrResult: response.ocr_result || null
        })
      } else {
        if (!isSameCurrentImage(expectedImageId)) {
          return
        }
        const errorMsg = response.error || '识别失败'
        showToast(errorMsg, 'error')
      }
    } catch (error) {
      if (!isSameCurrentImage(expectedImageId)) {
        return
      }
      const errorMessage = error instanceof Error ? error.message : 'OCR 识别出错'
      showToast(errorMessage, 'error')
    }
  }

  // ============================================================
  // 返回接口
  // ============================================================

  return {
    // 绘制模式状态
    isDrawingMode,
    isDrawingBox,
    currentDrawingRect,
    isMiddleButtonDown,

    // 气泡选择
    handleBubbleSelect,
    handleBubbleMultiSelect,
    handleClearMultiSelect,

    // 气泡拖拽
    handleBubbleDragStart,
    handleBubbleDragEnd,

    // 气泡调整大小
    handleBubbleResizeStart,
    handleBubbleResizeEnd,

    // 气泡旋转
    handleBubbleRotateStart,
    handleBubbleRotateEnd,

    // 气泡绘制
    toggleDrawingMode,
    handleDrawBubble,
    getDrawingRectStyle,

    // 气泡编辑
    handleBubbleUpdate,
    deleteSelectedBubbles,
    repairSelectedBubble,

    // 延迟渲染
    triggerDelayedPreview,

    // OCR 识别
    handleOcrRecognize
  }
}
