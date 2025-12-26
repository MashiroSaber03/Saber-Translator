/**
 * 气泡操作组合式函数
 * 管理气泡的选择、拖拽、调整大小、旋转、OCR识别、背景修复等操作
 * 便于后续频繁修改气泡逻辑
 */

import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { ocrSingleBubble as ocrSingleBubbleApi, inpaintSingleBubble as inpaintSingleBubbleApi } from '@/api/translate'
import type { BubbleState, BubbleCoords } from '@/types/bubble'

// ============================================================
// 类型定义
// ============================================================

export interface BubbleActionCallbacks {
  /** 触发重新渲染 */
  onReRender?: () => void | Promise<unknown>
  /** 【修复问题5】触发延迟渲染预览（用于实时预览，有防抖），支持返回Promise */
  onDelayedPreview?: () => void | Promise<unknown>
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 确保坐标为整数格式（后端 OpenCV 需要整数坐标）
 * 【复刻原版】原版坐标始终是整数，这里作为防御性转换
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

  /** 绘制起始点 */
  const drawStartX = ref(0)
  const drawStartY = ref(0)

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
    console.log(`开始拖动气泡 #${index + 1}`)
  }

  /** 处理气泡拖动中 */
  function handleBubbleDragging(_deltaX: number, _deltaY: number): void {
    // 视觉反馈由BubbleOverlay内部处理，这里不更新store
    // 只在dragEnd时更新最终坐标
  }

  /** 处理气泡拖动结束 */
  function handleBubbleDragEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    console.log(`气泡 #${index + 1} 拖动完成:`, newCoords)
    // 拖动结束后触发重新渲染
    triggerDelayedPreview()
  }

  // ============================================================
  // 气泡调整大小操作
  // ============================================================

  /** 处理气泡调整大小开始 */
  function handleBubbleResizeStart(index: number, handle: string, _event: MouseEvent): void {
    console.log(`开始调整气泡 #${index + 1} 大小，手柄: ${handle}`)
  }

  /** 处理气泡调整大小中 */
  // 【复刻原版】调整大小过程中只更新视觉显示（通过 resizeCurrentCoords），不更新实际坐标数据
  // 原版 edit_mode.js 在调整过程中只更新 CSS，完成时才调用 updateBubbleCoords
  // 这样可以避免存储浮点数坐标，因为只有 finishResizing 中有 Math.round() 转换
  function handleBubbleResizing(_newCoords: BubbleCoords): void {
    // 不在过程中更新 bubbleStore，视觉反馈由 BubbleOverlay 的 resizeCurrentCoords 处理
  }

  /** 处理气泡调整大小结束 */
  function handleBubbleResizeEnd(index: number, newCoords: BubbleCoords): void {
    bubbleStore.updateBubble(index, { coords: newCoords })
    console.log(`气泡 #${index + 1} 调整大小完成:`, newCoords)
    // 调整大小结束后触发重新渲染
    triggerDelayedPreview()
  }

  // ============================================================
  // 气泡旋转操作
  // ============================================================

  /** 处理气泡旋转开始 */
  function handleBubbleRotateStart(index: number, _event: MouseEvent): void {
    console.log(`开始旋转气泡 #${index + 1}`)
  }

  /** 处理气泡旋转中 */
  // 【复刻原版】旋转过程中只更新视觉显示（通过 rotateCurrentAngle），不更新实际角度数据
  // 原版 edit_mode.js 在旋转结束 handleRotateMouseUp 时才调用 updateSingleBubbleState
  function handleBubbleRotating(_angle: number): void {
    // 不在过程中更新 bubbleStore，视觉反馈由 BubbleOverlay 的 rotateCurrentAngle 处理
  }

  /** 处理气泡旋转结束 */
  function handleBubbleRotateEnd(index: number, angle: number): void {
    bubbleStore.updateBubble(index, { rotationAngle: angle })
    console.log(`气泡 #${index + 1} 旋转完成: ${angle}°`)
    // 旋转结束后触发重新渲染
    triggerDelayedPreview()
  }

  // ============================================================
  // 气泡绘制操作
  // ============================================================

  /** 切换绘制模式 */
  function toggleDrawingMode(): void {
    isDrawingMode.value = !isDrawingMode.value
    if (isDrawingMode.value) {
      console.log('进入绘制模式')
    } else {
      console.log('退出绘制模式')
    }
  }

  /** 处理绘制新气泡 */
  function handleDrawBubble(coords: BubbleCoords): void {
    bubbleStore.addBubble(coords)
    bubbleStore.selectBubble(bubbleStore.bubbleCount - 1)
    console.log('已添加新气泡:', coords)
    // 添加新气泡后触发重新渲染
    callbacks?.onReRender?.()
  }

  /** 开始绘制气泡框 */
  function startDrawingBox(x: number, y: number): void {
    isDrawingBox.value = true
    drawStartX.value = x
    drawStartY.value = y
    currentDrawingRect.value = [x, y, x, y]
  }

  /** 更新绘制中的气泡框 */
  function updateDrawingBox(x: number, y: number): void {
    if (!isDrawingBox.value) return
    currentDrawingRect.value = [
      drawStartX.value,
      drawStartY.value,
      x,
      y
    ]
  }

  /** 完成绘制气泡框 */
  function finishDrawingBox(): BubbleCoords | null {
    if (!isDrawingBox.value || !currentDrawingRect.value) {
      isDrawingBox.value = false
      currentDrawingRect.value = null
      return null
    }

    const [x1, y1, x2, y2] = currentDrawingRect.value
    const minSize = 10

    // 检查最小尺寸
    if (Math.abs(x2 - x1) < minSize || Math.abs(y2 - y1) < minSize) {
      isDrawingBox.value = false
      currentDrawingRect.value = null
      return null
    }

    // 规范化坐标（确保x1<x2, y1<y2）
    const normalizedCoords: BubbleCoords = [
      Math.min(x1, x2),
      Math.min(y1, y2),
      Math.max(x1, x2),
      Math.max(y1, y2)
    ]

    isDrawingBox.value = false
    currentDrawingRect.value = null

    return normalizedCoords
  }

  /** 取消绘制 */
  function cancelDrawing(): void {
    isDrawingBox.value = false
    currentDrawingRect.value = null
    isDrawingMode.value = false
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
      height: `${Math.abs(y2 - y1)}px`,
      border: '2px dashed #00d4ff',
      background: 'rgba(0, 212, 255, 0.1)',
      pointerEvents: 'none',
      zIndex: '25',
      boxSizing: 'border-box'
    }
  }

  // ============================================================
  // 延迟渲染机制（类似原版的triggerDelayedPreview）
  // ============================================================

  /** 延迟渲染计时器 */
  let previewTimer: ReturnType<typeof setTimeout> | null = null
  /** 渲染状态锁，防止竞态条件 */
  let isRenderingPreview = false
  /** 延迟时间（毫秒） */
  const PREVIEW_DELAY = 150

  /**
   * 【修复问题5】触发延迟渲染预览
   * 使用防抖机制避免频繁渲染，等待渲染Promise完成后才解锁
   */
  function triggerDelayedPreview(): void {
    if (previewTimer) {
      clearTimeout(previewTimer)
    }
    previewTimer = setTimeout(async () => {
      if (isRenderingPreview) {
        console.log('跳过渲染，上一次渲染仍在进行中')
        return
      }
      console.log('触发延迟渲染预览')
      isRenderingPreview = true

      try {
        // 【修复问题5】优先使用延迟预览回调，否则使用重新渲染回调
        // 等待Promise完成后才释放锁
        if (callbacks?.onDelayedPreview) {
          await callbacks.onDelayedPreview()
        } else if (callbacks?.onReRender) {
          await callbacks.onReRender()
        }
      } catch (error) {
        console.error('延迟渲染预览失败:', error)
      } finally {
        // 【修复问题5】渲染完成后才重置状态
        isRenderingPreview = false
      }
    }, PREVIEW_DELAY)
  }

  // ============================================================
  // 气泡编辑操作
  // ============================================================

  /** 处理气泡更新（带延迟渲染） */
  function handleBubbleUpdate(updates: Partial<BubbleState>): void {
    bubbleStore.updateSelectedBubble(updates)
    // 触发延迟渲染预览（类似原版的triggerDelayedPreview）
    triggerDelayedPreview()
  }

  /** 删除选中的气泡 */
  function deleteSelectedBubbles(): void {
    if (hasSelection.value) {
      bubbleStore.deleteSelected()
      console.log('已删除选中的气泡')
      // 删除后触发重新渲染（原版逻辑）
      callbacks?.onReRender?.()
    }
  }

  /** 修复选中的气泡（支持LAMA或纯色填充） */
  async function repairSelectedBubble(): Promise<void> {
    const index = selectedBubbleIndex.value
    if (index < 0) {
      console.warn('请先选中要修复的气泡框')
      return
    }

    const bubble = bubbles.value[index]
    const image = currentImage.value
    if (!bubble || !image?.originalDataURL) {
      console.warn('无法修复背景：缺少气泡或图片数据')
      return
    }

    // 获取修复方法和填充颜色
    const inpaintMethod = bubble.inpaintMethod || 'solid'
    const fillColor = bubble.fillColor || '#FFFFFF'
    const rotationAngle = bubble.rotationAngle || 0

    try {
      console.log(`开始修复气泡 #${index + 1} 背景，方法: ${inpaintMethod}`)

      // 获取基础图像数据（优先使用cleanImageData保留之前的修复效果）
      let baseImageData: string
      if (image.cleanImageData) {
        baseImageData = image.cleanImageData
      } else {
        const match = image.originalDataURL.match(/^data:image\/[^;]+;base64,(.+)$/)
        baseImageData = match && match[1] ? match[1] : ''
        if (!baseImageData) {
          console.error('无法解析图像数据')
          return
        }
      }

      const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama'

      if (isLamaMethod) {
        // 确定 LAMA 模型类型（与原版逻辑一致）
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
          imageStore.updateCurrentImage({ cleanImageData: response.inpainted_image })
          console.log(`气泡 #${index + 1} LAMA背景修复成功`)
          triggerDelayedPreview()
        } else {
          console.error('LAMA修复失败:', response.error || '未知错误')
          // 回退到纯色填充
          await fillBubbleWithColor(bubble.coords, fillColor, rotationAngle)
          triggerDelayedPreview()
        }
      } else {
        // 使用纯色填充
        await fillBubbleWithColor(bubble.coords, fillColor, rotationAngle)
        console.log(`气泡 #${index + 1} 纯色填充修复成功`)
        triggerDelayedPreview()
      }
    } catch (error) {
      console.error('背景修复出错:', error)
    }
  }

  /** 使用纯色填充气泡区域 */
  async function fillBubbleWithColor(
    coords: [number, number, number, number],
    fillColor: string,
    rotationAngle: number = 0
  ): Promise<void> {
    const image = currentImage.value
    if (!image) return

    const [x1, y1, x2, y2] = coords

    // 获取基础图像
    let baseSrc: string
    if (image.cleanImageData) {
      baseSrc = 'data:image/png;base64,' + image.cleanImageData
    } else if (image.originalDataURL) {
      baseSrc = image.originalDataURL
    } else {
      console.error('无法找到基础图像用于填充')
      return
    }

    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          resolve()
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
          ctx.moveTo(corners[0][0], corners[0][1])
          for (let i = 1; i < corners.length; i++) {
            ctx.lineTo(corners[i][0], corners[i][1])
          }
          ctx.closePath()
          ctx.fill()
        }

        // 更新cleanImageData
        const newCleanData = canvas.toDataURL('image/png').split(',')[1]
        imageStore.updateCurrentImage({ cleanImageData: newCleanData })

        console.log('已对气泡区域进行纯色填充:', coords, fillColor, '角度:', rotationAngle)
        resolve()
      }
      img.onerror = () => {
        console.error('加载基础图像失败')
        resolve()
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
      console.warn('无法进行 OCR 识别：缺少气泡或图片数据')
      return
    }

    try {
      console.log(`开始 OCR 识别气泡 #${index + 1}`)
      const imageData = image.originalDataURL.split(',')[1]
      const settings = settingsStore.settings
      const response = await ocrSingleBubbleApi(
        imageData,
        bubble.coords,
        settings.ocrEngine || 'manga_ocr',
        {
          source_language: settings.sourceLanguage,
          // 百度 OCR 参数（复刻原版 edit_mode.js）
          baidu_ocr_api_key: settings.baiduOcr.apiKey,
          baidu_ocr_secret_key: settings.baiduOcr.secretKey,
          baidu_version: settings.baiduOcr.version,
          baidu_source_language: settings.baiduOcr.sourceLanguage,
          // AI 视觉 OCR 参数（复刻原版 edit_mode.js）
          ai_vision_provider: settings.aiVisionOcr.provider,
          ai_vision_api_key: settings.aiVisionOcr.apiKey,
          ai_vision_model_name: settings.aiVisionOcr.modelName,
          ai_vision_ocr_prompt: settings.aiVisionOcr.prompt,
          custom_ai_vision_base_url: settings.aiVisionOcr.customBaseUrl
        }
      )

      if (response.success && response.text !== undefined) {
        bubbleStore.updateBubble(index, { originalText: response.text })
        console.log(`OCR 识别成功: "${response.text}"`)
      } else {
        console.error('OCR 识别失败:', response.error || '未知错误')
      }
    } catch (error) {
      console.error('OCR 识别出错:', error)
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
    handleBubbleDragging,
    handleBubbleDragEnd,

    // 气泡调整大小
    handleBubbleResizeStart,
    handleBubbleResizing,
    handleBubbleResizeEnd,

    // 气泡旋转
    handleBubbleRotateStart,
    handleBubbleRotating,
    handleBubbleRotateEnd,

    // 气泡绘制
    toggleDrawingMode,
    handleDrawBubble,
    startDrawingBox,
    updateDrawingBox,
    finishDrawingBox,
    cancelDrawing,
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
