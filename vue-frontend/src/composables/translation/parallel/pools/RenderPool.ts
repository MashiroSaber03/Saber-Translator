/**
 * 渲染池
 * 
 * 负责将翻译结果渲染到图片上
 * 关键特性：每完成一张图片的渲染，立即更新到界面
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import type { ResultCollector } from '../ResultCollector'
import { parallelRender } from '@/api/parallelTranslate'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import type { BubbleState, BubbleCoords, TextDirection, InpaintMethod } from '@/types/bubble'

export class RenderPool extends TaskPool {
  private resultCollector: ResultCollector

  constructor(
    progressTracker: ParallelProgressTracker,
    resultCollector: ResultCollector,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    // 渲染池是最后一个，无下一个池子，不需要深度学习锁
    super('渲染', '✨', null, null, progressTracker, onTaskComplete)
    this.resultCollector = resultCollector
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    // 在方法内部获取 store
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()

    // 构建气泡状态
    const bubbleStates = this.buildBubbleStates(task, settingsStore)

    // 如果没有气泡或没有译文，跳过渲染（消除文字模式会走这里）
    if (bubbleStates.length === 0) {
      // 获取最终图片：优先使用修复后的干净图片
      const finalImage = task.inpaintResult?.cleanImage
        || task.imageData.cleanImageData
        || this.extractBase64(task.imageData.originalDataURL)
      task.renderResult = {
        finalImage: finalImage,
        bubbleStates: []
      }
      task.status = 'completed'
      this.updateImageToUI(task, imageStore, bubbleStore, settingsStore)
      this.resultCollector.add(task)
      return task
    }

    // 获取干净背景图：优先使用inpaintResult，其次使用已有的cleanImageData
    const cleanImage = task.inpaintResult?.cleanImage
      || task.imageData.cleanImageData
      || this.extractBase64(task.imageData.translatedDataURL || task.imageData.originalDataURL)

    // 调用后端渲染 API
    const response = await parallelRender({
      clean_image: cleanImage,
      bubble_states: bubbleStates,
      fontSize: settingsStore.settings.textStyle.fontSize,
      fontFamily: settingsStore.settings.textStyle.fontFamily,
      textDirection: settingsStore.settings.textStyle.layoutDirection,
      textColor: settingsStore.settings.textStyle.textColor,
      strokeEnabled: settingsStore.settings.textStyle.strokeEnabled,
      strokeColor: settingsStore.settings.textStyle.strokeColor,
      strokeWidth: settingsStore.settings.textStyle.strokeWidth,
      autoFontSize: settingsStore.settings.textStyle.autoFontSize,
      use_individual_styles: true
    })

    if (!response.success) {
      throw new Error(response.error || '渲染失败')
    }

    task.renderResult = {
      finalImage: response.final_image || '',
      bubbleStates: response.bubble_states || bubbleStates
    }

    task.status = 'completed'

    // 实时更新到界面
    this.updateImageToUI(task, imageStore, bubbleStore, settingsStore)

    // 收集结果
    this.resultCollector.add(task)

    return task
  }

  /**
   * 实时更新图片到界面
   */
  private updateImageToUI(
    task: PipelineTask,
    imageStore: ReturnType<typeof useImageStore>,
    bubbleStore: ReturnType<typeof useBubbleStore>,
    settingsStore: ReturnType<typeof useSettingsStore>
  ): void {
    const imageIndex = task.imageIndex

    // 1. 更新 imageStore
    // 转换bubbleCoords为正确的类型
    const bubbleCoords = (task.detectionResult?.bubbleCoords || []).map(coord =>
      (coord.length >= 4 ? [coord[0], coord[1], coord[2], coord[3]] : [0, 0, 0, 0]) as BubbleCoords
    )

    imageStore.updateImageByIndex(imageIndex, {
      translatedDataURL: `data:image/png;base64,${task.renderResult!.finalImage}`,
      cleanImageData: task.inpaintResult?.cleanImage || null,
      bubbleStates: task.renderResult!.bubbleStates,
      bubbleCoords: bubbleCoords,
      bubbleAngles: task.detectionResult?.bubbleAngles || [],
      originalTexts: task.ocrResult?.originalTexts || [],
      bubbleTexts: task.translateResult?.translatedTexts || [],
      textboxTexts: task.translateResult?.textboxTexts || [],
      translationStatus: 'completed',
      translationFailed: false,
      showOriginal: false,
      hasUnsavedChanges: true,
      // 保存用户翻译时选择的设置（用于切换图片时恢复）
      layoutDirection: settingsStore.settings.textStyle.layoutDirection,
      autoFontSize: settingsStore.settings.textStyle.autoFontSize,
      useAutoTextColor: settingsStore.settings.textStyle.useAutoTextColor
    })

    // 2. 如果是当前显示的图片，同步更新 bubbleStore
    if (imageIndex === imageStore.currentImageIndex && task.renderResult?.bubbleStates) {
      bubbleStore.setBubbles(task.renderResult.bubbleStates)
    }

    // 3. 更新进度
    this.progressTracker.incrementCompleted()

    console.log(`✅ 图片 ${imageIndex + 1} 渲染完成并已更新到界面`)
  }

  /**
   * 构建 BubbleState 数组
   */
  private buildBubbleStates(
    task: PipelineTask,
    settingsStore: ReturnType<typeof useSettingsStore>
  ): BubbleState[] {
    // AI校对模式：使用已有的bubbleStates，只更新translatedText
    if (!task.detectionResult && task.imageData.bubbleStates && task.imageData.bubbleStates.length > 0) {
      const translatedTexts = task.translateResult?.translatedTexts || []
      return task.imageData.bubbleStates.map((state, idx) => ({
        ...state,
        translatedText: translatedTexts[idx] || state.translatedText || ''
      }))
    }

    // removeText模式检测：有检测结果但没有翻译结果，返回空数组跳过渲染
    // 这样会让 process() 方法直接使用干净背景图作为最终图片
    if (task.detectionResult && !task.translateResult) {
      console.log(`[渲染池] 图片 ${task.imageIndex + 1}: 检测到消除文字模式（无翻译结果），跳过渲染`)
      return []
    }

    const coords = task.detectionResult?.bubbleCoords || []
    const texts = task.translateResult?.translatedTexts || []
    const originals = task.ocrResult?.originalTexts || []
    const colors = task.colorResult?.colors || []
    const angles = task.detectionResult?.bubbleAngles || []
    const directions = task.detectionResult?.autoDirections || []
    const polygons = task.detectionResult?.bubblePolygons || []
    const settings = settingsStore.settings

    return coords.map((coord, idx) => {
      const autoDir = directions[idx]
      const mappedDirection: TextDirection = autoDir === 'v' ? 'vertical' : autoDir === 'h' ? 'horizontal' : 'vertical'

      // 【简化设计】textDirection 直接使用具体方向值
      // - 如果全局设置是 'auto'，使用检测结果
      // - 否则使用全局设置的值
      const globalTextDir = settings.textStyle.layoutDirection
      const textDirection: TextDirection =
        (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
          ? globalTextDir
          : mappedDirection

      // 颜色处理：优先使用自动提取的颜色
      let textColor = settings.textStyle.textColor
      let fillColor = settings.textStyle.fillColor
      const colorInfo = colors[idx]

      if (settings.textStyle.useAutoTextColor && colorInfo) {
        if (colorInfo.textColor) textColor = colorInfo.textColor
        if (colorInfo.bgColor) fillColor = colorInfo.bgColor
      }

      return {
        originalText: originals[idx] || '',
        translatedText: texts[idx] || '',
        textboxText: '',
        coords: (coord.length >= 4 ? [coord[0], coord[1], coord[2], coord[3]] : [0, 0, 0, 0]) as BubbleCoords,
        polygon: polygons[idx] || [],
        fontSize: settings.textStyle.fontSize,
        fontFamily: settings.textStyle.fontFamily,
        textDirection: textDirection,         // 渲染用的具体方向
        autoTextDirection: mappedDirection,   // 备份检测结果
        textColor: textColor,
        fillColor: fillColor,
        rotationAngle: angles[idx] || 0,
        position: { x: 0, y: 0 },
        strokeEnabled: settings.textStyle.strokeEnabled,
        strokeColor: settings.textStyle.strokeColor,
        strokeWidth: settings.textStyle.strokeWidth,
        inpaintMethod: settings.textStyle.inpaintMethod as InpaintMethod,
        autoFgColor: colorInfo?.autoFgColor || null,
        autoBgColor: colorInfo?.autoBgColor || null
      }
    })
  }

  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
  }
}
