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
import { executeRender } from '@/composables/translation/core/steps'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { shouldEnableAutoSave, saveTranslatedImage } from '../../core/saveStep'
import { useParallelTranslation } from '../useParallelTranslation'
import type { BubbleCoords } from '@/types/bubble'

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

    // 获取干净背景图：优先使用inpaintResult，其次使用已有的cleanImageData
    const cleanImage = task.inpaintResult?.cleanImage
      || task.imageData.cleanImageData
      || this.extractBase64(task.imageData.translatedDataURL || task.imageData.originalDataURL)

    // 构建渲染输入数据
    const coords = task.detectionResult?.bubbleCoords || []
    const texts = task.translateResult?.translatedTexts || []
    const originals = task.ocrResult?.originalTexts || []
    const colors = task.colorResult?.colors || []
    const angles = task.detectionResult?.bubbleAngles || []
    const directions = task.detectionResult?.autoDirections || []
    const textboxTexts = task.translateResult?.textboxTexts || []

    // 如果没有气泡或没有译文，跳过渲染（消除文字模式会走这里）
    if (coords.length === 0) {
      task.renderResult = {
        finalImage: cleanImage,
        bubbleStates: []
      }
      task.status = 'completed'
      this.updateImageToUI(task, imageStore, bubbleStore, settingsStore)
      this.resultCollector.add(task)
      return task
    }

    // AI校对模式：使用已有的bubbleStates，只更新译文
    if (!task.detectionResult && task.imageData.bubbleStates && task.imageData.bubbleStates.length > 0) {
      const existingBubbles = task.imageData.bubbleStates

      // 准备数据
      const bubbleCoords = existingBubbles.map(bs => bs.coords)
      const bubbleAngles = existingBubbles.map(bs => bs.rotationAngle || 0)
      const autoDirections = existingBubbles.map(bs => bs.autoTextDirection || 'vertical')
      const originalTexts = existingBubbles.map(bs => bs.originalText || '')
      const translatedTexts = existingBubbles.map((bs, idx) => texts[idx] || bs.translatedText || '')
      const textboxTexts = existingBubbles.map(bs => bs.textboxText || '')

      const colors = existingBubbles.map(bs => ({
        textColor: bs.textColor || '#000000',
        bgColor: bs.fillColor || '#FFFFFF',
        autoFgColor: bs.autoFgColor || null,
        autoBgColor: bs.autoBgColor || null
      }))

      // 使用executeRender步骤模块
      const result = await executeRender({
        imageIndex: task.imageIndex,
        cleanImage: cleanImage,
        bubbleCoords: bubbleCoords as any,
        bubbleAngles: bubbleAngles,
        autoDirections: autoDirections,
        originalTexts: originalTexts,
        translatedTexts: translatedTexts,
        textboxTexts: textboxTexts,
        colors: colors,
        savedTextStyles: null,
        currentMode: 'proofread'
      })

      task.renderResult = {
        finalImage: result.finalImage,
        bubbleStates: result.bubbleStates
      }
      task.status = 'completed'
      this.updateImageToUI(task, imageStore, bubbleStore, settingsStore)
      this.resultCollector.add(task)
      return task
    }

    // removeText模式检测：有检测结果但没有翻译结果
    // 需要构建bubbleStates以便编辑模式能显示检测框（无论是否有OCR结果）
    if (task.detectionResult && !task.translateResult) {
      console.log(`[渲染池] 图片 ${task.imageIndex + 1}: 检测到消除文字模式（无翻译结果），构建基本bubbleStates`)

      // 构建基本的bubbleStates（包含检测框信息但无译文）
      const { textStyle } = settingsStore.settings
      const bubbleStates = coords.map((coord, idx) => {
        const autoDir = directions[idx] || 'vertical'
        const mappedAutoDir: 'vertical' | 'horizontal' = autoDir === 'v' ? 'vertical'
          : autoDir === 'h' ? 'horizontal'
            : (autoDir === 'vertical' || autoDir === 'horizontal') ? autoDir : 'vertical'

        return {
          coords: (coord.length >= 4 ? [coord[0], coord[1], coord[2], coord[3]] : [0, 0, 0, 0]) as BubbleCoords,
          polygon: [] as number[][],
          position: { x: 0, y: 0 },
          rotationAngle: angles[idx] || 0,
          originalText: originals[idx] || '',  // 如果启用了removeTextWithOcr，这里会有OCR结果
          translatedText: '',  // 消除文字模式无译文
          textboxText: '',
          textDirection: mappedAutoDir,
          autoTextDirection: mappedAutoDir,
          fontSize: textStyle.fontSize,
          fontFamily: textStyle.fontFamily,
          autoFontSize: textStyle.autoFontSize,
          textColor: textStyle.textColor,
          fillColor: textStyle.fillColor,
          strokeEnabled: textStyle.strokeEnabled,
          strokeColor: textStyle.strokeColor,
          strokeWidth: textStyle.strokeWidth,
          inpaintMethod: textStyle.inpaintMethod,
          autoFgColor: null,
          autoBgColor: null
        }
      })

      task.renderResult = {
        finalImage: cleanImage,
        bubbleStates: bubbleStates
      }
      task.status = 'completed'
      this.updateImageToUI(task, imageStore, bubbleStore, settingsStore)
      this.resultCollector.add(task)
      return task
    }

    // 标准翻译模式：使用executeRender步骤模块
    const result = await executeRender({
      imageIndex: task.imageIndex,
      cleanImage: cleanImage,
      bubbleCoords: coords as any,
      bubbleAngles: angles,
      autoDirections: directions,
      originalTexts: originals,
      translatedTexts: texts,
      textboxTexts: textboxTexts,
      colors: colors,
      savedTextStyles: null,  // 并行翻译不使用savedTextStyles
      currentMode: 'standard'  // 固定为standard模式
    })

    task.renderResult = {
      finalImage: result.finalImage,
      bubbleStates: result.bubbleStates
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
      textMask: task.detectionResult?.textMask || null,  // 保存精确文字掩膜
      userMask: task.imageData.userMask || null,  // 【重要】保留用户笔刷掩膜
      translationStatus: 'completed',
      translationFailed: false,
      showOriginal: false,
      hasUnsavedChanges: true,
      // 【修复】保存翻译时的设置，优先使用已预分发的设置，确保批量翻译时所有图片使用统一设置
      // 这样即使用户在翻译过程中修改了设置，也不会影响正在翻译的图片
      fontSize: task.imageData.fontSize ?? settingsStore.settings.textStyle.fontSize,
      autoFontSize: task.imageData.autoFontSize ?? settingsStore.settings.textStyle.autoFontSize,
      fontFamily: task.imageData.fontFamily ?? settingsStore.settings.textStyle.fontFamily,
      layoutDirection: task.imageData.layoutDirection ?? settingsStore.settings.textStyle.layoutDirection,
      textColor: task.imageData.textColor ?? settingsStore.settings.textStyle.textColor,
      fillColor: task.imageData.fillColor ?? settingsStore.settings.textStyle.fillColor,
      strokeEnabled: task.imageData.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
      strokeColor: task.imageData.strokeColor ?? settingsStore.settings.textStyle.strokeColor,
      strokeWidth: task.imageData.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
      inpaintMethod: task.imageData.inpaintMethod ?? settingsStore.settings.textStyle.inpaintMethod,
      useAutoTextColor: task.imageData.useAutoTextColor ?? settingsStore.settings.textStyle.useAutoTextColor
    })

    // 2. 如果是当前显示的图片，同步更新 bubbleStore
    if (imageIndex === imageStore.currentImageIndex && task.renderResult?.bubbleStates) {
      bubbleStore.setBubbles(task.renderResult.bubbleStates)
    }

    // 3. 更新进度
    this.progressTracker.incrementCompleted()

    // 4. 自动保存（书架模式下）
    if (shouldEnableAutoSave()) {
      saveTranslatedImage(imageIndex)
        .catch(err => {
          console.error(`[RenderPool] 自动保存图片 ${imageIndex + 1} 失败:`, err)
        })
        .finally(() => {
          // 无论成功还是失败，都更新保存进度
          const { progress } = useParallelTranslation()
          const globalProgress = progress.value
          if (globalProgress.save) {
            globalProgress.save.completed = (globalProgress.save.completed || 0) + 1
          }
        })
    }

    console.log(`✅ 图片 ${imageIndex + 1} 渲染完成`)
  }

  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
  }
}
