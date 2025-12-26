/**
 * 导出导入功能组合式函数
 * 提供文本导出、文本导入、图片下载等功能
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useToast } from '@/utils/toast'
import {
  downloadStartSession,
  downloadUploadImage,
  downloadFinalize,
  getDownloadFileUrl,
  cleanTempFiles
} from '@/api/system'
import { reRenderImage } from '@/api/translate'
import type { BubbleState } from '@/types/bubble'
import { getEffectiveDirection } from '@/types/bubble'

/**
 * 导出文本数据结构
 */
export interface ExportTextData {
  imageIndex: number
  bubbles: Array<{
    bubbleIndex: number
    original: string
    translated: string
    textDirection: 'vertical' | 'horizontal'
  }>
}

/**
 * 下载格式类型
 */
export type DownloadFormat = 'zip' | 'pdf' | 'cbz'

/**
 * 导出导入功能组合式函数
 */
export function useExportImport() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const toast = useToast()

  // ============================================================
  // 状态
  // ============================================================

  /** 是否正在下载 */
  const isDownloading = ref(false)

  /** 下载进度 (0-100) */
  const downloadProgress = ref(0)

  /** 下载进度文本 */
  const downloadProgressText = ref('')

  /** 是否正在导入 */
  const isImporting = ref(false)

  /** 导入进度 (0-100) */
  const importProgress = ref(0)

  /** 导入进度文本 */
  const importProgressText = ref('')

  // ============================================================
  // 计算属性
  // ============================================================

  /** 是否可以导出文本 */
  const canExportText = computed(() => imageStore.hasImages)

  /** 是否可以导入文本 */
  const canImportText = computed(() => imageStore.hasImages)

  /** 是否可以下载 */
  const canDownload = computed(() => imageStore.hasImages)

  // ============================================================
  // 文本导出功能
  // ============================================================

  /**
   * 导出所有图片的文本（原文和译文）为 JSON 文件
   */
  function exportText(): void {
    const allImages = imageStore.images
    if (allImages.length === 0) {
      toast.warning('没有可导出的图片文本')
      return
    }

    // 准备导出数据
    const exportData: ExportTextData[] = []

    // 遍历所有图片
    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
      const image = allImages[imageIndex]
      if (!image) continue

      const bubbleStates = image.bubbleStates || []

      // 构建该图片的文本数据
      const imageTextData: ExportTextData = {
        imageIndex: imageIndex,
        bubbles: []
      }

      // 构建每个气泡的文本数据
      for (let bubbleIndex = 0; bubbleIndex < bubbleStates.length; bubbleIndex++) {
        const bubble = bubbleStates[bubbleIndex]
        if (!bubble) continue

        const original = bubble.originalText || ''
        const translated = bubble.translatedText || bubble.textboxText || ''

        // 获取气泡的排版方向，确保不传递 'auto'
        let textDirection: 'vertical' | 'horizontal' = 'vertical'
        if (bubble.textDirection && bubble.textDirection !== 'auto') {
          textDirection = bubble.textDirection as 'vertical' | 'horizontal'
        } else if (bubble.autoTextDirection) {
          textDirection = bubble.autoTextDirection as 'vertical' | 'horizontal'
        }

        imageTextData.bubbles.push({
          bubbleIndex: bubbleIndex,
          original: original,
          translated: translated,
          textDirection: textDirection
        })
      }

      exportData.push(imageTextData)
    }

    // 将数据转换为 JSON 字符串
    const jsonData = JSON.stringify(exportData, null, 2)

    // 创建 Blob 并触发下载
    const blob = new Blob([jsonData], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url

    // 生成文件名：translations_YYYYMMDD_HHMMSS.json
    const now = new Date()
    const dateStr = now.toISOString().replace(/[-:T]/g, '').slice(0, 15)
    a.download = `translations_${dateStr}.json`

    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast.success('文本导出成功！')
  }

  /**
   * 导出文本为 JSON 数据（不触发下载，用于高质量翻译等内部使用）
   * @returns 导出的 JSON 数据，如果没有图片则返回 null
   */
  function exportTextToJson(): ExportTextData[] | null {
    const allImages = imageStore.images
    if (allImages.length === 0) return null

    const exportData: ExportTextData[] = []

    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
      const image = allImages[imageIndex]
      if (!image) continue

      const bubbleStates = image.bubbleStates || []

      const imageTextData: ExportTextData = {
        imageIndex: imageIndex,
        bubbles: []
      }

      for (let bubbleIndex = 0; bubbleIndex < bubbleStates.length; bubbleIndex++) {
        const bubble = bubbleStates[bubbleIndex]
        if (!bubble) continue

        const original = bubble.originalText || ''
        const translated = bubble.translatedText || bubble.textboxText || ''

        let textDirection: 'vertical' | 'horizontal' = 'vertical'
        if (bubble.textDirection && bubble.textDirection !== 'auto') {
          textDirection = bubble.textDirection as 'vertical' | 'horizontal'
        } else if (bubble.autoTextDirection) {
          textDirection = bubble.autoTextDirection as 'vertical' | 'horizontal'
        }

        imageTextData.bubbles.push({
          bubbleIndex: bubbleIndex,
          original: original,
          translated: translated,
          textDirection: textDirection
        })
      }

      exportData.push(imageTextData)
    }

    return exportData
  }

  // ============================================================
  // 文本导入功能
  // ============================================================

  /**
   * 导入 JSON 文本文件并应用到当前图片集
   * 【复刻原版】导入后自动重新渲染已翻译的图片
   * @param jsonFile - 用户选择的 JSON 文件
   */
  async function importText(jsonFile: File): Promise<void> {
    if (!jsonFile) {
      toast.warning('未选择文件')
      return
    }

    isImporting.value = true
    importProgress.value = 0
    importProgressText.value = '准备导入文本...'
    toast.info('正在导入文本...', 0)

    try {
      // 读取文件内容
      const fileContent = await readFileAsText(jsonFile)

      // 解析 JSON 数据
      importProgress.value = 10
      importProgressText.value = '解析 JSON 数据...'

      const importedData: ExportTextData[] = JSON.parse(fileContent)

      // 验证数据格式
      if (!Array.isArray(importedData)) {
        throw new Error('导入的 JSON 格式不正确，应为数组')
      }

      // 统计信息
      let updatedImages = 0
      let updatedBubbles = 0

      // 获取当前设置
      const textStyle = settingsStore.settings.textStyle
      const currentFontSize = textStyle.autoFontSize ? 'auto' : textStyle.fontSize
      const currentFontFamily = textStyle.fontFamily
      const currentTextColor = textStyle.textColor
      const currentFillColor = textStyle.fillColor

      importProgress.value = 20
      importProgressText.value = '开始更新图片...'

      // 遍历导入的数据
      const totalImages = importedData.length
      let processedImages = 0

      // 需要重新渲染的图片索引列表
      const imagesToReRender: number[] = []

      for (const imageData of importedData) {
        processedImages++
        const progress = 20 + (processedImages / totalImages) * 60 // 从 20% 到 80%
        importProgress.value = progress
        importProgressText.value = `处理图片 ${processedImages}/${totalImages}`

        const imageIndex = imageData.imageIndex

        // 检查图片索引是否有效
        if (imageIndex < 0 || imageIndex >= imageStore.images.length) {
          console.warn(`跳过无效的图片索引: ${imageIndex}`)
          continue
        }

        const image = imageStore.images[imageIndex]
        if (!image) continue

        let imageUpdated = false

        // 【复刻原版】确保必要的数组存在
        if (!image.bubbleStates) {
          image.bubbleStates = []
        }
        if (!image.bubbleTexts) {
          image.bubbleTexts = []
        }
        if (!image.originalTexts) {
          image.originalTexts = []
        }

        // 遍历气泡数据
        for (const bubbleData of imageData.bubbles) {
          const bubbleIndex = bubbleData.bubbleIndex
          const original = bubbleData.original
          const translated = bubbleData.translated
          // 获取排版方向，确保不是 'auto'（复刻原版）
          const rawDir = bubbleData.textDirection as string | undefined
          const textDirection: 'vertical' | 'horizontal' =
            (rawDir === 'vertical' || rawDir === 'horizontal') ? rawDir : 'vertical'

          // 【复刻原版】确保数组索引存在
          while (image.bubbleTexts.length <= bubbleIndex) {
            image.bubbleTexts.push('')
          }
          while (image.originalTexts.length <= bubbleIndex) {
            image.originalTexts.push('')
          }

          // 【复刻原版】更新文本数组
          if (original) image.originalTexts[bubbleIndex] = original
          if (translated) image.bubbleTexts[bubbleIndex] = translated

          // 确保气泡状态数组索引存在
          while (image.bubbleStates.length <= bubbleIndex) {
            image.bubbleStates.push(createDefaultBubbleState(
              currentFontSize,
              currentFontFamily,
              currentTextColor,
              currentFillColor
            ))
          }

          const bubbleState = image.bubbleStates[bubbleIndex]
          if (bubbleState) {
            // 更新文本
            if (original) bubbleState.originalText = original
            if (translated) {
              bubbleState.translatedText = translated
              bubbleState.textboxText = translated
            }
            // 更新排版方向
            bubbleState.textDirection = textDirection as 'vertical' | 'horizontal' | 'auto'

            imageUpdated = true
            updatedBubbles++
          }
        }

        // 【复刻原版】确保 bubbleTexts 与 bubbleStates 同步
        if (imageUpdated && image.bubbleStates) {
          image.bubbleTexts = image.bubbleStates.map(bs => bs.translatedText || '')
        }

        if (imageUpdated) {
          updatedImages++
          image.hasUnsavedChanges = true

          // 如果图片已翻译且有气泡，添加到重新渲染列表
          if (image.translatedDataURL && image.bubbleStates && image.bubbleStates.length > 0) {
            imagesToReRender.push(imageIndex)
          }
        }
      }

      // 【复刻原版】重新渲染需要更新的图片
      if (imagesToReRender.length > 0) {
        importProgress.value = 80
        importProgressText.value = '开始渲染图片...'
        toast.info('正在渲染图片，请稍候...', 0)

        const textStyle = settingsStore.settings.textStyle
        const layoutDir = textStyle.layoutDirection
        const isAutoLayout = layoutDir === 'auto'

        for (let i = 0; i < imagesToReRender.length; i++) {
          const imageIndex = imagesToReRender[i]
          if (imageIndex === undefined) continue

          const img = imageStore.images[imageIndex]
          if (!img || !img.bubbleStates) continue

          importProgress.value = 80 + (i / imagesToReRender.length) * 20
          importProgressText.value = `渲染图片 ${i + 1}/${imagesToReRender.length}`

          try {
            // 【复刻原版】背景兜底策略：clean → original
            let cleanImageBase64 = ''
            if (img.cleanImageData) {
              cleanImageBase64 = img.cleanImageData.includes('base64,')
                ? (img.cleanImageData.split('base64,')[1] || '')
                : img.cleanImageData
            } else if (img.originalDataURL) {
              // 兜底：使用原图作为背景
              cleanImageBase64 = img.originalDataURL.includes('base64,')
                ? (img.originalDataURL.split('base64,')[1] || '')
                : img.originalDataURL
              console.log(`importText: 图片 ${imageIndex} 使用原图作为背景（兜底）`)
            }

            if (!cleanImageBase64) {
              console.log(`importText: 图片 ${imageIndex} 没有可用的背景图，跳过`)
              continue
            }

            // 构建 API 参数
            const bubbleStatesForApi = img.bubbleStates.map(bs => ({
              translatedText: bs.translatedText || '',
              coords: bs.coords,
              fontSize: bs.fontSize || textStyle.fontSize,
              fontFamily: bs.fontFamily || textStyle.fontFamily,
              textDirection: getEffectiveDirection(bs),
              textColor: bs.textColor || textStyle.textColor,
              rotationAngle: bs.rotationAngle || 0,
              position: bs.position || { x: 0, y: 0 },
              strokeEnabled: bs.strokeEnabled ?? textStyle.strokeEnabled,
              strokeColor: bs.strokeColor || textStyle.strokeColor,
              strokeWidth: bs.strokeWidth ?? textStyle.strokeWidth,
            }))

            const response = await reRenderImage({
              clean_image: cleanImageBase64,
              bubble_texts: bubbleStatesForApi.map(s => s.translatedText),
              bubble_coords: bubbleStatesForApi.map(s => s.coords),
              fontSize: textStyle.fontSize,
              fontFamily: textStyle.fontFamily,
              textDirection: isAutoLayout ? 'vertical' : layoutDir,
              textColor: textStyle.textColor,
              bubble_states: bubbleStatesForApi,
              use_individual_styles: true,
              use_inpainting: false,
              use_lama: false,
              is_font_style_change: true,
              strokeEnabled: textStyle.strokeEnabled,
              strokeColor: textStyle.strokeColor,
              strokeWidth: textStyle.strokeWidth,
            })

            if (response.rendered_image) {
              imageStore.updateImageByIndex(imageIndex, {
                translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
                hasUnsavedChanges: true
              })
              console.log(`importText: 图片 ${imageIndex} 渲染成功`)
            }
          } catch (err) {
            console.error(`importText: 重渲染图片 ${imageIndex} 失败:`, err)
          }
        }
      }

      importProgress.value = 100
      importProgressText.value = '导入完成'

      // 显示导入结果
      const reRenderedCount = imagesToReRender.length
      const message = reRenderedCount > 0
        ? `导入成功！更新了 ${updatedImages} 张图片中的 ${updatedBubbles} 个气泡文本，重渲染了 ${reRenderedCount} 张图片`
        : `导入成功！更新了 ${updatedImages} 张图片中的 ${updatedBubbles} 个气泡文本`
      toast.success(message)
    } catch (error) {
      console.error('导入文本出错:', error)
      toast.error(`导入失败: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isImporting.value = false
      // 延时重置进度
      setTimeout(() => {
        importProgress.value = 0
        importProgressText.value = ''
      }, 2000)
    }
  }

  /**
   * 读取文件为文本
   * @param file - 文件对象
   * @returns 文件内容字符串
   */
  function readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        if (e.target?.result) {
          resolve(e.target.result as string)
        } else {
          reject(new Error('读取文件失败'))
        }
      }
      reader.onerror = () => reject(new Error('读取文件时出错'))
      reader.readAsText(file)
    })
  }

  /**
   * 创建默认气泡状态
   */
  function createDefaultBubbleState(
    fontSize: number | 'auto',
    fontFamily: string,
    textColor: string,
    fillColor: string
  ): BubbleState {
    const textStyle = settingsStore.settings.textStyle
    return {
      coords: [0, 0, 100, 100],
      polygon: [],
      originalText: '',
      translatedText: '',
      textboxText: '',
      fontSize: typeof fontSize === 'number' ? fontSize : 25,
      fontFamily: fontFamily,
      textDirection: 'vertical',
      autoTextDirection: 'vertical',
      textColor: textColor,
      fillColor: fillColor,
      rotationAngle: 0,
      position: { x: 0, y: 0 },
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth,
      inpaintMethod: textStyle.inpaintMethod
    }
  }

  // ============================================================
  // 图片下载功能
  // ============================================================

  /**
   * 下载当前图片（翻译后或原始图片）
   */
  function downloadCurrentImage(): void {
    const currentImage = imageStore.currentImage
    if (!currentImage) {
      toast.warning('没有可下载的图片')
      return
    }

    // 优先使用翻译后图片，如无则使用原始图片
    const imageDataURL = currentImage.translatedDataURL || currentImage.originalDataURL

    if (!imageDataURL) {
      toast.warning('没有可下载的图片')
      return
    }

    isDownloading.value = true

    try {
      // 从 Base64 数据创建 Blob
      const base64Data = imageDataURL.split(',')[1]
      if (!base64Data) {
        throw new Error('无效的图片数据')
      }

      const byteCharacters = atob(base64Data)
      const byteArrays: ArrayBuffer[] = []

      for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512)
        const byteNumbers = new Array(slice.length)
        for (let i = 0; i < slice.length; i++) {
          byteNumbers[i] = slice.charCodeAt(i)
        }
        const uint8Array = new Uint8Array(byteNumbers)
        byteArrays.push(uint8Array.buffer as ArrayBuffer)
      }

      const blob = new Blob(byteArrays, { type: 'image/png' })
      const url = URL.createObjectURL(blob)

      // 创建下载链接
      const a = document.createElement('a')
      a.href = url

      // 生成文件名
      let fileName = currentImage.fileName || `image_${imageStore.currentImageIndex}.png`
      // 为已翻译和未翻译的图片使用不同前缀
      const prefix = currentImage.translatedDataURL ? 'translated' : 'original'
      fileName = `${prefix}_${fileName.replace(/\.[^/.]+$/, '')}.png`
      a.download = fileName

      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      toast.success(`下载成功: ${fileName}`)
    } catch (e) {
      console.error('下载图片时出错:', e)
      toast.error('下载失败')
    } finally {
      isDownloading.value = false
    }
  }

  /**
   * 下载所有图片（逐张上传到后端，避免大数据量导致的字符串长度限制）
   * @param format - 下载格式 (zip/pdf/cbz)
   */
  async function downloadAllImages(format: DownloadFormat = 'zip'): Promise<void> {
    const allImages = imageStore.images
    if (allImages.length === 0) {
      toast.warning('没有可下载的图片')
      return
    }

    isDownloading.value = true
    downloadProgress.value = 0
    downloadProgressText.value = '准备下载...'
    toast.info('下载中...处理可能需要一定时间，请耐心等待...', 0)

    try {
      // 收集需要发送的图像数据（只记录索引和类型，不一次性收集所有数据）
      const imageInfoList: Array<{ index: number; type: 'translated' | 'original' }> = []
      let translatedCount = 0
      let originalCount = 0

      downloadProgress.value = 5
      downloadProgressText.value = '检查图片数据...'

      for (let i = 0; i < allImages.length; i++) {
        const imgData = allImages[i]
        if (!imgData) continue

        // 优先使用翻译后的图片，如果没有则使用原始图片
        if (imgData.translatedDataURL) {
          imageInfoList.push({ index: i, type: 'translated' })
          translatedCount++
        } else if (imgData.originalDataURL) {
          imageInfoList.push({ index: i, type: 'original' })
          originalCount++
        }
      }

      if (imageInfoList.length === 0) {
        toast.warning('没有可下载的图片')
        return
      }

      // 步骤1: 创建下载会话（传递总图片数，与后端一致）
      downloadProgress.value = 10
      downloadProgressText.value = '创建下载会话...'

      const sessionResponse = await downloadStartSession(imageInfoList.length)
      if (!sessionResponse.success || !sessionResponse.session_id) {
        throw new Error(sessionResponse.error || '创建会话失败')
      }
      const sessionId = sessionResponse.session_id

      // 步骤2: 逐张上传图片
      const totalImages = imageInfoList.length
      let uploadedCount = 0
      let failedCount = 0

      for (let i = 0; i < imageInfoList.length; i++) {
        const info = imageInfoList[i]
        if (!info) continue

        const imgData = allImages[info.index]
        if (!imgData) continue

        const imageDataURL =
          info.type === 'translated' ? imgData.translatedDataURL : imgData.originalDataURL
        if (!imageDataURL) continue

        // 更新进度条
        const progress = 10 + (i / totalImages) * 70 // 10% - 80%
        downloadProgress.value = progress
        downloadProgressText.value = `上传图片 ${i + 1}/${totalImages}...`

        try {
          // 传递 image_index 而不是 fileName（与后端一致）
          const uploadResponse = await downloadUploadImage(
            sessionId,
            imageDataURL,
            i
          )

          if (uploadResponse.success) {
            uploadedCount++
          } else {
            console.error(`上传图片 ${i} 失败:`, uploadResponse.error)
            failedCount++
          }
        } catch (e) {
          console.error(`上传图片 ${i} 出错:`, e)
          failedCount++
        }
      }

      if (uploadedCount === 0) {
        throw new Error('所有图片上传失败')
      }

      // 步骤3: 请求打包
      downloadProgress.value = 85
      downloadProgressText.value = '打包文件...'

      const finalizeResponse = await downloadFinalize(sessionId, format)
      if (!finalizeResponse.success || !finalizeResponse.file_id) {
        throw new Error(finalizeResponse.error || '打包失败')
      }

      // 步骤4: 触发下载
      downloadProgress.value = 95
      downloadProgressText.value = '准备下载...'

      const downloadUrl = getDownloadFileUrl(finalizeResponse.file_id, format)
      const link = document.createElement('a')
      link.href = downloadUrl
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      downloadProgress.value = 100
      downloadProgressText.value = '下载已开始'

      // 更新下载成功信息
      let successMessage = `已成功处理 ${uploadedCount} 张图片`
      if (failedCount > 0) {
        successMessage += `（${failedCount} 张失败）`
      }
      if (translatedCount > 0 && originalCount > 0) {
        successMessage += `（${translatedCount} 张翻译图片和 ${originalCount} 张原始图片）`
      } else if (translatedCount > 0) {
        successMessage += `（全部为翻译后图片）`
      } else if (originalCount > 0) {
        successMessage += `（全部为原始图片）`
      }
      successMessage += '，下载即将开始'

      toast.success(successMessage)

      // 启动后台清理过期文件的请求（1分钟后）
      setTimeout(async () => {
        try {
          const cleanResponse = await cleanTempFiles()
          console.log('临时文件清理结果:', cleanResponse)
        } catch (error) {
          console.error('清理临时文件失败:', error)
        }
      }, 60000)
    } catch (e) {
      console.error('下载所有图片时出错:', e)
      toast.error(`下载失败: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      isDownloading.value = false
      // 延时重置进度
      setTimeout(() => {
        downloadProgress.value = 0
        downloadProgressText.value = ''
      }, 2000)
    }
  }

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 状态
    isDownloading,
    downloadProgress,
    downloadProgressText,
    isImporting,
    importProgress,
    importProgressText,

    // 计算属性
    canExportText,
    canImportText,
    canDownload,

    // 文本导出功能
    exportText,
    exportTextToJson,

    // 文本导入功能
    importText,

    // 图片下载功能
    downloadCurrentImage,
    downloadAllImages
  }
}
