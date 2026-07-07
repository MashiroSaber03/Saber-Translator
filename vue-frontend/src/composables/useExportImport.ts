import { computed, getCurrentInstance, onUnmounted, ref } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { useToast } from '@/utils/toast'
import {
  downloadStartSession,
  downloadUploadImage,
  downloadFinalize,
  getDownloadFileUrl,
  cleanTempFiles,
} from '@/api/system'
import type { BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import { executeRender } from '@/composables/translation/core/steps'
import { buildEditRenderInput } from '@/composables/edit/editRenderRequest'
import { triggerBlobDownload, triggerUrlDownload } from '@/utils/browserDownload'
import { readFileAsText } from '@/utils/dataUrl'

export interface ExportTextData {
  imageIndex: number
  bubbles: Array<{
    bubbleIndex: number
    original: string
    translated: string
    textDirection: 'vertical' | 'horizontal'
  }>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isTextDirection(value: unknown): value is 'vertical' | 'horizontal' {
  return value === 'vertical' || value === 'horizontal'
}

function parseImportTextData(value: unknown): ExportTextData[] {
  if (!Array.isArray(value)) {
    throw new Error('导入的 JSON 格式不正确，应为数组')
  }

  return value.map((imageData, imagePosition): ExportTextData => {
    if (!isRecord(imageData)) {
      throw new Error(`第 ${imagePosition + 1} 个图片条目格式不正确`)
    }

    const imageIndex = imageData.imageIndex
    const bubbles = imageData.bubbles
    if (!Number.isInteger(imageIndex) || imageIndex < 0) {
      throw new Error(`第 ${imagePosition + 1} 个图片条目缺少有效 imageIndex`)
    }
    if (!Array.isArray(bubbles)) {
      throw new Error(`第 ${imagePosition + 1} 个图片条目缺少 bubbles 数组`)
    }

    return {
      imageIndex,
      bubbles: bubbles.map((bubbleData, bubblePosition) => {
        if (!isRecord(bubbleData)) {
          throw new Error(
            `第 ${imagePosition + 1} 个图片条目的第 ${bubblePosition + 1} 个气泡格式不正确`
          )
        }

        const bubbleIndex = bubbleData.bubbleIndex
        const original = bubbleData.original
        const translated = bubbleData.translated
        const textDirection = bubbleData.textDirection
        if (!Number.isInteger(bubbleIndex) || bubbleIndex < 0) {
          throw new Error(
            `第 ${imagePosition + 1} 个图片条目的第 ${bubblePosition + 1} 个气泡缺少有效 bubbleIndex`
          )
        }
        if (
          typeof original !== 'string' ||
          typeof translated !== 'string' ||
          !isTextDirection(textDirection)
        ) {
          throw new Error(
            `第 ${imagePosition + 1} 个图片条目的第 ${bubblePosition + 1} 个气泡不符合当前文本导入 schema`
          )
        }

        return {
          bubbleIndex,
          original,
          translated,
          textDirection,
        }
      }),
    }
  })
}

export const DOWNLOAD_FORMATS = ['zip', 'pdf', 'cbz'] as const
export type DownloadFormat = (typeof DOWNLOAD_FORMATS)[number]
export type DownloadImageType = 'translated' | 'original'

export interface DownloadImageEntry {
  index: number
  type: DownloadImageType
}

function resolveExportTextDirection(bubble: BubbleState): 'vertical' | 'horizontal' {
  const direction =
    bubble.textDirection !== 'auto' ? bubble.textDirection : bubble.autoTextDirection

  return isTextDirection(direction) ? direction : 'vertical'
}

function buildExportTextData(images: ImageData[]): ExportTextData[] {
  return images.map((image, imageIndex) => ({
    imageIndex,
    bubbles: (image.bubbleStates || []).map((bubble, bubbleIndex) => ({
      bubbleIndex,
      original: bubble.originalText || '',
      translated: bubble.translatedText || bubble.textboxText || '',
      textDirection: resolveExportTextDirection(bubble),
    })),
  }))
}

function dataUrlToPngBlob(imageDataURL: string): Blob {
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

  return new Blob(byteArrays, { type: 'image/png' })
}

export function resolveDownloadFileName(
  originalFileName: string,
  imageIndex: number,
  type: DownloadImageType,
): string {
  const fileName = originalFileName || `image_${imageIndex}.png`
  return `${type}_${fileName.replace(/\.[^/.]+$/, '')}.png`
}

export function collectDownloadImageEntries(images: ImageData[]): DownloadImageEntry[] {
  const entries: DownloadImageEntry[] = []

  for (const [index, imgData] of images.entries()) {
    if (imgData.translatedDataURL) {
      entries.push({ index, type: 'translated' })
    } else if (imgData.originalDataURL) {
      entries.push({ index, type: 'original' })
    }
  }

  return entries
}

export function useExportImport() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const toast = useToast()

  const isDownloading = ref(false)
  const downloadProgress = ref(0)
  const downloadProgressText = ref('')
  const isImporting = ref(false)
  const importProgress = ref(0)
  const importProgressText = ref('')
  let importProgressResetTimer: ReturnType<typeof setTimeout> | null = null
  let downloadProgressResetTimer: ReturnType<typeof setTimeout> | null = null

  const canExportText = computed(() => imageStore.hasImages)
  const canImportText = computed(() => imageStore.hasImages)
  const canDownload = computed(() => imageStore.hasImages)

  function clearImportProgressResetTimer(): void {
    if (importProgressResetTimer) {
      clearTimeout(importProgressResetTimer)
      importProgressResetTimer = null
    }
  }

  function clearDownloadProgressResetTimer(): void {
    if (downloadProgressResetTimer) {
      clearTimeout(downloadProgressResetTimer)
      downloadProgressResetTimer = null
    }
  }

  function scheduleImportProgressReset(): void {
    clearImportProgressResetTimer()
    importProgressResetTimer = setTimeout(() => {
      importProgressResetTimer = null
      importProgress.value = 0
      importProgressText.value = ''
    }, 2000)
  }

  function scheduleDownloadProgressReset(): void {
    clearDownloadProgressResetTimer()
    downloadProgressResetTimer = setTimeout(() => {
      downloadProgressResetTimer = null
      downloadProgress.value = 0
      downloadProgressText.value = ''
    }, 2000)
  }

  function exportText(): void {
    const allImages = imageStore.images
    if (allImages.length === 0) {
      toast.warning('没有可导出的图片文本')
      return
    }

    const exportData = buildExportTextData(allImages)
    const jsonData = JSON.stringify(exportData, null, 2)
    const blob = new Blob([jsonData], { type: 'application/json' })
    const now = new Date()
    const dateStr = now.toISOString().replace(/[-:T]/g, '').slice(0, 15)
    triggerBlobDownload(blob, `translations_${dateStr}.json`)

    toast.success('文本导出成功！')
  }

  function exportTextToJson(): ExportTextData[] | null {
    const allImages = imageStore.images
    if (allImages.length === 0) return null

    return buildExportTextData(allImages)
  }

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
      const fileContent = await readFileAsText(jsonFile)

      importProgress.value = 10
      importProgressText.value = '解析 JSON 数据...'

      const importedData = parseImportTextData(JSON.parse(fileContent) as unknown)

      let updatedImages = 0
      let updatedBubbles = 0

      const textStyle = settingsStore.settings.textStyle
      const currentFontSize = textStyle.autoFontSize ? 'auto' : textStyle.fontSize
      const currentFontFamily = textStyle.fontFamily
      const currentTextColor = textStyle.textColor
      const currentFillColor = textStyle.fillColor

      importProgress.value = 20
      importProgressText.value = '开始更新图片...'

      const totalImages = importedData.length
      let processedImages = 0

      const imagesToReRender: number[] = []

      for (const imageData of importedData) {
        processedImages++
        const progress = 20 + (processedImages / totalImages) * 60 // 从 20% 到 80%
        importProgress.value = progress
        importProgressText.value = `处理图片 ${processedImages}/${totalImages}`

        const imageIndex = imageData.imageIndex

        if (imageIndex < 0 || imageIndex >= imageStore.images.length) {
          continue
        }

        const image = imageStore.images[imageIndex]
        if (!image) continue

        let imageUpdated = false

        if (!image.bubbleStates) {
          image.bubbleStates = []
        }
        if (!image.bubbleTexts) {
          image.bubbleTexts = []
        }
        if (!image.originalTexts) {
          image.originalTexts = []
        }

        for (const bubbleData of imageData.bubbles) {
          const bubbleIndex = bubbleData.bubbleIndex
          const original = bubbleData.original
          const translated = bubbleData.translated
          const rawDir = bubbleData.textDirection as string | undefined
          const textDirection: 'vertical' | 'horizontal' =
            rawDir === 'vertical' || rawDir === 'horizontal' ? rawDir : 'vertical'

          while (image.bubbleTexts.length <= bubbleIndex) {
            image.bubbleTexts.push('')
          }
          while (image.originalTexts.length <= bubbleIndex) {
            image.originalTexts.push('')
          }

          if (original) image.originalTexts[bubbleIndex] = original
          if (translated) image.bubbleTexts[bubbleIndex] = translated

          while (image.bubbleStates.length <= bubbleIndex) {
            image.bubbleStates.push(
              createDefaultBubbleState(
                currentFontSize,
                currentFontFamily,
                currentTextColor,
                currentFillColor
              )
            )
          }

          const bubbleState = image.bubbleStates[bubbleIndex]
          if (bubbleState) {
            if (original) bubbleState.originalText = original
            if (translated) {
              bubbleState.translatedText = translated
              bubbleState.textboxText = translated
            }
            bubbleState.textDirection = textDirection

            imageUpdated = true
            updatedBubbles++
          }
        }

        if (imageUpdated && image.bubbleStates) {
          image.bubbleTexts = image.bubbleStates.map(bs => bs.translatedText || '')
        }

        if (imageUpdated) {
          updatedImages++
          image.hasUnsavedChanges = true

          if (image.translatedDataURL && image.bubbleStates && image.bubbleStates.length > 0) {
            imagesToReRender.push(imageIndex)
          }
        }
      }

      if (imagesToReRender.length > 0) {
        importProgress.value = 80
        importProgressText.value = '开始渲染图片...'
        toast.info('正在渲染图片，请稍候...', 0)

        for (let i = 0; i < imagesToReRender.length; i++) {
          const imageIndex = imagesToReRender[i]
          if (imageIndex === undefined) continue

          const img = imageStore.images[imageIndex]
          if (!img || !img.bubbleStates) continue

          importProgress.value = 80 + (i / imagesToReRender.length) * 20
          importProgressText.value = `渲染图片 ${i + 1}/${imagesToReRender.length}`

          try {
            let cleanImageBase64 = ''
            if (img.cleanImageData) {
              cleanImageBase64 = img.cleanImageData.includes('base64,')
                ? img.cleanImageData.split('base64,')[1] || ''
                : img.cleanImageData
            } else if (img.originalDataURL) {
              cleanImageBase64 = img.originalDataURL.includes('base64,')
                ? img.originalDataURL.split('base64,')[1] || ''
                : img.originalDataURL
            }

            if (!cleanImageBase64) {
              continue
            }

            const result = await executeRender(buildEditRenderInput({
              imageIndex,
              cleanImage: cleanImageBase64,
              bubbleStates: img.bubbleStates,
              settings: settingsStore.settings,
            }))

            if (result.finalImage) {
              imageStore.updateImageByIndex(imageIndex, {
                translatedDataURL: `data:image/png;base64,${result.finalImage}`,
                bubbleStates: result.bubbleStates || img.bubbleStates,
                hasUnsavedChanges: true,
              })
            }
          } catch {
            // Individual render failures do not block importing the remaining text entries.
          }
        }
      }

      importProgress.value = 100
      importProgressText.value = '导入完成'

      const reRenderedCount = imagesToReRender.length
      const message =
        reRenderedCount > 0
          ? `导入成功！更新了 ${updatedImages} 张图片中的 ${updatedBubbles} 个气泡文本，重渲染了 ${reRenderedCount} 张图片`
          : `导入成功！更新了 ${updatedImages} 张图片中的 ${updatedBubbles} 个气泡文本`
      toast.success(message)
    } catch (error) {
      toast.error(`导入失败: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isImporting.value = false
      scheduleImportProgressReset()
    }
  }

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
      fontSize: typeof fontSize === 'number' ? fontSize : TEXT_STYLE_DEFAULTS.fontSize,
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
      lineSpacing: textStyle.lineSpacing,
      textAlign: textStyle.textAlign,
      inpaintMethod: textStyle.inpaintMethod,
      textlines: [],
    }
  }

  function downloadCurrentImage(): void {
    const currentImage = imageStore.currentImage
    if (!currentImage) {
      toast.warning('没有可下载的图片')
      return
    }

    const imageDataURL = currentImage.translatedDataURL || currentImage.originalDataURL

    if (!imageDataURL) {
      toast.warning('没有可下载的图片')
      return
    }

    isDownloading.value = true

    try {
      const blob = dataUrlToPngBlob(imageDataURL)
      const imageType: DownloadImageType = currentImage.translatedDataURL
        ? 'translated'
        : 'original'
      const fileName = resolveDownloadFileName(
        currentImage.fileName,
        imageStore.currentImageIndex,
        imageType,
      )
      triggerBlobDownload(blob, fileName)

      toast.success(`下载成功: ${fileName}`)
    } catch {
      toast.error('下载失败')
    } finally {
      isDownloading.value = false
    }
  }

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
      downloadProgress.value = 5
      downloadProgressText.value = '检查图片数据...'

      const downloadEntries = collectDownloadImageEntries(allImages)
      const translatedCount = downloadEntries.filter(info => info.type === 'translated').length
      const originalCount = downloadEntries.filter(info => info.type === 'original').length

      if (downloadEntries.length === 0) {
        toast.warning('没有可下载的图片')
        return
      }

      downloadProgress.value = 10
      downloadProgressText.value = '创建下载会话...'

      const sessionResponse = await downloadStartSession(downloadEntries.length)
      if (!sessionResponse.success || !sessionResponse.session_id) {
        throw new Error(sessionResponse.error || '创建会话失败')
      }
      const sessionId = sessionResponse.session_id

      const totalImages = downloadEntries.length
      let uploadedCount = 0
      let failedCount = 0

      for (let i = 0; i < downloadEntries.length; i++) {
        const info = downloadEntries[i]
        if (!info) continue

        const imgData = allImages[info.index]
        if (!imgData) continue

        const imageDataURL =
          info.type === 'translated' ? imgData.translatedDataURL : imgData.originalDataURL
        if (!imageDataURL) continue

        const progress = 10 + (i / totalImages) * 70 // 10% - 80%
        downloadProgress.value = progress
        downloadProgressText.value = `上传图片 ${i + 1}/${totalImages}...`

        try {
          const filePath = imgData?.relativePath || imgData?.fileName || undefined

          const uploadResponse = await downloadUploadImage(sessionId, imageDataURL, i, filePath)

          if (uploadResponse.success) {
            uploadedCount++
          } else {
            failedCount++
          }
        } catch {
          failedCount++
        }
      }

      if (uploadedCount === 0) {
        throw new Error('所有图片上传失败')
      }

      downloadProgress.value = 85
      downloadProgressText.value = '打包文件...'

      const finalizeResponse = await downloadFinalize(sessionId, format)
      if (!finalizeResponse.success || !finalizeResponse.file_id) {
        throw new Error(finalizeResponse.error || '打包失败')
      }

      downloadProgress.value = 95
      downloadProgressText.value = '准备下载...'

      triggerUrlDownload(getDownloadFileUrl(finalizeResponse.file_id, format))

      downloadProgress.value = 100
      downloadProgressText.value = '下载已开始'

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

      setTimeout(async () => {
        try {
          await cleanTempFiles()
        } catch {
          // Temporary-file cleanup is best-effort after the download has started.
        }
      }, 60000)
    } catch (e) {
      toast.error(`下载失败: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      isDownloading.value = false
      scheduleDownloadProgressReset()
    }
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      clearImportProgressResetTimer()
      clearDownloadProgressResetTimer()
    })
  }

  return {
    isDownloading,
    downloadProgress,
    downloadProgressText,
    isImporting,
    importProgress,
    importProgressText,

    canExportText,
    canImportText,
    canDownload,

    exportText,
    exportTextToJson,

    importText,

    downloadCurrentImage,
    downloadAllImages,
  }
}
