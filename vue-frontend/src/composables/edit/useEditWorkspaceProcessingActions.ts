import { ref, type Ref } from 'vue'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { executeDetection } from '@/composables/translation/core/steps'
import { saveDetectionResultToImage } from '@/composables/translation/core/detectionResultWriter'
import { resolveConstraintPayloadForTranslation } from '@/utils/bookTranslationConstraints'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'
import { showToast } from '@/utils/toast'
import type { BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'

interface UseEditWorkspaceProcessingActionsOptions {
  images: Ref<ImageData[]>
  currentImage: Ref<ImageData | null | undefined>
  currentImageIndex: Ref<number>
  bubbles: Ref<BubbleState[]>
  reRenderFullImage: () => Promise<boolean> | boolean
  loadBubbleStatesFromImage: () => void
  selectFirstBubbleIfExists: () => void
}

export function useEditWorkspaceProcessingActions(options: UseEditWorkspaceProcessingActionsOptions) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const sessionStore = useSessionStore()
  const bookTranslationConstraintsStore = useBookTranslationConstraintsStore()
  const settingsStore = useSettingsStore()
  const { translateWithCurrentBubbles: translateWithBubbles } = useTranslation()

  const isProcessing = ref(false)
  const progressText = ref('处理中...')
  const progressCurrent = ref(0)
  const progressTotal = ref(0)
  const isTranslateLoading = ref(false)

  async function handleReTranslateBubble(index: number): Promise<void> {
    const bubble = options.bubbles.value[index]
    if (!bubble?.originalText) {
      console.warn('无法重新翻译：缺少气泡或原文')
      return
    }
    const expectedImageId = options.currentImage.value?.id
    const expectedBubble = bubble

    isTranslateLoading.value = true
    try {
      console.log(`开始重新翻译气泡 #${index + 1}`)
      const { translateSingleText } = await import('@/api/translate')
      const settings = settingsStore.settings

      const promptContent = settings.translation.openaiOptions.request.forceJsonOutput
        ? settings.translation.singleJsonPrompt
        : settings.translation.singleNormalPrompt

      const response = await translateSingleText({
        original_text: bubble.originalText,
        model_provider: settings.translation.provider,
        api_key: settings.translation.apiKey,
        model_name: settings.translation.modelName,
        custom_base_url: settings.translation.customBaseUrl,
        target_language: settings.targetLanguage,
        prompt_content: promptContent,
        ...resolveConstraintPayloadForTranslation({
          isBookshelfMode: Boolean(sessionStore.currentBookId && sessionStore.currentChapterId),
          constraints: bookTranslationConstraintsStore.constraints,
        }),
        openai_options: serializeOpenAICompatibleOptionsForApi(settings.translation.openaiOptions),
      })

      if (response.success && response.data?.translated_text) {
        if (!expectedImageId || options.currentImage.value?.id !== expectedImageId || options.bubbles.value[index] !== expectedBubble) {
          console.log(`翻译结果已过期，忽略气泡 #${index + 1} 的更新`)
          return
        }
        bubbleStore.updateBubble(index, { translatedText: response.data.translated_text })
        console.log(`翻译成功: "${response.data.translated_text}"`)
        if (response.data.warnings && response.data.warnings.length > 0) {
          showToast(`有 ${response.data.warnings.length} 处术语未遵守`, 'warning')
          console.warn('[SingleBubbleTranslationWarnings]', response.data.warnings)
        }
        await options.reRenderFullImage()
      } else {
        if (!expectedImageId || options.currentImage.value?.id !== expectedImageId || options.bubbles.value[index] !== expectedBubble) {
          console.log(`翻译失败结果已过期，忽略气泡 #${index + 1} 的错误提示`)
          return
        }
        console.error('翻译失败:', response.error || '未知错误')
        showToast(response.error || '重新翻译失败', 'error')
      }
    } catch (error) {
      if (!expectedImageId || options.currentImage.value?.id !== expectedImageId || options.bubbles.value[index] !== expectedBubble) {
        console.log(`翻译异常结果已过期，忽略气泡 #${index + 1} 的错误提示`)
        return
      }
      console.error('翻译出错:', error)
      showToast(error instanceof Error ? error.message : '重新翻译失败', 'error')
    } finally {
      isTranslateLoading.value = false
    }
  }

  async function autoDetectBubbles(): Promise<void> {
    const image = options.currentImage.value
    if (!image?.originalDataURL) {
      showToast('没有有效的图片用于检测', 'warning')
      return
    }
    const expectedImageId = image.id
    const expectedImageIndex = options.currentImageIndex.value

    try {
      showToast('正在自动检测文本框...', 'info')

      const result = await executeDetection({
        imageIndex: options.currentImageIndex.value,
        image,
        forceDetect: true,
        settingsSnapshot: settingsStore.settings,
      })

      if (options.currentImage.value?.id !== expectedImageId || options.currentImageIndex.value !== expectedImageIndex) {
        console.log('自动检测结果已过期，当前图片已切换，忽略本次结果')
        return
      }

      if (result.bubbleCoords.length > 0) {
        saveDetectionResultToImage(expectedImageIndex, result)
        bubbleStore.setBubbles(result.bubbleStates)
        options.selectFirstBubbleIfExists()
        showToast(`自动检测到 ${result.bubbleCoords.length} 个文本框`, 'success')
      } else {
        showToast('未检测到文本框', 'info')
      }
    } catch (error) {
      if (options.currentImage.value?.id !== expectedImageId) {
        console.log('自动检测失败结果已过期，忽略当前图片切换后的错误提示')
        return
      }
      console.error('自动检测失败:', error)
      showToast('自动检测失败', 'error')
    }
  }

  async function detectAllImages(): Promise<void> {
    if (options.images.value.length <= 1) {
      showToast('至少需要两张图片才能执行批量检测', 'warning')
      return
    }

    if (!confirm('此操作将对所有图片进行文本框检测，可能会覆盖已有的检测结果。确定继续吗？')) {
      return
    }

    const originalIndex = options.currentImageIndex.value
    const totalImages = options.images.value.length

    isProcessing.value = true
    progressText.value = '批量检测中'
    progressTotal.value = totalImages
    progressCurrent.value = 0

    try {
      let totalDetected = 0

      for (let i = 0; i < totalImages; i++) {
        const image = options.images.value[i]
        if (!image?.originalDataURL) continue

        progressCurrent.value = i + 1

        try {
          const result = await executeDetection({
            imageIndex: i,
            image,
            forceDetect: true,
            settingsSnapshot: settingsStore.settings,
          })

          if (result.bubbleCoords.length > 0) {
            saveDetectionResultToImage(i, result)
            totalDetected += result.bubbleCoords.length

            if (i === options.currentImageIndex.value) {
              options.loadBubbleStatesFromImage()
            }
          }
        } catch (error) {
          console.error(`图片 ${i + 1} 检测失败:`, error)
        }
      }

      progressText.value = '检测完成'
      progressCurrent.value = totalImages

      if (originalIndex !== options.currentImageIndex.value) {
        imageStore.setCurrentImageIndex(originalIndex)
      }
      options.loadBubbleStatesFromImage()

      showToast(`批量检测完成！共处理 ${totalImages} 张图片，检测到 ${totalDetected} 个文本框`, 'success')

      setTimeout(() => {
        isProcessing.value = false
      }, 2000)
    } catch (error) {
      console.error('批量检测失败:', error)
      showToast('批量检测失败', 'error')
      isProcessing.value = false
    }
  }

  async function translateWithCurrentBubbles(): Promise<void> {
    const image = options.currentImage.value
    if (!image?.originalDataURL) {
      showToast('没有有效的图片用于翻译', 'warning')
      return
    }

    if (options.bubbles.value.length === 0) {
      showToast('没有文本框可用于翻译，请先检测或添加文本框', 'warning')
      return
    }

    showToast('正在使用当前文本框翻译...', 'info')

    try {
      const success = await translateWithBubbles()
      if (success) {
        showToast('翻译成功！', 'success')
        options.selectFirstBubbleIfExists()
      }
    } catch (error) {
      console.error('翻译失败:', error)
      showToast(`翻译失败: ${error instanceof Error ? error.message : '未知错误'}`, 'error')
    }
  }

  return {
    isProcessing,
    progressText,
    progressCurrent,
    progressTotal,
    isTranslateLoading,
    handleReTranslateBubble,
    autoDetectBubbles,
    detectAllImages,
    translateWithCurrentBubbles,
  }
}
