<script setup lang="ts">
/**
 * 翻译页面视图组件
 * 提供图片上传、翻译设置、翻译执行和编辑模式功能
 * 
 * 核心功能：
 * - 图片上传（支持拖拽、多图片、PDF、MOBI/AZW）
 * - 翻译设置侧边栏
 * - 缩略图列表
 * - 翻译进度显示
 * - 翻译结果显示
 * - 编辑模式入口
 */

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useSessionStore } from '@/stores/sessionStore'
import { showToast } from '@/utils/toast'
import { cleanDebugFiles, cleanTempFiles } from '@/api/system'
import ImageUpload from '@/components/translate/ImageUpload.vue'
import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'
import ImageResultDisplay from '@/components/translate/ImageResultDisplay.vue'
import FirstTimeGuide from '@/components/common/FirstTimeGuide.vue'
import { useValidation } from '@/composables/useValidation'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { useTranslateInit } from '@/composables/useTranslateInit'
import TranslationProgress from '@/components/translate/TranslationProgress.vue'
import SponsorModal from '@/components/bookshelf/SponsorModal.vue'
import ThumbnailSidebar from '@/components/translate/ThumbnailSidebar.vue'
import SettingsModal from '@/components/settings/SettingsModal.vue'
import EditWorkspace from '@/components/edit/EditWorkspace.vue'
import ProgressBar from '@/components/common/ProgressBar.vue'
import { getEffectiveDirection } from '@/types/bubble'

import WebImportModal from '@/components/translate/WebImportModal.vue'
import WebImportDisclaimer from '@/components/translate/WebImportDisclaimer.vue'

// 路由
const route = useRoute()

// Stores
const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const sessionStore = useSessionStore()
const bubbleStore = useBubbleStore()

// 配置验证
const { 
  validateBeforeTranslation, 
  initValidation 
} = useValidation()

// 翻译功能
const translation = useTranslation()

// 导出导入功能已移至具体按钮事件处理函数中

// 翻译页面初始化
const translateInit = useTranslateInit()

// ============================================================
// 状态定义
// ============================================================

/** 是否显示设置模态框 */
const showSettingsModal = ref(false)

/** 设置模态框初始Tab（用于插件管理直接跳转） */
const settingsInitialTab = ref<string | undefined>(undefined)

/** 是否显示赞助模态框 */
const showSponsorModal = ref(false)

/** 是否处于编辑模式 */
const isEditMode = ref(false)

/** ImageUpload 组件引用 */
const imageUploadRef = ref<InstanceType<typeof ImageUpload> | null>(null)

/** ImageResultDisplay 组件引用 */
const imageResultRef = ref<InstanceType<typeof ImageResultDisplay> | null>(null)

// ============================================================
// 计算属性
// ============================================================

/** 当前图片 */
const currentImage = computed(() => imageStore.currentImage)

/** 是否有图片 */
const hasImages = computed(() => imageStore.hasImages)

/** 图片总数 */
const imageCount = computed(() => imageStore.imageCount)

/** 当前图片索引（从1开始显示） */
const currentImageNum = computed(() => imageStore.currentImageIndex + 1)

/** 是否可以翻译（有图片且不在批量翻译中） */
const canTranslate = computed(() => 
  hasImages.value && !imageStore.isBatchTranslationInProgress
)

/** 是否可以切换上一张 */
const canGoPrevious = computed(() => imageStore.canGoPrevious)

/** 是否可以切换下一张 */
const canGoNext = computed(() => imageStore.canGoNext)

/** 批量翻译是否进行中 */
const isBatchTranslating = computed(() => imageStore.isBatchTranslationInProgress)

/** 翻译进度百分比 */
const translationProgress = computed(() => {
  if (!isBatchTranslating.value) return 0
  const completed = imageStore.completedImageCount
  const total = imageStore.imageCount
  return total > 0 ? Math.round((completed / total) * 100) : 0
})

/** 翻译进度文本 */
const progressText = computed(() => {
  return `${imageStore.completedImageCount}/${imageStore.imageCount}`
})

/** 是否有翻译失败的图片 */
const hasFailedImages = computed(() => imageStore.failedImageCount > 0)

/** 是否为书架模式（有书籍和章节参数） */
const isBookshelfMode = computed(() => {
  return !!route.query.book && !!route.query.chapter
})

/** 当前书籍ID */
const currentBookId = computed(() => route.query.book as string | undefined)

/** 当前章节ID */
const currentChapterId = computed(() => route.query.chapter as string | undefined)

/** 当前书籍标题（从 translateInit 获取） */
const currentBookTitle = computed(() => translateInit.currentBookTitle.value)

/** 当前章节标题（从 translateInit 获取） */
const currentChapterTitle = computed(() => translateInit.currentChapterTitle.value)

/** 页面标题（书架模式下显示书籍和章节名） */
const pageTitle = computed(() => {
  if (isBookshelfMode.value && currentChapterTitle.value && currentBookTitle.value) {
    return `${currentChapterTitle.value} - ${currentBookTitle.value}`
  }
  return 'Saber-Translator'
})

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  // 【关键修复】复刻原版多页应用的行为：每次进入翻译页面都是全新的空白状态
  // 原版行为：每次访问 /translate 都是一个全新的 HTTP 请求，JS 状态从零开始
  // Vue SPA 行为：Pinia store 状态在整个应用生命周期内持久存在
  // 修复：无论是书架模式还是快速翻译模式，都清空旧数据
  imageStore.clearImages()
  bubbleStore.clearBubbles()
  
  // 使用 useTranslateInit 进行完整初始化
  // 包括：设置初始化、字体列表、提示词、主题、书架模式会话加载
  await translateInit.initializeApp()
  
  // 初始化配置验证（延迟显示首次使用引导）
  initValidation()
  
  // 添加全局键盘事件监听
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  // 移除全局键盘事件监听
  window.removeEventListener('keydown', handleKeydown)
})

// 监听路由参数变化
watch(
  () => [route.query.book, route.query.chapter],
  async ([newBook, newChapter], [oldBook, oldChapter]) => {
    // 【修复】处理所有路由参数变化场景，复刻原版多页应用的行为
    
    if (newBook && newChapter) {
      // 场景1：进入书架模式（加载新章节）
      // 关键修复：在任何异步操作之前，立即同步清空旧数据
      imageStore.clearImages()
      bubbleStore.clearBubbles()
      
      await loadChapterSession()
    } else if (oldBook && oldChapter && !newBook && !newChapter) {
      // 场景2：从书架模式切换到快速翻译模式（参数消失）
      // 同样需要清空数据，复刻"全新页面"的行为
      imageStore.clearImages()
      bubbleStore.clearBubbles()
      // 清空书籍/章节上下文
      await translateInit.initializeBookChapterContext()
      console.log('[TranslateView] 从书架模式切换到快速翻译模式，已清空数据')
    }
  }
)

// 监听页面标题变化，更新 document.title
watch(
  pageTitle,
  (newTitle) => {
    if (typeof document !== 'undefined') {
      document.title = newTitle
    }
  },
  { immediate: true }
)

// ============================================================
// 方法
// ============================================================

/**
 * 加载章节会话
 * 当路由参数变化时重新加载章节数据
 */
async function loadChapterSession() {
  if (!currentBookId.value || !currentChapterId.value) return
  
  try {
    // 使用 translateInit 的初始化方法，它会正确调用 loadSessionByPath
    await translateInit.initializeBookChapterContext()
    
  } catch (error) {
    console.error('加载章节会话失败:', error)
    showToast('加载章节会话失败', 'error')
  }
}

/**
 * 处理上传完成事件
 * 复刻原版 main.js handleFiles 完成逻辑：
 * 1. 对所有图片按文件名进行自然排序
 * 2. 跳转显示第一张图片
 */
function handleUploadComplete(count: number) {
  console.log(`上传完成，共 ${count} 张图片`)
  
  // 复刻原版逻辑：如果有图片，先排序再跳转到第一张
  if (imageStore.hasImages) {
    // 按文件名自然排序（复刻 sortImagesByName）
    imageStore.sortImagesByFileName()
    // 跳转到第一张图片（复刻 switchImage(0)）
    translateInit.switchImage(0)
  }
}

/**
 * 应用设置选项接口
 */
interface ApplySettingsOptions {
  fontSize: boolean
  fontFamily: boolean
  layoutDirection: boolean
  textColor: boolean
  fillColor: boolean
  strokeEnabled: boolean
  strokeColor: boolean
  strokeWidth: boolean
}

/**
 * 处理应用设置到全部
 * 【复刻原版 main.js applySettingsToAll】
 * 核心逻辑：从当前图片的 bubbleStates[0] 读取设置，应用到所有图片的 bubbleStates
 * @param options - 选择要应用的设置项
 */
async function handleApplyToAll(options: ApplySettingsOptions) {
  // 【复刻原版】检查当前图片是否有 bubbleStates
  const currentImg = currentImage.value
  if (!currentImg || !currentImg.bubbleStates || currentImg.bubbleStates.length === 0) {
    showToast('请先选择一张已翻译的图片', 'warning')
    return
  }
  
  if (imageStore.images.length <= 1) {
    showToast('只有一张图片，无需应用到全部', 'info')
    return
  }

  // 检查是否至少选择了一个选项
  const hasSelectedOption = Object.values(options).some(v => v)
  if (!hasSelectedOption) {
    showToast('请至少选择一个要应用的设置项', 'warning')
    return
  }

  try {
    
    // 【复刻原版】从当前图片的第一个气泡读取设置（而不是全局 settingsStore）
    // 注：前面已经检查过 bubbleStates.length > 0，所以这里使用非空断言
    const source = currentImg.bubbleStates![0]!
    
    // 构建要应用的设置对象（复刻原版逻辑）
    const settingsToApply: Record<string, unknown> = {}
    
    if (options.fontSize) {
      settingsToApply.fontSize = source.fontSize
    }
    if (options.fontFamily) {
      settingsToApply.fontFamily = source.fontFamily
    }
    if (options.layoutDirection) {
      // 【复刻原版修复C】textDirection 如果是 'auto' 则转为 'vertical'
      settingsToApply.textDirection = source.textDirection === 'auto' ? 'vertical' : source.textDirection
    }
    if (options.textColor) {
      settingsToApply.textColor = source.textColor
    }
    if (options.fillColor) {
      settingsToApply.fillColor = source.fillColor
    }
    if (options.strokeEnabled) {
      settingsToApply.strokeEnabled = source.strokeEnabled
    }
    if (options.strokeColor) {
      settingsToApply.strokeColor = source.strokeColor
    }
    if (options.strokeWidth) {
      settingsToApply.strokeWidth = source.strokeWidth
    }

    // 辅助函数：应用设置到单个气泡
    const applySettingsToBubble = (bubble: typeof bubbleStore.bubbles[0]) => {
      const updatedBubble = { ...bubble }
      if (options.fontSize && settingsToApply.fontSize !== undefined) {
        updatedBubble.fontSize = settingsToApply.fontSize as number
      }
      if (options.fontFamily && settingsToApply.fontFamily !== undefined) {
        updatedBubble.fontFamily = settingsToApply.fontFamily as string
      }
      if (options.layoutDirection && settingsToApply.textDirection !== undefined) {
        // settingsToApply.textDirection 已在第 316 行处理，确保不是 'auto'
        updatedBubble.textDirection = settingsToApply.textDirection as 'vertical' | 'horizontal'
      }
      if (options.textColor && settingsToApply.textColor !== undefined) {
        updatedBubble.textColor = settingsToApply.textColor as string
      }
      if (options.fillColor && settingsToApply.fillColor !== undefined) {
        updatedBubble.fillColor = settingsToApply.fillColor as string
      }
      if (options.strokeEnabled && settingsToApply.strokeEnabled !== undefined) {
        updatedBubble.strokeEnabled = settingsToApply.strokeEnabled as boolean
      }
      if (options.strokeColor && settingsToApply.strokeColor !== undefined) {
        updatedBubble.strokeColor = settingsToApply.strokeColor as string
      }
      if (options.strokeWidth && settingsToApply.strokeWidth !== undefined) {
        updatedBubble.strokeWidth = settingsToApply.strokeWidth as number
      }
      return updatedBubble
    }

    // 更新所有图片的气泡状态
    let updatedCount = 0
    const images = imageStore.images
    
    for (let i = 0; i < images.length; i++) {
      const image = images[i]
      if (!image) continue
      if (image.bubbleStates && image.bubbleStates.length > 0) {
        // 使用辅助函数更新每个气泡的设置
        const updatedBubbleStates = image.bubbleStates.map(applySettingsToBubble)
        
        // 更新图片的气泡状态
        imageStore.updateImageByIndex(i, { bubbleStates: updatedBubbleStates })
        updatedCount++
      }
    }

    // 同时更新当前气泡 store 中的气泡（如果有）
    if (bubbleStore.bubbles.length > 0) {
      const updatedCurrentBubbles = bubbleStore.bubbles.map(applySettingsToBubble)
      bubbleStore.setBubbles(updatedCurrentBubbles)
    }

    // 构建应用的设置项描述
    const appliedItems: string[] = []
    if (options.fontSize) appliedItems.push('字号')
    if (options.fontFamily) appliedItems.push('字体')
    if (options.layoutDirection) appliedItems.push('排版方向')
    if (options.textColor) appliedItems.push('文字颜色')
    if (options.fillColor) appliedItems.push('填充颜色')
    if (options.strokeEnabled) appliedItems.push('描边开关')
    if (options.strokeColor) appliedItems.push('描边颜色')
    if (options.strokeWidth) appliedItems.push('描边宽度')

    // 【修复P1】逐张重新渲染已翻译的图片（与原版 applySettingsToAll 一致）
    // 原版判定条件：translatedDataURL 存在即可，背景用 clean → original 兜底
    const imagesToReRender: number[] = []
    for (let i = 0; i < images.length; i++) {
      const img = images[i]
      // 只要有翻译结果且有气泡，就可以重渲染（背景会兜底）
      if (img && img.translatedDataURL && img.bubbleStates && img.bubbleStates.length > 0) {
        imagesToReRender.push(i)
      }
    }

    if (imagesToReRender.length > 0) {
      const { apiClient } = await import('@/api/client')
      const layoutDir = settingsStore.settings.textStyle.layoutDirection
      const isAutoLayout = layoutDir === 'auto'

      for (let idx = 0; idx < imagesToReRender.length; idx++) {
        const imageIndex = imagesToReRender[idx]
        if (imageIndex === undefined) continue
        const img = imageStore.images[imageIndex]
        if (!img || !img.bubbleStates) continue

        try {
          // 【修复P1】背景兜底策略：clean → original
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
            console.log(`handleApplyToAll: 图片 ${imageIndex} 使用原图作为背景（兜底）`)
          }
          
          if (!cleanImageBase64) {
            console.log(`handleApplyToAll: 图片 ${imageIndex} 没有可用的背景图，跳过`)
            continue
          }

          const bubbleStatesForApi = img.bubbleStates.map(bs => ({
            translatedText: bs.translatedText || '',
            coords: bs.coords,
            fontSize: bs.fontSize || settingsStore.settings.textStyle.fontSize,
            fontFamily: bs.fontFamily || settingsStore.settings.textStyle.fontFamily,
            textDirection: getEffectiveDirection(bs),
            textColor: bs.textColor || settingsStore.settings.textStyle.textColor,
            rotationAngle: bs.rotationAngle || 0,
            position: bs.position || { x: 0, y: 0 },
            strokeEnabled: bs.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
            strokeColor: bs.strokeColor || settingsStore.settings.textStyle.strokeColor,
            strokeWidth: bs.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
          }))

          const response = await apiClient.post<{ rendered_image?: string; error?: string }>(
            '/api/re_render_image',
            {
              clean_image: cleanImageBase64,
              bubble_texts: bubbleStatesForApi.map(s => s.translatedText),
              bubble_coords: bubbleStatesForApi.map(s => s.coords),
              fontSize: settingsStore.settings.textStyle.fontSize,
              fontFamily: settingsStore.settings.textStyle.fontFamily,
              textDirection: isAutoLayout ? 'vertical' : layoutDir,
              textColor: settingsStore.settings.textStyle.textColor,
              bubble_states: bubbleStatesForApi,
              use_individual_styles: true,
              use_inpainting: false,
              use_lama: false,
              fillColor: null,
              is_font_style_change: true,
              strokeEnabled: settingsStore.settings.textStyle.strokeEnabled,
              strokeColor: settingsStore.settings.textStyle.strokeColor,
              strokeWidth: settingsStore.settings.textStyle.strokeWidth,
            }
          )

          if (response.rendered_image) {
            imageStore.updateImageByIndex(imageIndex, {
              translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
              hasUnsavedChanges: true
            })
          }
        } catch (err) {
          console.error(`重渲染图片 ${imageIndex} 失败:`, err)
        }
      }
    }

    showToast(`已将 ${appliedItems.join('、')} 应用到 ${updatedCount} 张图片`, 'success')
    console.log(`[TranslateView] 应用设置到全部完成，更新了 ${updatedCount} 张图片，重渲染了 ${imagesToReRender.length} 张`)
    
  } catch (error) {
    console.error('应用设置到全部失败:', error)
    showToast('应用设置失败', 'error')
  }
}

/**
 * 翻译当前图片
 */
async function translateCurrentImage() {
  if (!currentImage.value) return
  
  // 验证翻译配置（useTranslation 内部也会验证，这里提前验证以便显示引导）
  if (!validateBeforeTranslation('normal')) {
    return
  }
  
  await translation.translateCurrentImage()
}

/**
 * 翻译所有图片
 */
async function translateAllImages() {
  if (!hasImages.value) return
  
  // 验证翻译配置
  if (!validateBeforeTranslation('normal')) {
    return
  }
  
  await translation.translateAllImages()
}

/**
 * 高质量翻译
 */
async function startHqTranslation() {
  if (!hasImages.value) return
  
  // 验证高质量翻译配置
  if (!validateBeforeTranslation('hq')) {
    return
  }
  
  await translation.executeHqTranslation()
}

/**
 * AI 校对
 */
async function startProofreading() {
  if (!hasImages.value) return
  
  // 验证 AI 校对配置
  if (!validateBeforeTranslation('proofread')) {
    return
  }
  
  await translation.executeProofreading()
}

/**
 * 仅消除文字
 */
async function removeTextOnly() {
  if (!currentImage.value) return
  await translation.removeTextOnly()
}

/**
 * 消除所有图片文字
 */
async function removeAllText() {
  if (!hasImages.value) return
  await translation.removeAllTexts()
}


/**
 * 删除当前图片
 * 对齐原版 events.js handleDeleteCurrent
 */
function deleteCurrentImage() {
  if (!currentImage.value) return
  const fileName = currentImage.value.fileName || `图片 ${imageStore.currentImageIndex + 1}`
  if (confirm(`确定要删除当前图片 (${fileName}) 吗？`)) {
    imageStore.deleteCurrentImage()
    showToast('图片已删除', 'success')
  }
}

/**
 * 清除所有图片
 * 对齐原版 events.js handleClearAll
 */
function clearAllImages() {
  if (!hasImages.value) return
  if (confirm('确定要清除所有图片吗？这将丢失所有未保存的进度。')) {
    imageStore.clearImages()
    showToast('所有图片已清除', 'success')
  }
}

/**
 * 清理临时文件
 * 调用后端API清理调试文件和临时下载文件
 */
async function handleCleanTempFiles() {
  try {
    
    // 清理调试文件
    const debugResult = await cleanDebugFiles()
    
    // 清理临时下载文件
    const tempResult = await cleanTempFiles()
    
    if (debugResult.success && tempResult.success) {
      showToast('临时文件清理完成', 'success')
    } else {
      // 部分成功
      const messages: string[] = []
      if (!debugResult.success) {
        messages.push('调试文件清理失败')
      }
      if (!tempResult.success) {
        messages.push('临时文件清理失败')
      }
      showToast(messages.join('，'), 'warning')
    }
  } catch (error) {
    showToast('清理临时文件失败', 'error')
  }
}

/**
 * 切换上一张图片
 * 使用 translateInit.switchImage 以正确保存/加载气泡状态
 */
function goToPrevious() {
  translateInit.goToPrevious()
}

/**
 * 切换下一张图片
 * 使用 translateInit.switchImage 以正确保存/加载气泡状态
 */
function goToNext() {
  translateInit.goToNext()
}

/**
 * 进入/退出编辑模式
 */
function toggleEditMode() {
  isEditMode.value = !isEditMode.value
}


/**
 * 处理重新翻译失败图片
 * 重新翻译所有标记为失败的图片
 */
async function handleRetryFailed() {
  if (!hasFailedImages.value) {
    showToast('没有失败的图片需要重新翻译', 'info')
    return
  }
  
  // 验证翻译配置
  if (!validateBeforeTranslation('normal')) {
    return
  }
  
  await translation.retryFailedImages()
}

/**
 * 保存当前会话
 */
async function saveCurrentSession() {
  if (!hasImages.value) {
    showToast('没有可保存的内容', 'warning')
    return
  }
  
  if (!currentBookId.value || !currentChapterId.value) {
    return
  }
  
  try {
    const success = await sessionStore.saveChapterSession(currentBookId.value, currentChapterId.value)
    if (success) {
      showToast('章节进度已保存', 'success')
    } else {
      showToast('保存失败', 'error')
    }
  } catch (error) {
    console.error('保存会话失败:', error)
    showToast('保存失败: ' + (error instanceof Error ? error.message : '未知错误'), 'error')
  }
}


/**
 * 打开设置模态框
 * @param initialTab - 可选的初始Tab，如 'plugins'
 */
function openSettings(initialTab?: string) {
  settingsInitialTab.value = initialTab
  showSettingsModal.value = true
}

/**
 * 打开插件管理
 * 【修复问题2】复刻原版：点击插件管理按钮直接进入插件管理界面
 */
function openPlugins() {
  openSettings('plugins')
}

/**
 * 处理设置保存
 */
function handleSettingsSave() {
  showToast('设置已保存', 'success')
}

/**
 * 打开赞助模态框
 */
function openSponsor() {
  showSponsorModal.value = true
}

/**
 * 显示功能开发中提示
 */
function showFeatureNotice() {
  showToast('🌙 该功能正在开发中，敬请期待！', 'info')
}

/**
 * 处理键盘事件（非编辑模式）
 * 【复刻原版 events.js handleGlobalKeyDown】
 */
function handleKeydown(event: KeyboardEvent) {
  const target = event.target as HTMLElement
  
  // 【复刻原版修复D】检查是否在文本输入框中
  // 原版豁免范围：input[type="text"], textarea, [contenteditable="true"], #bubbleTextEditor
  const isInTextInput = 
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target.getAttribute('contenteditable') === 'true' ||
    target.id === 'bubbleTextEditor'
  
  // 如果在文本输入框中，不拦截键盘事件，让浏览器处理默认行为
  if (isInTextInput) {
    return
  }
  
  // 编辑模式下的快捷键由 EditWorkspace 组件处理
  if (isEditMode.value) {
    return
  }
  
  // 非编辑模式：Alt + 方向键
  if (event.altKey) {
    switch (event.key) {
      case 'ArrowLeft':
        // Alt + ←：上一张图片
        event.preventDefault()
        goToPrevious()
        break
      case 'ArrowRight':
        // Alt + →：下一张图片
        event.preventDefault()
        goToNext()
        break
      case 'ArrowUp':
        // Alt + ↑：字号+1（仅非自动字号时）
        event.preventDefault()
        if (!settingsStore.settings.textStyle.autoFontSize) {
          const currentSize = settingsStore.settings.textStyle.fontSize
          settingsStore.updateTextStyle({ fontSize: currentSize + 1 })
        }
        break
      case 'ArrowDown':
        // Alt + ↓：字号-1（仅非自动字号时，最小10）
        event.preventDefault()
        if (!settingsStore.settings.textStyle.autoFontSize) {
          const currentSize = settingsStore.settings.textStyle.fontSize
          settingsStore.updateTextStyle({ fontSize: Math.max(10, currentSize - 1) })
        }
        break
    }
  }
}

/**
 * 处理自动字号开关变更
 * 【复刻原版 events.js handleAutoFontSizeChange】
 * 核心逻辑：
 * - 开启自动字号：调用 reRenderFullImage(..., useAutoFontSize=true) 重新计算字号并渲染
 * - 关闭自动字号：将所有气泡设为输入框中的固定字号，然后渲染
 * @param isAutoFontSize - 自动字号是否启用
 */
async function handleAutoFontSizeChanged(isAutoFontSize: boolean) {
  const image = currentImage.value
  if (!image || !image.translatedDataURL) {
    // 没有已翻译的图片，仅影响下次翻译（与原版一致）
    console.log(`自动字号设置变更: ${isAutoFontSize} (仅影响下次翻译)`)
    return
  }

  const bubbleStates = image.bubbleStates
  if (!bubbleStates || !Array.isArray(bubbleStates) || bubbleStates.length === 0) {
    console.log('当前图片没有 bubbleStates，跳过重渲染')
    return
  }

  console.log(`自动字号设置变更: ${isAutoFontSize}，将重新渲染...`)

  if (isAutoFontSize) {
    // 【复刻原版】开启自动字号：重新计算每个气泡的字号
    // 原版调用 editMode.reRenderFullImage(false, false, true)
    // 第三个参数 true 表示 useAutoFontSize，对应后端 autoFontSize 参数
    console.log('自动字号已开启，重新计算字号并渲染...')
    
    try {
      const { apiClient } = await import('@/api/client')
      
      // 提取 clean_image 的 base64 部分
      let cleanImageBase64 = ''
      if (image.cleanImageData) {
        const cleanData = image.cleanImageData
        cleanImageBase64 = cleanData.includes('base64,') 
          ? (cleanData.split('base64,')[1] || '') 
          : cleanData
      } else if (image.originalDataURL) {
        cleanImageBase64 = image.originalDataURL.includes('base64,')
          ? (image.originalDataURL.split('base64,')[1] || '')
          : image.originalDataURL
      }
      
      if (!cleanImageBase64) {
        console.log('没有可用的背景图，跳过重渲染')
        return
      }
      
      const bubbleStatesForApi = bubbleStates.map((bs) => ({
        translatedText: bs.translatedText || '',
        coords: bs.coords,
        fontSize: bs.fontSize || settingsStore.settings.textStyle.fontSize,  // 传递当前字号，后端会根据 autoFontSize=true 重新计算
        fontFamily: bs.fontFamily || settingsStore.settings.textStyle.fontFamily,
        textDirection: getEffectiveDirection(bs),
        textColor: bs.textColor || settingsStore.settings.textStyle.textColor,
        rotationAngle: bs.rotationAngle || 0,
        position: bs.position || { x: 0, y: 0 },
        strokeEnabled: bs.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
        strokeColor: bs.strokeColor || settingsStore.settings.textStyle.strokeColor,
        strokeWidth: bs.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
      }))

      const bubbleTexts = bubbleStatesForApi.map(s => s.translatedText)
      const bubbleCoords = bubbleStatesForApi.map(s => s.coords)

      const response = await apiClient.post<{ rendered_image?: string; error?: string; bubble_states?: Array<{ fontSize?: number }> }>(
        '/api/re_render_image',
        {
          clean_image: cleanImageBase64,
          bubble_texts: bubbleTexts,
          bubble_coords: bubbleCoords,
          fontSize: settingsStore.settings.textStyle.fontSize,  // 后端需要数字类型
          fontFamily: settingsStore.settings.textStyle.fontFamily,
          textDirection: settingsStore.settings.textStyle.layoutDirection === 'auto' ? 'vertical' : settingsStore.settings.textStyle.layoutDirection,
          textColor: settingsStore.settings.textStyle.textColor,
          bubble_states: bubbleStatesForApi,
          use_individual_styles: true,
          use_inpainting: false,
          use_lama: false,
          fillColor: null,
          is_font_style_change: true,
          autoFontSize: true,  // 【修复】使用正确的参数名 autoFontSize（与原版 edit_mode.js 行 407 一致）
          strokeEnabled: settingsStore.settings.textStyle.strokeEnabled,
          strokeColor: settingsStore.settings.textStyle.strokeColor,
          strokeWidth: settingsStore.settings.textStyle.strokeWidth,
        }
      )

      if (response.rendered_image) {
        // 【复刻原版】如果后端返回了更新后的 bubble_states，需要回写字号
        if (response.bubble_states && Array.isArray(response.bubble_states)) {
          const updatedBubbles = bubbleStates.map((bs, idx) => ({
            ...bs,
            fontSize: response.bubble_states![idx]?.fontSize ?? bs.fontSize
          }))
          imageStore.updateCurrentImage({
            translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
            bubbleStates: updatedBubbles,
            hasUnsavedChanges: true
          })
          bubbleStore.setBubbles(updatedBubbles)
        } else {
          imageStore.updateCurrentImage({
            translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
            hasUnsavedChanges: true
          })
        }
        console.log('自动字号渲染成功')
      } else if (response.error) {
        console.error('自动字号渲染失败:', response.error)
        showToast('重新渲染失败: ' + response.error, 'error')
      }
    } catch (error) {
      console.error('自动字号渲染出错:', error)
    }
  } else {
    // 【复刻原版】关闭自动字号：将所有气泡设为输入框中的固定字号
    const fixedFontSize = settingsStore.settings.textStyle.fontSize
    console.log(`自动字号已关闭，使用固定字号 ${fixedFontSize} 渲染...`)
    
    // 更新所有气泡的字号
    const updatedBubbles = bubbleStates.map(bs => ({
      ...bs,
      fontSize: fixedFontSize
    }))
    
    // 更新状态
    imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
    bubbleStore.setBubbles(updatedBubbles)
    
    // 触发重渲染（复用 handleTextStyleChanged 的逻辑）
    await handleTextStyleChanged('fontSize', fixedFontSize)
  }
}

/**
 * 处理文字样式设置变更
 * 与原版 handleGlobalSettingChange 对应：更新所有气泡的对应参数，然后重新渲染
 * @param settingKey - 变更的设置项
 * @param newValue - 新值
 */
async function handleTextStyleChanged(settingKey: string, newValue: unknown) {
  const image = currentImage.value
  if (!image || !image.translatedDataURL || !image.bubbleStates || image.bubbleStates.length === 0) {
    // 没有已翻译的图片或气泡，不需要重新渲染
    return
  }

  // 注意：原版有 _isChangingFromSwitchImage 标记来避免切换图片时重渲染
  // Vue 版暂时不实现此检查，因为切换图片时不会触发设置变更事件

  // 需要重新渲染的设置项（与原版 renderSettings 一致）
  const renderSettings = ['fontSize', 'fontFamily', 'layoutDirection', 'textColor', 
                         'strokeEnabled', 'strokeColor', 'strokeWidth', 'fillColor']
  
  if (!renderSettings.includes(settingKey)) {
    return
  }

  console.log(`全局设置变更 (${settingKey}=${newValue})，准备重渲染...`)

  // 更新所有气泡的对应属性（与原版 propertyMap 一致）
  const propertyMap: Record<string, string> = {
    'fontSize': 'fontSize',
    'fontFamily': 'fontFamily',
    'layoutDirection': 'textDirection',  // UI 是 layoutDirection，状态是 textDirection
    'textColor': 'textColor',
    'strokeEnabled': 'strokeEnabled',
    'strokeColor': 'strokeColor',
    'strokeWidth': 'strokeWidth',
    'fillColor': 'fillColor'
  }

  const stateProperty = propertyMap[settingKey]
  if (stateProperty && image.bubbleStates) {
    // 【简化设计】处理 layoutDirection 变更
    if (settingKey === 'layoutDirection') {
      if (newValue === 'auto') {
        // 切换到"自动"：从备份的 autoTextDirection 恢复到 textDirection
        console.log("排版方向设置为 'auto'，从 autoTextDirection 恢复每个气泡的排版方向")
        const updatedBubbles = image.bubbleStates.map(bs => ({
          ...bs,
          // 直接用备份的检测结果，不再是 'auto'
          textDirection: (bs.autoTextDirection === 'vertical' || bs.autoTextDirection === 'horizontal') 
            ? bs.autoTextDirection 
            : 'vertical'
        }))
        imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
        bubbleStore.setBubbles(updatedBubbles)
      } else {
        // 切换到强制横排/竖排：直接赋值
        console.log(`排版方向设置为 '${newValue}'，应用到所有气泡`)
        const updatedBubbles = image.bubbleStates.map(bs => ({
          ...bs,
          textDirection: newValue as 'vertical' | 'horizontal'
        }))
        imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
        bubbleStore.setBubbles(updatedBubbles)
      }
    } else {
      // 其他设置项：正常更新
      const updatedBubbles = image.bubbleStates.map(bs => ({
        ...bs,
        [stateProperty]: newValue
      }))
      
      // 更新图片的 bubbleStates
      imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
      
      // 同步更新 bubbleStore
      bubbleStore.setBubbles(updatedBubbles)
    }
  }

  // 触发重新渲染（调用 reRenderImage API）
  // 后端需要的参数格式：clean_image, bubble_texts, bubble_coords, bubble_states
  try {
    // 获取最新的 bubbleStates（可能刚刚被更新）
    const latestImage = imageStore.currentImage
    const bubbleStates = latestImage?.bubbleStates || image.bubbleStates || []
    
    // 检查是否有有效的气泡坐标
    if (bubbleStates.length === 0 || !bubbleStates[0]?.coords) {
      console.log('没有有效的气泡坐标，跳过重渲染')
      return
    }

    // 构建 API 参数（与原版 edit_mode.js reRenderFullImage 一致）
    const layoutDir = settingsStore.settings.textStyle.layoutDirection

    // 构建气泡状态数组（与原版 bubbleStatesForApi 格式一致）
    const bubbleStatesForApi = bubbleStates.map((bs) => ({
      translatedText: bs.translatedText || '',
      coords: bs.coords,
      fontSize: bs.fontSize || settingsStore.settings.textStyle.fontSize,
      fontFamily: bs.fontFamily || settingsStore.settings.textStyle.fontFamily,
      textDirection: getEffectiveDirection(bs),
      textColor: bs.textColor || settingsStore.settings.textStyle.textColor,
      rotationAngle: bs.rotationAngle || 0,
      position: bs.position || { x: 0, y: 0 },
      strokeEnabled: bs.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
      strokeColor: bs.strokeColor || settingsStore.settings.textStyle.strokeColor,
      strokeWidth: bs.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
    }))

    const bubbleTexts = bubbleStatesForApi.map(s => s.translatedText)
    const bubbleCoords = bubbleStatesForApi.map(s => s.coords)

    // 【修复P1】提取 clean_image 的 base64 部分，原版兜底策略：clean → original
    let cleanImageBase64 = ''
    if (image.cleanImageData) {
      const cleanData = image.cleanImageData
      cleanImageBase64 = cleanData.includes('base64,') 
        ? (cleanData.split('base64,')[1] || '') 
        : cleanData
    } else if (image.originalDataURL) {
      // 兜底：使用原图作为背景
      cleanImageBase64 = image.originalDataURL.includes('base64,')
        ? (image.originalDataURL.split('base64,')[1] || '')
        : image.originalDataURL
      console.log('handleTextStyleChanged: 使用原图作为背景（兜底）')
    }
    
    if (!cleanImageBase64) {
      console.log('没有可用的背景图，跳过重渲染')
      return
    }

    const { apiClient } = await import('@/api/client')
    const response = await apiClient.post<{ rendered_image?: string; error?: string }>(
      '/api/re_render_image',
      {
        clean_image: cleanImageBase64,
        bubble_texts: bubbleTexts,
        bubble_coords: bubbleCoords,
        fontSize: settingsStore.settings.textStyle.fontSize,
        fontFamily: settingsStore.settings.textStyle.fontFamily,
        textDirection: layoutDir === 'auto' ? 'vertical' : layoutDir,
        textColor: settingsStore.settings.textStyle.textColor,
        bubble_states: bubbleStatesForApi,
        use_individual_styles: true,
        use_inpainting: false,
        use_lama: false,
        fillColor: null,
        is_font_style_change: true,
        strokeEnabled: settingsStore.settings.textStyle.strokeEnabled,
        strokeColor: settingsStore.settings.textStyle.strokeColor,
        strokeWidth: settingsStore.settings.textStyle.strokeWidth,
      }
    )

    if (response.rendered_image) {
      imageStore.updateCurrentImage({
        translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
        hasUnsavedChanges: true
      })
      console.log('设置变更后重新渲染成功')
    } else if (response.error) {
      console.error('重新渲染失败:', response.error)
    }
  } catch (error) {
    console.error('设置变更后重新渲染失败:', error)
  }
}

/**
 * 点击缩略图切换图片
 * 使用 translateInit.switchImage 以正确保存/加载气泡状态
 */
function selectImage(index: number) {
  translateInit.switchImage(index)
}
</script>

<template>
  <div class="translate-page" :class="{ 'edit-mode-active': isEditMode }">
    <!-- 页面头部 -->
    <header class="app-header">
      <div class="header-content">
        <div class="logo-container">
          <router-link to="/" title="返回书架">
            <img :src="'/pic/logo.png'" alt="Saber-Translator Logo" class="app-logo">
            <span class="app-name">Saber-Translator</span>
          </router-link>
        </div>
        <div class="header-links">
          <router-link to="/" class="back-to-shelf" title="返回书架">📚</router-link>
          <button 
            v-if="isBookshelfMode"
            class="save-header-btn" 
            title="保存进度"
            @click="saveCurrentSession"
          >
            💾
          </button>
          <button 
            id="openSettingsBtn"
            class="settings-header-btn" 
            title="打开设置"
            @click="openSettings()"
          >
            <span class="icon">⚙️</span>
            <span>设置</span>
          </button>
          <a href="http://www.mashirosaber.top" target="_blank" class="tutorial-link">使用教程</a>
          <a href="javascript:void(0)" class="donate-link" @click="openSponsor">
            <span>❤️ 请作者喝奶茶</span>
          </a>
          <a href="https://github.com/MashiroSaber03" target="_blank" class="github-link">
            <img :src="'/pic/github.jpg'" alt="GitHub" class="github-icon">
          </a>
          <button 
            class="theme-toggle" 
            title="功能开发中"
            @click="showFeatureNotice"
          >
            <span class="theme-icon">☀️</span>
          </button>
        </div>
      </div>
    </header>

    <div class="container">
      <!-- 左侧设置侧边栏组件 -->
      <SettingsSidebar
        @translate-current="translateCurrentImage"
        @translate-all="translateAllImages"
        @hq-translate="startHqTranslation"
        @proofread="startProofreading"
        @remove-text="removeTextOnly"
        @remove-all-text="removeAllText"
        @retry-failed="handleRetryFailed"
        @delete-current="deleteCurrentImage"
        @clear-all="clearAllImages"
        @clean-temp="handleCleanTempFiles"
        @open-plugins="openPlugins"
        @open-settings="openSettings"
        @previous="goToPrevious"
        @next="goToNext"
        @apply-to-all="handleApplyToAll"
        @text-style-changed="handleTextStyleChanged"
        @auto-font-size-changed="handleAutoFontSizeChanged"
      />

      <!-- 主内容区 -->
      <main id="image-display-area">
        <!-- 上传区域 -->
        <section id="upload-section" class="card upload-card">
          <!-- 图片上传组件 -->
          <div class="upload-actions">
            <ImageUpload
              ref="imageUploadRef"
              @upload-complete="handleUploadComplete"
            />

          </div>
          
          <!-- 缩略图列表已移至右侧侧边栏 -->
          
          <!-- 会话加载进度条 -->
          <ProgressBar
            v-if="sessionStore.loadingProgress.total > 0"
            :visible="true"
            :percentage="(sessionStore.loadingProgress.current / sessionStore.loadingProgress.total * 100)"
            :label="sessionStore.loadingProgress.message"
          />
          
          <!-- 翻译进度组件 -->
          <TranslationProgress
            :progress="translation.progress.value"
          />
          
          <!-- 书架模式提示 -->
          <div v-if="isBatchTranslating && isBookshelfMode" class="bookshelf-mode-hint">
            <span style="color: #888; font-size: 0.85em;">
              （书架模式下退出前请点击顶部保存按钮）
            </span>
          </div>
        </section>

        <!-- 结果显示区域 -->
        <ImageResultDisplay
          ref="imageResultRef"
          :is-edit-mode="isEditMode"
          @toggle-edit-mode="toggleEditMode"
          @retry-failed="handleRetryFailed"
        />
      </main>

      <!-- 右侧缩略图侧边栏 -->
      <ThumbnailSidebar 
        v-if="hasImages && !isEditMode"
        @select="selectImage"
      />
    </div>
    
    <!-- 编辑工作区（编辑模式时显示，放在 container 外面实现全屏覆盖） -->
    <EditWorkspace
      v-if="currentImage && isEditMode"
      :is-edit-mode-active="isEditMode"
      @exit="toggleEditMode"
    />


    
    <!-- 首次使用引导 -->
    <FirstTimeGuide @open-settings="openSettings" />
    
    <!-- 设置模态框 -->
    <SettingsModal 
      v-model="showSettingsModal"
      :initial-tab="settingsInitialTab"
      @save="handleSettingsSave"
    />
    
    <!-- 赞助模态框 -->
    <SponsorModal 
      v-if="showSponsorModal" 
      @close="showSponsorModal = false" 
    />
    
    <!-- 网页导入免责声明弹窗 -->
    <WebImportDisclaimer />
    
    <!-- 网页导入模态框 -->
    <WebImportModal />
  </div>
</template>

<style scoped>
/* 翻译页面样式 - 匹配原版样式 */

/* 页面容器 */
.translate-page {
  min-height: 100vh;
  background-color: #f4f7f9;
}

/* 主容器 - 匹配原版 .container 样式 */
.container {
  display: flex;
  max-width: 1400px;
  margin: 20px auto;
  padding-left: 0;
  padding-right: 0;
  margin-top: 10px;
}

/* 主内容区 - 匹配原版 #image-display-area 样式 */
#image-display-area {
  flex-grow: 2.4;
  padding: 20px;
  margin-left: 340px;
  margin-right: 240px;
  max-width: none;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 上传区域卡片 - 匹配原版 #upload-section 样式 */
.upload-card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  text-align: center;
  flex: 0 0 auto;
  min-height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-bottom: 15px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.upload-card:hover {
  box-shadow: 0 8px 16px rgba(0,0,0,0.12);
}

/* 上传操作按钮组 */
.upload-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

/* 拖拽区域高亮 */
#drop-area.drag-over {
  border-color: var(--primary-color, #4a90d9);
  background-color: var(--hover-bg, rgba(74, 144, 217, 0.1));
}

/* 缩略图状态样式 */
.thumbnail-item {
  position: relative;
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 4px;
  overflow: hidden;
  transition: border-color 0.2s;
}

.thumbnail-item.active {
  border-color: var(--primary-color, #4a90d9);
}

.thumbnail-item.failed {
  border-color: var(--error-color, #e74c3c);
}

.thumbnail-item.processing {
  border-color: var(--warning-color, #f39c12);
}

.thumbnail-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.status-icon {
  position: absolute;
  top: 2px;
  right: 2px;
  font-size: 12px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 50%;
  padding: 2px;
}

.status-icon.failed {
  color: var(--error-color, #e74c3c);
}

.status-icon.processing {
  animation: pulse 1s infinite;
}

.status-icon.completed {
  color: var(--success-color, #27ae60);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 编辑模式占位符 */
.edit-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  background: var(--card-bg, #fff);
  border-radius: 8px;
}

/* 进度条样式 */
.progress-bar {
  width: 100%;
  height: 8px;
  background: var(--border-color, #e0e0e0);
  border-radius: 4px;
  overflow: hidden;
  margin: 8px 0;
}

.progress {
  height: 100%;
  background: var(--primary-color, #4a90d9);
  transition: width 0.3s ease;
}

/* 设置按钮高亮引导动画 */
@keyframes settingsBtnPulse {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(74, 144, 217, 0.4);
  }
  50% {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(74, 144, 217, 0.6);
  }
}

:deep(.settings-header-btn.highlight) {
  animation: settingsBtnPulse 0.5s ease-in-out 3;
  box-shadow: 0 0 10px var(--primary-color, #4a90d9);
}

/* 书籍/章节信息样式 */
.book-chapter-info {
  display: inline-flex;
  align-items: center;
  margin-left: 8px;
  font-size: 0.9em;
  color: var(--text-secondary, #666);
  max-width: 400px;
  overflow: hidden;
}

.book-chapter-info .separator {
  margin: 0 6px;
  color: var(--text-muted, #999);
}

.book-chapter-info .book-title,
.book-chapter-info .chapter-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
}

.book-chapter-info .book-title {
  color: var(--text-primary, #333);
  font-weight: 500;
}

.book-chapter-info .chapter-title {
  color: var(--primary-color, #4a90d9);
}

/* 响应式：小屏幕隐藏书籍/章节信息 */
@media (max-width: 768px) {
  .book-chapter-info {
    display: none;
  }
}

/* 开源声明样式 - 匹配原版 .open-source-notice 样式 */
.open-source-notice {
  font-weight: bold;
  color: #e53e3e;
  padding: 5px 12px;
  background-color: rgba(0,0,0,0.05);
  border-radius: 20px;
  font-size: 0.9em;
  white-space: nowrap;
}

/* 响应式：小屏幕隐藏开源声明 */
@media (max-width: 900px) {
  .open-source-notice {
    display: none;
  }
}

/* 头部样式 - 匹配原版 .app-header 样式 */
.app-header {
  background: transparent;
  color: #2c3e50;
  padding: 10px 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  width: auto;
  margin: 0 auto;
  max-width: calc(100% - 700px);
  z-index: 100;
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.logo-container {
  display: flex;
  align-items: center;
}

.logo-container a {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: #2c3e50;
}

.app-logo {
  height: 40px;
  width: auto;
  margin-right: 15px;
  border-radius: 8px;
}

.app-name {
  font-size: 1.5em;
  font-weight: bold;
  letter-spacing: 0.5px;
}

.header-links {
  display: flex;
  align-items: center;
  gap: 15px;
}

/* 教程链接和GitHub链接 */
.tutorial-link, .github-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: rgba(0,0,0,0.05);
  border-radius: 20px;
  color: #2c3e50;
  text-decoration: none;
  transition: all 0.3s ease;
}

.tutorial-link:hover, .github-link:hover {
  background-color: rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.github-icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

/* 赞助按钮样式 */
.donate-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: rgba(255, 105, 180, 0.15);
  border-radius: 20px;
  color: #e91e63;
  text-decoration: none;
  transition: all 0.3s ease;
}

.donate-link:hover {
  background-color: rgba(255, 105, 180, 0.25);
  transform: translateY(-2px);
}

/* 返回书架按钮样式 */
.back-to-shelf {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: white;
  text-decoration: none;
  font-size: 0.9em;
  font-weight: 500;
  transition: all 0.3s ease;
}

.back-to-shelf:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* 保存按钮样式（顶部） */
.save-header-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
  border: none;
  border-radius: 20px;
  color: white;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.save-header-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
}

/* 设置按钮样式 */
.settings-header-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: rgba(0,0,0,0.05);
  border: none;
  border-radius: 20px;
  color: #2c3e50;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9em;
}

.settings-header-btn:hover {
  background-color: rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  background-color: rgba(0,0,0,0.05);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.theme-toggle:hover {
  background-color: rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.theme-icon {
  font-size: 1.1em;
}

/* 书架模式提示 */
.bookshelf-mode-hint {
  margin-top: 10px;
  text-align: center;
}

/* 编辑工作区 - 不添加任何额外样式，使用全局 edit-mode.css 中的样式 */
/* .edit-workspace 样式由全局 edit-mode.css 控制，确保全屏覆盖 */

/* ============ 编辑模式激活时隐藏其他元素 ============ */

/* 编辑模式下隐藏所有非编辑内容 */
.translate-page.edit-mode-active .app-header,
.translate-page.edit-mode-active .container {
  display: none !important;
}

/* 编辑模式下 body 禁止滚动 */
.translate-page.edit-mode-active {
  overflow: hidden !important;
}
</style>
