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
import AppHeader from '@/components/common/AppHeader.vue'
import { useTextStyleSync } from '@/composables/useTextStyleSync'
import type { WorkflowRunRequest } from '@/types/workflow'

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

// 文字样式同步与应用
const {
  handleTextStyleChanged,
  handleAutoFontSizeChanged,
  handleApplyToAll,
} = useTextStyleSync()

// 导出导入功能已移至具体按钮事件处理函数中

// 翻译页面初始化
const translateInit = useTranslateInit()

// ============================================================
// 状态定义
// ============================================================

/** 是否显示设置模态框 */
const showSettingsModal = ref(false)

/** 是否显示赞助模态框 */
const showSponsorModal = ref(false)

/** 是否处于编辑模式 */
const isEditMode = ref(false)

// ============================================================
// 计算属性
// ============================================================

/** 当前图片 */
const currentImage = computed(() => imageStore.currentImage)

/** 是否有图片 */
const hasImages = computed(() => imageStore.hasImages)

/** 批量翻译是否进行中 */
const isBatchTranslating = computed(() => imageStore.isBatchTranslationInProgress)

/** 是否有翻译失败的图片 */
const hasFailedImages = computed(() => imageStore.failedImageCount > 0)

/** 是否显示缩略图侧边栏（有图片且不在编辑模式） */
const showThumbnailSidebar = computed(() => hasImages.value && !isEditMode.value)

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
 * 翻译指定范围的图片
 * @param startPage 起始页（1开始）
 * @param endPage 结束页（1开始）
 */
async function translateImageRange(startPage: number, endPage: number) {
  if (!hasImages.value) return
  
  // 验证翻译配置
  if (!validateBeforeTranslation('normal')) {
    return
  }
  
  await translation.translateImageRange({ startPage, endPage })
}

/**
 * 高质量翻译指定范围
 * @param startPage 起始页（1开始）
 * @param endPage 结束页（1开始）
 */
async function startHqTranslationRange(startPage: number, endPage: number) {
  if (!hasImages.value) return
  
  // 验证高质量翻译配置
  if (!validateBeforeTranslation('hq')) {
    return
  }
  
  await translation.executeHqTranslation({ startPage, endPage })
}

/**
 * AI校对指定范围
 * @param startPage 起始页（1开始）
 * @param endPage 结束页（1开始）
 */
async function startProofreadingRange(startPage: number, endPage: number) {
  if (!hasImages.value) return
  
  // 验证 AI 校对配置
  if (!validateBeforeTranslation('proofread')) {
    return
  }
  
  await translation.executeProofreading({ startPage, endPage })
}

/**
 * 消除指定范围图片的文字
 * @param startPage 起始页（1开始）
 * @param endPage 结束页（1开始）
 */
async function removeTextRange(startPage: number, endPage: number) {
  if (!hasImages.value) return
  await translation.removeTextRange({ startPage, endPage })
}

/**
 * 统一处理侧边栏工作流启动
 */
async function handleRunWorkflow(payload: WorkflowRunRequest) {
  const range = payload.range

  switch (payload.mode) {
    case 'translate-current':
      await translateCurrentImage()
      return
    case 'translate-batch':
      if (range) {
        await translateImageRange(range.startPage, range.endPage)
      } else {
        await translateAllImages()
      }
      return
    case 'hq-batch':
      if (range) {
        await startHqTranslationRange(range.startPage, range.endPage)
      } else {
        await startHqTranslation()
      }
      return
    case 'proofread-batch':
      if (range) {
        await startProofreadingRange(range.startPage, range.endPage)
      } else {
        await startProofreading()
      }
      return
    case 'remove-current':
      await removeTextOnly()
      return
    case 'remove-batch':
      if (range) {
        await removeTextRange(range.startPage, range.endPage)
      } else {
        await removeAllText()
      }
      return
    case 'retry-failed':
      await handleRetryFailed()
      return
    case 'delete-current':
      deleteCurrentImage()
      return
    case 'clear-all':
      clearAllImages()
      return
    default:
      return
  }
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
 */
function openSettings() {
  showSettingsModal.value = true
}

/**
 * 处理设置保存
 */
function handleSettingsSave(payload?: { textDefaultsChanged?: boolean }) {
  if (payload?.textDefaultsChanged) {
    showToast('已修改默认值，将在下次启动时生效', 'success')
    return
  }
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
    <AppHeader logo-title="返回书架">
      <template #header-links>
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
      </template>
    </AppHeader>

    <div class="container">
      <!-- 左侧设置侧边栏组件 -->
      <SettingsSidebar
        @run-workflow="handleRunWorkflow"
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
            <span class="hint-text">
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
        v-show="showThumbnailSidebar"
        :is-visible="showThumbnailSidebar"
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
  box-shadow: 0 4px 12px rgb(0,0,0,0.08);
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
  box-shadow: 0 8px 16px rgb(0,0,0,0.12);
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
  border-color: var(--color-primary, #4a90d9);
  background-color: var(--hover-bg, rgb(74, 144, 217, 0.1));
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
  border-color: var(--color-primary, #4a90d9);
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
  background: rgb(255, 255, 255, 0.9);
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
  background: var(--color-primary, #4a90d9);
  transition: width 0.3s ease;
}

/* 设置按钮高亮引导动画 */
@keyframes settingsBtnPulse {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgb(74, 144, 217, 0.4);
  }

  50% {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgb(74, 144, 217, 0.6);
  }
}

:deep(.settings-header-btn.highlight) {
  animation: settingsBtnPulse 0.5s ease-in-out 3;
  box-shadow: 0 0 10px var(--color-primary, #4a90d9);
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
  color: var(--color-primary, #4a90d9);
}

/* 响应式：小屏幕隐藏书籍/章节信息 */
@media (width <= 768px) {
  .book-chapter-info {
    display: none;
  }
}

/* 开源声明样式 - 匹配原版 .open-source-notice 样式 */
.open-source-notice {
  font-weight: bold;
  color: #e53e3e;
  padding: 5px 12px;
  background-color: rgb(0,0,0,0.05);
  border-radius: 20px;
  font-size: 0.9em;
  white-space: nowrap;
}

/* 响应式：小屏幕隐藏开源声明 */
@media (width <= 900px) {
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
  z-index: var(--z-dropdown);
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  background: rgb(255, 255, 255, 0.9);
  border-radius: 12px;
  box-shadow: 0 2px 10px rgb(0,0,0,0.1);
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
  background-color: rgb(0,0,0,0.05);
  border-radius: 20px;
  color: #2c3e50;
  text-decoration: none;
  transition: all 0.3s ease;
}

.tutorial-link:hover, .github-link:hover {
  background-color: rgb(0,0,0,0.1);
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
  background-color: rgb(255, 105, 180, 0.15);
  border-radius: 20px;
  color: #e91e63;
  text-decoration: none;
  transition: all 0.3s ease;
}

.donate-link:hover {
  background-color: rgb(255, 105, 180, 0.25);
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
  box-shadow: 0 4px 12px rgb(102, 126, 234, 0.4);
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
  box-shadow: 0 4px 12px rgb(40, 167, 69, 0.4);
}

/* 设置按钮样式 */
.settings-header-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: rgb(0,0,0,0.05);
  border: none;
  border-radius: 20px;
  color: #2c3e50;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9em;
}

.settings-header-btn:hover {
  background-color: rgb(0,0,0,0.1);
  transform: translateY(-2px);
}

.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  background-color: rgb(0,0,0,0.05);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.theme-toggle:hover {
  background-color: rgb(0,0,0,0.1);
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

.bookshelf-mode-hint .hint-text {
  color: #888;
  font-size: 0.85em;
}

/* 编辑工作区 - 不添加任何额外样式，使用全局 edit-mode.css 中的样式 */

/* .edit-workspace 样式由全局 edit-mode.css 控制，确保全屏覆盖 */

/* ============ 编辑模式激活时隐藏其他元素 ============ */

/* 编辑模式下隐藏所有非编辑内容 */
.translate-page.edit-mode-active .app-header,
.translate-page.edit-mode-active .container {
  display: none;
}

/* 编辑模式下 body 禁止滚动 */
.translate-page.edit-mode-active {
  overflow: hidden;
}
</style>
