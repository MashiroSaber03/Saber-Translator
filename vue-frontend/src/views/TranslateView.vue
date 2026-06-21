<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import AppShell from '@/components/ui/AppShell.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'
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
import { useSettingsStore } from '@/stores/settings'
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
import BookGlossaryModal from '@/components/translate/BookGlossaryModal.vue'
import BookNonTranslateModal from '@/components/translate/BookNonTranslateModal.vue'
import EditWorkspace from '@/components/edit/EditWorkspace.vue'
import ProgressBar from '@/components/common/ProgressBar.vue'
import AppHeader from '@/components/common/AppHeader.vue'
import { useTextStyleSync } from '@/composables/useTextStyleSync'
import { useTranslateViewActions } from './useTranslateViewActions'

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
  handleAutoTextColorChanged,
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
const showBookGlossaryModal = ref(false)
const showBookNonTranslateModal = ref(false)

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
  // 进入翻译路由时重置工作区状态，保证书架模式和快速翻译模式互不串会话。
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
    if (newBook && newChapter) {
      // 进入书架章节前先同步清空当前图片和气泡状态。
      imageStore.clearImages()
      bubbleStore.clearBubbles()
      
      await loadChapterSession()
    } else if (oldBook && oldChapter && !newBook && !newChapter) {
      // 从书架模式切回快速翻译时重置工作区。
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

const {
  goToNext,
  goToPrevious,
  handleKeydown,
  handleRetryFailed,
  handleRunWorkflow,
  handleUploadComplete,
  loadChapterSession,
  saveCurrentSession,
  selectImage,
  toggleEditMode,
} = useTranslateViewActions({
  imageStore,
  settingsStore,
  sessionStore,
  translation,
  translateInit,
  validateBeforeTranslation,
  currentImage,
  hasImages,
  hasFailedImages,
  currentBookId,
  currentChapterId,
  isEditMode,
})

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
</script>

<template>
  <AppShell class="translate-page" :class="{ 'edit-mode-active': isEditMode }">
    <!-- 页面头部 -->
    <AppHeader logo-title="返回书架">
      <template #header-links>
        <router-link to="/" class="translate-header__back-link" title="返回书架">📚</router-link>
        <UiButton
          variant="toolbar" 
          v-if="isBookshelfMode"
          class="translate-header__save-button" 
          title="保存进度"
          @click="saveCurrentSession"
        >
          💾
        </UiButton>
        <UiButton
          variant="toolbar" 
          id="openSettingsBtn"
          class="translate-header__settings-button" 
          title="打开设置"
          @click="openSettings()"
        >
          <span class="icon">⚙️</span>
          <span>设置</span>
        </UiButton>
        <a href="http://www.mashirosaber.top" target="_blank" class="translate-header__link translate-header__link--tutorial">使用教程</a>
        <a href="javascript:void(0)" class="translate-header__link translate-header__link--donate" @click="openSponsor">
          <span>❤️ 请作者喝奶茶</span>
        </a>
        <a href="https://github.com/MashiroSaber03" target="_blank" class="translate-header__link translate-header__link--github">
          <img :src="'/pic/github.jpg'" alt="GitHub" class="translate-header__github-icon">
        </a>
        <UiButton
          variant="toolbar" 
          class="translate-header__theme-toggle" 
          title="功能开发中"
          @click="showFeatureNotice"
        >
          <span class="translate-header__theme-icon">☀️</span>
        </UiButton>
      </template>
    </AppHeader>

    <SidebarLayout class="translate-shell">
      <!-- 左侧设置侧边栏组件 -->
      <SettingsSidebar
        @run-workflow="handleRunWorkflow"
        @previous="goToPrevious"
        @next="goToNext"
        @apply-to-all="handleApplyToAll"
        @text-style-changed="handleTextStyleChanged"
        @auto-font-size-changed="handleAutoFontSizeChanged"
        @auto-text-color-changed="handleAutoTextColorChanged"
        @open-glossary="showBookGlossaryModal = true"
        @open-non-translate="showBookNonTranslateModal = true"
      />

      <!-- 主内容区 -->
      <main class="translate-workspace">
        <!-- 上传区域 -->
        <section class="translate-upload-card">
          <!-- 图片上传组件 -->
          <div class="translate-upload-actions">
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
          <div v-if="isBatchTranslating && isBookshelfMode" class="translate-bookshelf-mode-hint">
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
    </SidebarLayout>
    
    <!-- 编辑工作区（编辑模式时显示，放在页面布局外面实现全屏覆盖） -->
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

    <BookGlossaryModal v-model="showBookGlossaryModal" />
    <BookNonTranslateModal v-model="showBookNonTranslateModal" />
    
    <!-- 赞助模态框 -->
    <SponsorModal 
      v-if="showSponsorModal" 
      @close="showSponsorModal = false" 
    />
    
    <!-- 网页导入免责声明弹窗 -->
    <WebImportDisclaimer />
    
    <!-- 网页导入模态框 -->
    <WebImportModal />
  </AppShell>
</template>

<style scoped>/* 翻译页面布局 */

/* 页面容器 */
.translate-page {
  --translate-sidebar-left-gutter: 340px;
  --translate-sidebar-right-gutter: 240px;

  min-height: 100vh;
  background-color: var(--translate-view-surface-base);
}

/* 页面主容器 */
.translate-shell {
  display: flex;
  width: calc(100% - 40px);
  max-width: 1400px;
  margin: 20px auto;
  padding-left: 0;
  padding-right: 0;
  margin-top: 10px;
}

/* 主内容区 */
.translate-workspace {
  flex-grow: 2.4;
  padding: 20px;
  margin-left: var(--translate-sidebar-left-gutter);
  margin-right: var(--translate-sidebar-right-gutter);
  max-width: none;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 上传区域卡片 */
.translate-upload-card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px var(--shadow-soft);
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

.translate-upload-card:hover {
  box-shadow: 0 8px 16px var(--shadow-medium);
}

/* 上传操作按钮组 */
.translate-upload-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

/* 设置按钮高亮引导动画 */
@keyframes settingsBtnPulse {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 var(--translate-view-shadow-default);
  }

  50% {
    transform: scale(1.05);
    box-shadow: 0 0 15px var(--translate-view-shadow-raised);
  }
}

.translate-header__settings-button.highlight {
  animation: settingsBtnPulse 0.5s ease-in-out 3;
  box-shadow: 0 0 10px var(--color-action-primary, var(--translate-view-shadow-floating));
}

/* 书籍/章节信息样式 */
.translate-book-chapter-info {
  display: inline-flex;
  align-items: center;
  margin-left: 8px;
  font-size: 0.9em;
  color: var(--color-text-supporting, var(--color-text-secondary));
  max-width: 400px;
  overflow: hidden;
}

.translate-book-chapter-info .translate-book-chapter-info__separator {
  margin: 0 6px;
  color: var(--color-text-disabled, var(--color-text-muted));
}

.translate-book-chapter-info__book-title,
.translate-book-chapter-info__chapter-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
}

.translate-book-chapter-info__book-title {
  color: var(--color-text-default, var(--color-text-default));
  font-weight: 500;
}

.translate-book-chapter-info__chapter-title {
  color: var(--color-action-primary, var(--translate-view-text-primary));
}

/* 响应式：小屏幕隐藏书籍/章节信息 */
@media (--breakpoint-md-down) {
  .translate-book-chapter-info {
    display: none;
  }
}

/* 开源声明 */
.translate-open-source-notice {
  font-weight: bold;
  color: var(--translate-view-text-secondary);
  padding: 5px 12px;
  background-color: var(--translate-view-surface-raised);
  border-radius: 20px;
  font-size: 0.9em;
  white-space: nowrap;
}

/* 响应式：小屏幕隐藏开源声明 */
@media (--breakpoint-lg-down) {
  .translate-open-source-notice {
    display: none;
  }
}

/* 翻译页 header slot 内容 */
.translate-header__link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: var(--translate-view-surface-raised);
  border-radius: 20px;
  color: var(--color-text-heading);
  text-decoration: none;
  transition: all 0.3s ease;
}

.translate-header__link:hover {
  background-color: var(--translate-view-surface-muted);
  transform: translateY(-2px);
}

.translate-header__github-icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

.translate-header__link--donate {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: var(--translate-view-surface-subtle);
  border-radius: 20px;
  color: var(--translate-view-text-muted);
  text-decoration: none;
  transition: all 0.3s ease;
}

.translate-header__link--donate:hover {
  background-color: var(--translate-view-surface-hover);
  transform: translateY(-2px);
}

/* 返回书架按钮样式 */
.translate-header__back-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
  border-radius: 20px;
  color: white;
  text-decoration: none;
  font-size: 0.9em;
  font-weight: 500;
  transition: all 0.3s ease;
}

.translate-header__back-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow-brand-soft);
}

/* 保存按钮样式（顶部） */
.translate-header__save-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  background: linear-gradient(135deg, var(--color-surface-success-gradient-start) 0%, var(--color-surface-success-gradient-end) 100%);
  border: none;
  border-radius: 20px;
  color: white;
  font-size: 1em;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.translate-header__save-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow-success-soft);
}

/* 设置按钮样式 */
.translate-header__settings-button {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: var(--translate-view-surface-raised);
  border: none;
  border-radius: 20px;
  color: var(--color-text-heading);
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9em;
}

.translate-header__settings-button:hover {
  background-color: var(--translate-view-surface-muted);
  transform: translateY(-2px);
}

.translate-header__theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  background-color: var(--translate-view-surface-raised);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.translate-header__theme-toggle:hover {
  background-color: var(--translate-view-surface-muted);
  transform: translateY(-2px);
}

.translate-header__theme-icon {
  font-size: 1.1em;
}

/* 书架模式提示 */
.translate-bookshelf-mode-hint {
  margin-top: 10px;
  text-align: center;
}

.translate-bookshelf-mode-hint .hint-text {
  color: var(--color-text-subtle);
  font-size: 0.85em;
}

/* 编辑工作区 - 不添加任何额外样式，使用全局 edit-mode.css 中的样式 */

/* .edit-workspace 样式由全局 edit-mode.css 控制，确保全屏覆盖 */

/* ============ 编辑模式激活时隐藏其他元素 ============ */

/* 编辑模式下隐藏所有非编辑内容 */
.translate-page.edit-mode-active .app-header,
.translate-page.edit-mode-active .translate-shell {
  display: none;
}

/* 编辑模式下 body 禁止滚动 */
.translate-page.edit-mode-active {
  overflow: hidden;
}

@media (--breakpoint-md-down) {
  .translate-shell {
    flex-direction: column;
    gap: 16px;
    width: calc(100% - 40px);
    max-width: 100%;
    margin: 8px auto 0;
    padding: 0 0 21px;
  }

  .translate-workspace {
    order: 1;
    width: 100%;
    margin-right: 0;
    margin-left: 0;
    padding: 0;
    gap: 16px;
  }

  .translate-upload-card {
    min-height: 160px;
    padding: 18px;
    margin-bottom: 0;
  }

  .translate-upload-actions {
    justify-content: center;
    width: 100%;
  }

  .translate-page .app-header__links {
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
  }

  .translate-header__back-link,
  .translate-header__save-button,
  .translate-header__settings-button,
  .translate-header__link,
  .translate-header__theme-toggle {
    min-height: 38px;
    flex-shrink: 0;
  }

  .translate-header__link--donate span {
    white-space: nowrap;
  }
}
</style>
