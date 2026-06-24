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
  window.addEventListener('keydown', handleKeydown)

  // 进入翻译路由时重置工作区状态，保证书架模式和快速翻译模式互不串会话。
  imageStore.clearImages()
  bubbleStore.clearBubbles()

  // 使用 useTranslateInit 进行完整初始化
  // 包括：设置初始化、字体列表、提示词、主题、书架模式会话加载
  await translateInit.initializeApp()

  // 初始化配置验证（延迟显示首次使用引导）
  initValidation()
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})

// 监听路由参数变化
watch(
  () => [route.query.book, route.query.chapter],
  async ([newBook, newChapter], [previousBook, previousChapter]) => {
    if (newBook && newChapter) {
      // 进入书架章节前先同步清空当前图片和气泡状态。
      imageStore.clearImages()
      bubbleStore.clearBubbles()

      await loadChapterSession()
    } else if (previousBook && previousChapter && !newBook && !newChapter) {
      // 从书架模式切回快速翻译时重置工作区。
      imageStore.clearImages()
      bubbleStore.clearBubbles()
      // 清空书籍/章节上下文
      await translateInit.initializeBookChapterContext()
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
        <router-link to="/" class="translate-header__back-link" title="返回书架" aria-label="返回书架">📚</router-link>
        <UiButton
          variant="toolbar"
          v-if="isBookshelfMode"
          class="translate-header__save-button"
          title="保存进度"
          aria-label="保存进度"
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
        <a href="http://www.mashirosaber.top" target="_blank" rel="noopener noreferrer" class="translate-header__link translate-header__link--tutorial">使用教程</a>
        <UiButton
          variant="toolbar"
          class="translate-header__link translate-header__link--donate"
          aria-label="请作者喝奶茶"
          @click="openSponsor"
        >
          <span>❤️ 请作者喝奶茶</span>
        </UiButton>
        <a href="https://github.com/MashiroSaber03" target="_blank" rel="noopener noreferrer" aria-label="GitHub 主页" class="translate-header__link translate-header__link--github">
          <img :src="'/pic/github.jpg'" alt="GitHub" class="translate-header__github-icon">
        </a>
        <UiButton
          variant="toolbar"
          class="translate-header__theme-toggle"
          title="功能开发中"
          aria-label="功能开发中"
          @click="showFeatureNotice"
        >
          <span class="translate-header__theme-icon">☀️</span>
        </UiButton>
      </template>
    </AppHeader>

    <SidebarLayout
      class="translate-shell"
      sidebars="fixed"
      left-width="300px"
      right-width="230px"
      left-inset="340px"
      right-inset="240px"
      left-offset="20px"
      right-offset="20px"
      left-top="70px"
      right-top="20px"
      left-height="calc(100dvh - 90px)"
      right-height="calc(100dvh - 40px)"
      main-class="translate-shell__main"
    >
      <!-- 左侧设置侧边栏组件 -->
      <template #left>
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
      </template>

      <!-- 主内容区 -->
      <main class="translate-workspace">
        <!-- 上传区域 -->
        <section class="translate-upload-card">
          <!-- 图片上传组件 -->
          <div class="translate-upload-actions">
            <ImageUpload
              @upload-complete="handleUploadComplete"
            />
          </div>

          <!-- 会话加载进度条 -->
          <ProgressBar
            v-if="sessionStore.loadingProgress.total > 0"
            :visible="true"
            :percentage="(sessionStore.loadingProgress.current / sessionStore.loadingProgress.total * 100)"
            :label="sessionStore.loadingProgress.message"
          />

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
          :is-edit-mode="isEditMode"
          @toggle-edit-mode="toggleEditMode"
          @retry-failed="handleRetryFailed"
        />
      </main>

      <!-- 右侧缩略图侧边栏 -->
      <template #right>
        <ThumbnailSidebar
          v-show="showThumbnailSidebar"
          :is-visible="showThumbnailSidebar"
          @select="selectImage"
        />
      </template>
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

<style scoped>
/* 翻译页面布局 */

/* 页面容器 */
.translate-page {
  /* owner tokens: translate-view */
  --translate-view-page-background: #f4f7f9;
  --translate-view-settings-pulse-shadow: rgba(74, 144, 217, .4);
  --translate-view-settings-pulse-shadow-strong: rgba(74, 144, 217, .6);
  --translate-view-header-control-background: rgba(0, 0, 0, .05);
  --translate-view-header-control-background-hover: rgba(0, 0, 0, .1);
  --translate-view-donate-background: rgba(255, 105, 180, .15);
  --translate-view-donate-background-hover: rgba(255, 105, 180, .25);
  --translate-view-donate-text: #e91e63;

  background-color: var(--translate-view-page-background);
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

.translate-shell__main {
  min-width: 0;
}

/* 主内容区 */
.translate-workspace {
  flex-grow: 2.4;
  padding: 20px;
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
    box-shadow: 0 0 0 0 var(--translate-view-settings-pulse-shadow);
  }

  50% {
    transform: scale(1.05);
    box-shadow: 0 0 15px var(--translate-view-settings-pulse-shadow-strong);
  }
}

.translate-header__settings-button.highlight {
  animation: settingsBtnPulse 0.5s ease-in-out 3;
  box-shadow: 0 0 10px var(--color-action-primary);
}

/* 翻译页 header slot 内容 */
.translate-header__link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  border: 0;
  background-color: var(--translate-view-header-control-background);
  border-radius: 20px;
  color: var(--color-text-heading);
  cursor: pointer;
  font: inherit;
  text-decoration: none;
  transition: all 0.3s ease;
}

.translate-header__link:hover {
  background-color: var(--translate-view-header-control-background-hover);
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
  background-color: var(--translate-view-donate-background);
  border-radius: 20px;
  color: var(--translate-view-donate-text);
  text-decoration: none;
  transition: all 0.3s ease;
}

.translate-header__link--donate:hover {
  background-color: var(--translate-view-donate-background-hover);
  transform: translateY(-2px);
}

/* 返回书架按钮样式 */
.translate-header__back-link {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 14px;
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  border-radius: 20px;
  color: white;
  text-decoration: none;
  font-size: 0.9em;
  font-weight: 500;
  transition: all 0.3s ease;
}

.translate-header__back-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow-action-brand);
}

/* 保存按钮样式（顶部） */
.translate-header__save-button {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  background: linear-gradient(135deg, var(--color-action-success) 0%, var(--color-action-success-strong) 100%);
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
  box-shadow: 0 4px 12px var(--shadow-action-success);
}

/* 设置按钮样式 */
.translate-header__settings-button {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
  background-color: var(--translate-view-header-control-background);
  border: none;
  border-radius: 20px;
  color: var(--color-text-heading);
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9em;
}

.translate-header__settings-button:hover {
  background-color: var(--translate-view-header-control-background-hover);
  transform: translateY(-2px);
}

.translate-header__theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  background-color: var(--translate-view-header-control-background);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.translate-header__theme-toggle:hover {
  background-color: var(--translate-view-header-control-background-hover);
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

/* EditWorkspace owns the fullscreen editor surface; the page only hides route chrome while it is active. */

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
