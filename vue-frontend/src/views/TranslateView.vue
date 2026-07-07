<script setup lang="ts">
import AppShell from '@/components/ui/AppShell.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useSessionStore } from '@/stores/sessionStore'
import { showToast } from '@/utils/toast'
import ImageUpload from '@/components/translate/ImageUpload.vue'
import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'
import ImageResultDisplay from '@/components/translate/ImageResultDisplay.vue'
import FirstTimeGuide from '@/components/translate/FirstTimeGuide.vue'
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
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'
import ProductThemeToggle from '@/components/product/ProductThemeToggle.vue'
import { useTextStyleSync } from '@/composables/useTextStyleSync'
import { useTranslateViewActions } from './useTranslateViewActions'

import WebImportModal from '@/components/translate/WebImportModal.vue'
import WebImportDisclaimer from '@/components/translate/WebImportDisclaimer.vue'

const route = useRoute()

const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const sessionStore = useSessionStore()
const bubbleStore = useBubbleStore()

const {
  validateBeforeTranslation,
  initValidation,
  isSettingsButtonHighlighted,
} = useValidation()

const translation = useTranslation()

const {
  handleTextStyleChanged,
  handleAutoFontSizeChanged,
  handleAutoTextColorChanged,
  handleApplyToAll,
} = useTextStyleSync()

const translateInit = useTranslateInit()

const showSettingsModal = ref(false)
const showBookGlossaryModal = ref(false)
const showBookNonTranslateModal = ref(false)

const showSponsorModal = ref(false)

const isEditMode = ref(false)

const currentImage = computed(() => imageStore.currentImage)
const hasImages = computed(() => imageStore.hasImages)
const isBatchTranslating = computed(() => imageStore.isBatchTranslationInProgress)
const hasFailedImages = computed(() => imageStore.failedImageCount > 0)
const showThumbnailSidebar = computed(() => hasImages.value && !isEditMode.value)
const isBookshelfMode = computed(() => {
  return !!route.query.book && !!route.query.chapter
})
const currentBookId = computed(() => route.query.book as string | undefined)
const currentChapterId = computed(() => route.query.chapter as string | undefined)
const currentBookTitle = computed(() => translateInit.currentBookTitle.value)
const currentChapterTitle = computed(() => translateInit.currentChapterTitle.value)
const sessionLoadingPercent = computed(() => {
  const { current, total } = sessionStore.loadingProgress
  if (total <= 0) return 0
  return Math.min(100, Math.max(0, Math.round((current / total) * 100)))
})
const pageTitle = computed(() => {
  if (isBookshelfMode.value && currentChapterTitle.value && currentBookTitle.value) {
    return `${currentChapterTitle.value} - ${currentBookTitle.value}`
  }
  return 'Saber-Translator'
})

onMounted(async () => {
  window.addEventListener('keydown', handleKeydown)

  imageStore.clearImages()
  bubbleStore.clearBubbles()

  await translateInit.initializeApp()

  initValidation()
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})

watch(
  () => [route.query.book, route.query.chapter],
  async ([newBook, newChapter], [previousBook, previousChapter]) => {
    if (newBook && newChapter) {
      imageStore.clearImages()
      bubbleStore.clearBubbles()

      await loadChapterSession()
    } else if (previousBook && previousChapter && !newBook && !newChapter) {
      imageStore.clearImages()
      bubbleStore.clearBubbles()
      await translateInit.initializeBookChapterContext()
    }
  }
)

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

function openSettings() {
  showSettingsModal.value = true
}

function handleSettingsSave(payload?: { textDefaultsChanged?: boolean }) {
  if (payload?.textDefaultsChanged) {
    showToast('已修改默认值，将在下次启动时生效', 'success')
    return
  }
  showToast('设置已保存', 'success')
}

function openSponsor() {
  showSponsorModal.value = true
}
</script>

<template>
  <AppShell class="translate-page" :class="{ 'edit-mode-active': isEditMode }">
    <ProductPageHeader
      v-show="!isEditMode"
      logo-title="返回书架"
      nav-label="翻译页面导航"
      actions-label="翻译页面操作"
    >
      <template #nav>
        <ProductHeaderAction
          as="router-link"
          to="/"
          class="translate-header__back-link"
          title="返回书架"
          aria-label="返回书架"
          icon-name="book-open"
          icon-only
        />
        <ProductHeaderAction
          as="a"
          href="http://www.mashirosaber.top"
          target="_blank"
          rel="noopener noreferrer"
          class="translate-header__link translate-header__link--tutorial"
          label="使用教程"
        />
        <ProductHeaderAction
          as="a"
          href="https://github.com/MashiroSaber03"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub 主页"
          class="translate-header__link translate-header__link--github"
          icon-name="github"
          icon-only
        />
      </template>

      <template #actions>
        <ProductHeaderAction
          v-if="isBookshelfMode"
          variant="solid"
          class="translate-header__save-button"
          title="保存进度"
          aria-label="保存进度"
          icon-name="save"
          icon-only
          @click="saveCurrentSession"
        />
        <ProductHeaderAction
          class="translate-header__settings-button"
          :class="{ 'translate-header__settings-button--highlighted': isSettingsButtonHighlighted }"
          title="打开设置"
          icon-name="settings"
          label="设置"
          @click="openSettings()"
        />
        <ProductHeaderAction
          class="translate-header__link translate-header__link--donate"
          aria-label="请作者喝奶茶"
          label="请作者喝奶茶"
          @click="openSponsor"
        />
        <ProductThemeToggle
          class="translate-header__theme-toggle"
          icon-size="lg"
        />
      </template>
    </ProductPageHeader>

    <SidebarLayout
      v-show="!isEditMode"
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

      <main class="translate-workspace">
        <section class="translate-upload-card">
          <div class="translate-upload-card__actions">
            <ImageUpload
              @upload-complete="handleUploadComplete"
            />
          </div>

          <UiProgressBar
            v-if="sessionStore.loadingProgress.total > 0"
            :label="sessionStore.loadingProgress.message"
            :value="sessionLoadingPercent"
          >
            <span class="translate-upload-card__progress-label">
              {{ sessionStore.loadingProgress.message }}
            </span>
          </UiProgressBar>

          <TranslationProgress
            :progress="translation.progress.value"
          />

          <div v-if="isBatchTranslating && isBookshelfMode" class="translate-bookshelf-mode-hint">
            <span class="translate-bookshelf-mode-hint__text">
              （书架模式下退出前请点击顶部保存按钮）
            </span>
          </div>
        </section>

        <ImageResultDisplay
          :is-edit-mode="isEditMode"
          @toggle-edit-mode="toggleEditMode"
          @retry-failed="handleRetryFailed"
        />
      </main>

      <template #right>
        <ThumbnailSidebar
          v-show="showThumbnailSidebar"
          :is-visible="showThumbnailSidebar"
          @select="selectImage"
        />
      </template>
    </SidebarLayout>

    <EditWorkspace
      v-if="currentImage && isEditMode"
      :is-edit-mode-active="isEditMode"
      @exit="toggleEditMode"
    />


    <FirstTimeGuide @open-settings="openSettings" />

    <SettingsModal
      v-model="showSettingsModal"
      @save="handleSettingsSave"
    />

    <BookGlossaryModal v-model="showBookGlossaryModal" />
    <BookNonTranslateModal v-model="showBookNonTranslateModal" />

    <SponsorModal
      v-if="showSponsorModal"
      @close="showSponsorModal = false"
    />

    <WebImportDisclaimer />

    <WebImportModal />
  </AppShell>
</template>

<style scoped>
.translate-page {
  /* owner tokens: translate-view */
  --translate-view-page-background: var(--color-surface-page);
  --translate-view-settings-pulse-shadow: var(--color-focus-brand-soft);
  --translate-view-settings-pulse-shadow-strong: var(--color-focus-brand-subtle);

  background-color: var(--translate-view-page-background);
}

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

.translate-workspace {
  flex-grow: 2.4;
  padding: 20px;
  max-width: none;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.translate-upload-card {
  background-color: var(--color-surface-card);
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

.translate-upload-card__actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

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

.translate-header__settings-button--highlighted {
  animation: settingsBtnPulse 0.5s ease-in-out 3;
  box-shadow: 0 0 10px var(--color-action-primary);
}

.translate-bookshelf-mode-hint {
  margin-top: 10px;
  text-align: center;
}

.translate-bookshelf-mode-hint__text {
  color: var(--color-text-subtle);
  font-size: 0.85em;
}

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
    padding: 0 0 23px;
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

  .translate-upload-card__actions {
    justify-content: center;
    width: 100%;
  }
}
</style>
