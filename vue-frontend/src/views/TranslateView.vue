<script setup lang="ts">
import AppShell from '@/components/ui/AppShell.vue'
import SidebarLayout from '@/components/ui/SidebarLayout.vue'
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import {
  onBeforeRouteLeave,
  onBeforeRouteUpdate,
  useRoute,
} from 'vue-router'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
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
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'
import ProductThemeToggle from '@/components/product/ProductThemeToggle.vue'
import { useTextStyleSync } from '@/composables/useTextStyleSync'
import { useTranslateViewActions } from './useTranslateViewActions'
import {
  flushPageDocument,
  discardPageDocument,
  isPageDocumentRegistered,
  queuePageDocumentSave,
} from '@/services/pageDocumentPersistence'

import WebImportModal from '@/components/translate/WebImportModal.vue'
import WebImportDisclaimer from '@/components/translate/WebImportDisclaimer.vue'
import QuickWorkspacePromoteModal from '@/components/translate/QuickWorkspacePromoteModal.vue'
import { resetQuickWorkspace } from '@/api/v2/content'
import { ApiClientError } from '@/api/client'
import { HISTORY_JOB_STATUSES } from '@/api/v2/jobs'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const route = useRoute()

const imageStore = useImageStore()
const settingsStore = useSettingsStore()
const bubbleStore = useBubbleStore()
const taskCenterStore = useTaskCenterStore()

const {
  validateBeforeTranslation,
  isSettingsButtonHighlighted,
} = useValidation()

const {
  handleTextStyleChanged,
  handleAutoFontSizeChanged,
  handleAutoTextColorChanged,
  handleApplyToAll,
} = useTextStyleSync()

const translateInit = useTranslateInit()
const translation = useTranslation({
  beforeCreateJob: translateInit.flushChapterWorkState,
})

const showSettingsModal = ref(false)
const showBookGlossaryModal = ref(false)
const showBookNonTranslateModal = ref(false)
const showQuickPromoteModal = ref(false)
const pendingContentImportJobIds = ref<Set<string>>(new Set())

const showSponsorModal = ref(false)

const isEditMode = ref(false)

const currentImage = computed(() => imageStore.currentImage)
const hasImages = computed(() => imageStore.hasImages)
const showThumbnailSidebar = computed(() => hasImages.value && !isEditMode.value)
const isBookshelfMode = computed(() => translateInit.isBookshelfMode.value)
const currentChapterId = computed(() => translateInit.currentChapterId.value || undefined)
const currentBookTitle = computed(() => translateInit.currentBookTitle.value)
const currentChapterTitle = computed(() => translateInit.currentChapterTitle.value)
const pageTitle = computed(() => {
  if (isBookshelfMode.value && currentChapterTitle.value && currentBookTitle.value) {
    return `${currentChapterTitle.value} - ${currentBookTitle.value}`
  }
  return '快速翻译 - Saber-Translator'
})

onMounted(async () => {
  window.addEventListener('keydown', handleKeydown)

  imageStore.clearImages()
  bubbleStore.clearBubbles()

  await translateInit.initializeApp()
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  void flushCurrentPageDocument().catch(() => {
    // Controlled route transitions report failures through their navigation guard.
  })
})

async function flushCurrentPageDocument(): Promise<void> {
  const image = imageStore.currentImage
  if (
    !image
    || image.documentRevision === undefined
    || !isPageDocumentRegistered(image.id)
  ) return
  queuePageDocumentSave(
    image.id,
    image.documentRevision,
    bubbleStore.bubbles,
  )
  await flushPageDocument(image.id)
}

async function guardDocumentFlush(): Promise<boolean> {
  try {
    if (!(await translateInit.flushChapterWorkState())) {
      throw new Error('章节工作态设置尚未写入后端')
    }
    await flushCurrentPageDocument()
    return true
  } catch (error) {
    showToast(
      `当前页写入后端失败：${error instanceof Error ? error.message : '未知错误'}`,
      'error',
    )
    return false
  }
}

onBeforeRouteUpdate(guardDocumentFlush)
onBeforeRouteLeave(guardDocumentFlush)

watch(
  () => [route.query.book, route.query.chapter],
  async ([newBook, newChapter], [previousBook, previousChapter]) => {
    if (newBook === previousBook && newChapter === previousChapter) return
    imageStore.clearImages()
    bubbleStore.clearBubbles()
    await loadChapterSession()
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

watch(
  () => bubbleStore.bubbles,
  bubbles => {
    const image = imageStore.currentImage
    if (
      !image
      || image.documentRevision === undefined
      || translateInit.isSwitchingImage.value
      || !isPageDocumentRegistered(image.id)
    ) {
      return
    }
    void queuePageDocumentSave(
      image.id,
      image.documentRevision,
      bubbles,
    ).catch(async error => {
      const message = error instanceof Error ? error.message : '未知错误'
      if (imageStore.currentImage?.id !== image.id) {
        showToast(`页面 ${image.fileName} 写入后端失败：${message}`, 'error')
        return
      }
      const restored = await translateInit.switchImage(
        imageStore.currentImageIndex,
        false,
      )
      if (error instanceof ApiClientError && error.status === 423) {
        showToast(
          restored
            ? '当前章节已被后端任务锁定，本次编辑未保存并已恢复后端版本；如需编辑，请先取消任务'
            : `当前章节已被后端任务锁定，且重新加载页面失败：${message}`,
          'warning',
        )
        return
      }
      showToast(
        restored
          ? `当前页写入后端失败，已恢复后端版本：${message}`
          : `当前页写入后端失败，且重新加载页面失败：${message}`,
        'error',
      )
    })
  },
  { deep: true },
)

const {
  goToNext,
  goToPrevious,
  handleKeydown,
  handleRetryFailed,
  handleRunWorkflow,
  handleUploadComplete,
  loadChapterSession,
  selectImage,
  toggleEditMode,
} = useTranslateViewActions({
  imageStore,
  bubbleStore,
  settingsStore,
  translation,
  translateInit,
  validateBeforeTranslation,
  isEditMode,
})

watch(
  () => currentImage.value?.id,
  pageId => {
    if (!pageId) isEditMode.value = false
  },
)

function handleWebImportCommitAccepted(jobIds: string[]): void {
  pendingContentImportJobIds.value = new Set([
    ...pendingContentImportJobIds.value,
    ...jobIds,
  ])
  void taskCenterStore.refresh().catch(() => undefined)
}

watch(
  () => [...taskCenterStore.queue, ...taskCenterStore.history],
  async jobs => {
    if (pendingContentImportJobIds.value.size === 0) return
    const terminal = jobs.filter(job => (
      pendingContentImportJobIds.value.has(job.jobId)
      && HISTORY_JOB_STATUSES.has(job.status)
    ))
    if (terminal.length === 0) return

    const remaining = new Set(pendingContentImportJobIds.value)
    terminal.forEach(job => remaining.delete(job.jobId))
    pendingContentImportJobIds.value = remaining

    imageStore.clearImages()
    bubbleStore.clearBubbles()
    await loadChapterSession()
  },
  { deep: false },
)

function openSettings() {
  showSettingsModal.value = true
}

function handleSettingsSave(payload?: { textDefaultsChanged?: boolean }) {
  if (payload?.textDefaultsChanged) {
    showToast('默认值已保存，仅用于之后导入或新建的页面', 'success')
    return
  }
  showToast('设置已保存', 'success')
}

function openSponsor() {
  showSponsorModal.value = true
}

function handleQuickWorkspaceLocked(error: unknown) {
  showToast(
    error instanceof Error
      ? `${error.message}；请先在任务中心处理相关任务或等待导入结束`
      : '快速工作区仍有活动任务或导入，请先在任务中心处理',
    'warning',
  )
  taskCenterStore.open({ bookId: translateInit.currentBookId.value || undefined })
}

async function createNewQuickWorkspace() {
  const confirmed = await confirmProductAction({
    title: '新建快速翻译',
    message: '这会永久清空当前快速工作区的页面、翻译结果和术语约束。确定继续吗？',
    confirmText: '清空并新建',
    tone: 'danger',
  })
  if (!confirmed || !(await guardDocumentFlush())) return
  const pageIds = imageStore.images.map(image => image.id)
  try {
    await resetQuickWorkspace()
  } catch (error) {
    if (
      error
      && typeof error === 'object'
      && 'status' in error
      && error.status === 423
    ) {
      handleQuickWorkspaceLocked(error)
      return
    }
    showToast(error instanceof Error ? error.message : '新建快速翻译失败', 'error')
    return
  }
  for (const pageId of pageIds) discardPageDocument(pageId)
  imageStore.clearImages()
  bubbleStore.clearBubblesLocal()
  showToast('新的快速翻译工作区已创建', 'success')
  await loadChapterSession()
}

async function handleQuickWorkspacePromoted() {
  const pageIds = imageStore.images.map(image => image.id)
  for (const pageId of pageIds) discardPageDocument(pageId)
  imageStore.clearImages()
  bubbleStore.clearBubblesLocal()
  showToast('快速翻译内容已保存到书架，当前工作区已切换为空白章节', 'success')
  await loadChapterSession()
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
          variant="solid"
          class="translate-header__back-link"
          title="返回书架"
          aria-label="返回书架"
          icon-only
        >
          <template #icon>📚</template>
        </ProductHeaderAction>
        <ProductHeaderAction
          v-if="isBookshelfMode"
          as="span"
          variant="solid"
          class="translate-header__autosave-indicator"
          title="自动保存已启用"
          aria-label="自动保存已启用"
          icon-only
        >
          <template #icon>💾</template>
        </ProductHeaderAction>
        <ProductHeaderAction
          v-if="!isBookshelfMode && hasImages"
          label="新建快速翻译"
          icon-name="plus"
          @click="createNewQuickWorkspace"
        />
        <ProductHeaderAction
          v-if="!isBookshelfMode && hasImages"
          label="保存到书架"
          icon-name="book-open"
          @click="showQuickPromoteModal = true"
        />
      </template>
      <template #actions>
        <ProductHeaderAction
          class="translate-header__settings-button"
          :class="{ 'translate-header__settings-button--highlighted': isSettingsButtonHighlighted }"
          title="打开设置"
          label="设置"
          @click="openSettings()"
        >
          <template #icon>⚙️</template>
        </ProductHeaderAction>
        <ProductHeaderAction
          as="a"
          href="http://www.mashirosaber.top"
          target="_blank"
          rel="noopener noreferrer"
          class="translate-header__link translate-header__link--tutorial"
          label="使用教程"
        />
        <ProductHeaderAction
          class="translate-header__link translate-header__link--donate"
          aria-label="请作者喝奶茶"
          title="请作者喝奶茶"
          @click="openSponsor"
        >
          <span>❤️ 请作者喝奶茶</span>
        </ProductHeaderAction>
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
              :chapter-id="currentChapterId || null"
              @upload-complete="handleUploadComplete"
            />
          </div>

          <TranslationProgress
            :progress="translation.progress.value"
          />
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
          @select="selectImage"
        />
      </template>
    </SidebarLayout>

    <EditWorkspace
      v-if="currentImage && isEditMode"
      @exit="toggleEditMode"
    />


    <FirstTimeGuide @open-settings="openSettings" />

    <SettingsModal
      v-model="showSettingsModal"
      @save="handleSettingsSave"
    />

    <BookGlossaryModal v-model="showBookGlossaryModal" />
    <BookNonTranslateModal v-model="showBookNonTranslateModal" />
    <QuickWorkspacePromoteModal
      v-model="showQuickPromoteModal"
      @locked="handleQuickWorkspaceLocked"
      @promoted="handleQuickWorkspacePromoted"
    />

    <SponsorModal
      v-if="showSponsorModal"
      @close="showSponsorModal = false"
    />

    <WebImportDisclaimer />

    <WebImportModal @commit-accepted="handleWebImportCommitAccepted" />
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
  transition: box-shadow 0.2s ease;
}

.translate-upload-card:hover {
  box-shadow: 0 8px 16px var(--shadow-medium);
}

.translate-upload-card__actions {
  display: flex;
  align-items: center;
  width: 100%;
  gap: 12px;
  flex-wrap: wrap;
}

.translate-header__link--donate {
  --product-header-action-surface: var(--color-surface-danger-soft);
  --product-header-action-hover-surface: color-mix(in srgb, var(--color-text-danger) 20%, var(--color-surface-base));
  --product-header-action-text-color: var(--color-text-danger);
  --product-header-action-line-height: 1.6;
  --product-header-action-font-size: 1rem;
  --product-header-action-font-weight: 400;
}

.translate-header__back-link {
  --product-header-action-icon-only-width: 44px;
  --product-header-action-min-height: 42px;
}

.translate-header__autosave-indicator {
  --product-header-action-icon-only-width: 44px;
  --product-header-action-min-height: 42px;
  --product-header-action-icon-font-size: 1rem;
}

.translate-header__link--tutorial,
.translate-header__link--github {
  --product-header-action-font-size: 1rem;
  --product-header-action-font-weight: 400;
}

.translate-header__autosave-indicator {
  --product-header-action-solid-surface: linear-gradient(
    135deg,
    var(--color-action-success) 0%,
    var(--color-action-success-strong) 100%
  );
  --product-header-action-solid-shadow-color: color-mix(
    in srgb,
    var(--color-action-success) 35%,
    transparent
  );
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
