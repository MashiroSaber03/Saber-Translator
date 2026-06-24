<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import AppShell from '@/components/ui/AppShell.vue'
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { getBookDetail, getChapterImages, type ChapterImageData } from '@/api/bookshelf'
import type { BookData, ChapterData } from '@/types'
import { useToast } from '@/utils/toast'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ReaderControls from '@/components/reader/ReaderControls.vue'
import type { ReaderSettings } from '@/components/reader/ReaderControls.vue'

const props = defineProps<{
  bookId: string
  chapterId: string
}>()

const router = useRouter()
const toast = useToast()

const readerControlsRef = ref<InstanceType<typeof ReaderControls> | null>(null)

const bookInfo = ref<BookData | null>(null)
const chaptersData = ref<ChapterData[]>([])
const currentChapterInfo = computed(() => 
  chaptersData.value.find(c => c.id === props.chapterId)
)

const imagesData = ref<ChapterImageData[]>([])
const isLoading = ref(true)
const currentViewMode = ref<'original' | 'translated'>('translated')
const currentPage = ref(1)

let failureRedirectTimer: ReturnType<typeof setTimeout> | null = null

const currentChapterIndex = computed(() => 
  chaptersData.value.findIndex(c => c.id === props.chapterId)
)

const hasPrevChapter = computed(() => currentChapterIndex.value > 0)

const hasNextChapter = computed(() => 
  currentChapterIndex.value >= 0 && 
  currentChapterIndex.value < chaptersData.value.length - 1
)

const pageTitle = computed(() => {
  const chapterTitle = currentChapterInfo.value?.title || '阅读'
  const bookTitle = bookInfo.value?.title || 'Saber-Translator'
  return `${chapterTitle} - ${bookTitle}`
})

const showChapterNav = computed(() => !isLoading.value && imagesData.value.length > 0)

function clearFailureRedirectTimer() {
  if (failureRedirectTimer !== null) {
    clearTimeout(failureRedirectTimer)
    failureRedirectTimer = null
  }
}

function scheduleFailureRedirect() {
  clearFailureRedirectTimer()
  failureRedirectTimer = setTimeout(() => {
    failureRedirectTimer = null
    router.push('/')
  }, 2000)
}

async function loadReaderData() {
  clearFailureRedirectTimer()
  isLoading.value = true
  
  try {
    const [bookResult, imagesResult] = await Promise.all([
      getBookDetail(props.bookId),
      getChapterImages(props.bookId, props.chapterId)
    ])
    
    if (bookResult.success && bookResult.book) {
      bookInfo.value = bookResult.book
      chaptersData.value = bookResult.book.chapters || []
    } else {
      throw new Error(bookResult.error || '获取书籍信息失败')
    }
    
    if (imagesResult.success && imagesResult.images) {
      imagesData.value = imagesResult.images
    } else {
      throw new Error(imagesResult.error || '获取章节图片失败')
    }
    
    document.title = pageTitle.value
    
  } catch (error) {
    toast.error('加载失败: ' + (error instanceof Error ? error.message : '未知错误'))
    scheduleFailureRedirect()
  } finally {
    isLoading.value = false
  }
}

function setViewMode(mode: 'original' | 'translated') {
  currentViewMode.value = mode
}

function navigateChapter(direction: 'prev' | 'next') {
  const newIndex = direction === 'prev' 
    ? currentChapterIndex.value - 1 
    : currentChapterIndex.value + 1
  
  if (newIndex >= 0 && newIndex < chaptersData.value.length) {
    const newChapter = chaptersData.value[newIndex]
    if (newChapter) {
      router.push(`/reader?book=${props.bookId}&chapter=${newChapter.id}`)
    }
  }
}

function goBack() {
  router.push('/')
}

function goToTranslate() {
  router.push(`/translate?book=${props.bookId}&chapter=${props.chapterId}`)
}

function openSettings() {
  readerControlsRef.value?.openSettings()
}

function handlePageChange(page: number) {
  currentPage.value = page
}

function handleSettingsChange(_settings: ReaderSettings) {
  // 设置已在 ReaderControls 组件中应用
}

onMounted(() => {
  void loadReaderData()
})

onUnmounted(() => {
  clearFailureRedirectTimer()
})

watch(
  () => [props.bookId, props.chapterId],
  () => {
    void loadReaderData()
    window.scrollTo({ top: 0 })
  }
)
</script>

<template>
  <AppShell
    class="reader-page"
    chrome="fixed"
    header-height="56px"
    header-offset="56px"
    content-padding="0 20px"
    content-class="reader-page__content"
  >
    <template #header>
      <header class="reader-header">
        <div class="reader-header__left">
          <UiButton variant="toolbar" id="backBtn" class="reader-header__button" title="返回书架" @click="goBack">
            <span class="reader-header__button-icon">←</span>
            <span class="reader-header__button-text">返回</span>
          </UiButton>
          <div class="book-info">
            <span id="bookTitle" class="reader-header__book-title">{{ bookInfo?.title || '加载中...' }}</span>
            <span class="separator">·</span>
            <span id="chapterTitle" class="chapter-title">{{ currentChapterInfo?.title || '-' }}</span>
          </div>
        </div>
        <div class="reader-header__center">
          <span id="pageInfo" class="page-info">{{ currentPage }} / {{ imagesData.length || '-' }}</span>
        </div>
        <div class="reader-header__right">
          <div class="view-mode-toggle">
            <UiButton
              variant="toolbar" 
              id="viewOriginalBtn" 
              class="reader-header__mode-button" 
              :class="{ active: currentViewMode === 'original' }"
              data-mode="original" 
              title="查看原图"
              @click="setViewMode('original')"
            >
              原图
            </UiButton>
            <UiButton
              variant="toolbar" 
              id="viewTranslatedBtn" 
              class="reader-header__mode-button" 
              :class="{ active: currentViewMode === 'translated' }"
              data-mode="translated" 
              title="查看翻译"
              @click="setViewMode('translated')"
            >
              翻译
            </UiButton>
          </div>
          <UiButton variant="toolbar" id="settingsBtn" class="reader-header__button" title="阅读设置" aria-label="阅读设置" @click="openSettings">
            <span class="reader-header__button-icon">⚙️</span>
          </UiButton>
          <UiButton variant="primary" id="translateBtn" class="reader-header__button reader-header__button--primary" title="进入翻译页面" @click="goToTranslate">
            <span class="reader-header__button-icon">✏️</span>
            <span class="reader-header__button-text">翻译</span>
          </UiButton>
        </div>
      </header>
    </template>

    <ReaderCanvas
      :images="imagesData"
      :view-mode="currentViewMode"
      :is-loading="isLoading"
      @page-change="handlePageChange"
      @go-translate="goToTranslate"
    />

    <ReaderControls
      ref="readerControlsRef"
      :current-page="currentPage"
      :total-pages="imagesData.length"
      :has-prev-chapter="hasPrevChapter"
      :has-next-chapter="hasNextChapter"
      :show-chapter-nav="showChapterNav"
      @navigate-chapter="navigateChapter"
      @settings-change="handleSettingsChange"
    />
  </AppShell>
</template>

<style scoped>
.reader-page {
  --reader-view-shadow-default: rgba(0, 0, 0, .2);
  --reader-view-surface-base: #1a1a2e;
  --reader-view-surface-raised: rgba(255, 255, 255, .15);
  --reader-view-surface-muted: rgba(255, 255, 255, .25);
  --reader-view-surface-subtle: rgba(255, 255, 255, .9);
  --reader-view-text-primary: #667eea;
  --reader-view-text-secondary: rgba(255, 255, 255, .7);

    width: 100%;
    margin: 0;
    padding: 0;
    background: var(--color-surface-page);
    overflow-x: hidden;
}

.reader-page__content {
    width: 100%;
}

.reader-header {
    height: 56px;
    background: var(--header-bg, linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%));
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    z-index: var(--z-overlay);
    box-shadow: 0 2px 10px var(--reader-view-shadow-default);
}

.reader-header__left,
.reader-header__right {
    display: flex;
    align-items: center;
    gap: 12px;
}

.reader-header__center {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
}

.reader-header__button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--reader-view-surface-raised);
    border: none;
    border-radius: 8px;
    color: var(--color-text-inverse);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
}

.reader-header__button:hover {
    background: var(--reader-view-surface-muted);
}

.reader-header__button--primary {
    background: var(--reader-view-surface-subtle);
    color: var(--reader-view-text-primary);
}

.reader-header__button--primary:hover {
    background: var(--color-surface-base);
}

.reader-header__button-icon {
    font-size: 16px;
}

.book-info {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--color-text-inverse);
    font-size: 14px;
}

.reader-header__book-title {
    font-weight: 600;
    max-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.separator {
    opacity: 0.6;
}

.chapter-title {
    opacity: 0.9;
    max-width: 150px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.page-info {
    color: var(--color-text-inverse);
    font-size: 14px;
    opacity: 0.9;
}

.view-mode-toggle {
    display: flex;
    background: var(--reader-view-surface-raised);
    border-radius: 8px;
    overflow: hidden;
}

.reader-header__mode-button {
    padding: 8px 16px;
    background: transparent;
    border: none;
    color: var(--reader-view-text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
}

.reader-header__mode-button:hover {
    color: var(--color-text-inverse);
}

.reader-header__mode-button.active {
    background: var(--reader-view-surface-subtle);
    color: var(--reader-view-text-primary);
    font-weight: 500;
}

@media (--breakpoint-md-down) {
    .reader-header__button .reader-header__button-text {
        display: none;
    }
    
    .reader-header__book-title {
        max-width: 120px;
    }
    
    .chapter-title {
        max-width: 80px;
    }
    
    .reader-header__center {
        display: none;
    }
    
    .reader-header__mode-button {
        padding: 8px 12px;
        font-size: 12px;
    }
}

@media (--breakpoint-xs-down) {
    .reader-header {
        padding: 0 8px;
    }
    
    .book-info {
        display: none;
    }
    
    .view-mode-toggle {
        gap: 0;
    }
}
</style>
