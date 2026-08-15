<script setup lang="ts">
import AppShell from '@/components/ui/AppShell.vue'
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  getBook,
  listChapterPages,
  type V2BookDetail,
  type V2Chapter,
  type V2PageSummary,
} from '@/api/v2/content'
import { useToast } from '@/utils/toast'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ReaderControls from '@/components/reader/ReaderControls.vue'
import { DEFAULT_READER_SETTINGS, type ReaderSettings } from '@/components/reader/readerSettings'

const props = defineProps<{
  bookId: string
  chapterId: string
}>()

const router = useRouter()
const toast = useToast()

const bookInfo = ref<V2BookDetail | null>(null)
const chaptersData = ref<V2Chapter[]>([])
const currentChapterInfo = computed(() => chaptersData.value.find(c => c.id === props.chapterId))

const imagesData = ref<V2PageSummary[]>([])
const isLoading = ref(true)
const currentViewMode = ref<'original' | 'translated'>('translated')
const currentPage = ref(1)
const settingsRequestId = ref(0)
const readerSettings = ref<ReaderSettings>({ ...DEFAULT_READER_SETTINGS })

let failureRedirectTimer: ReturnType<typeof setTimeout> | null = null
let readerLoadSequence = 0
let isReaderViewMounted = false

const currentChapterIndex = computed(() =>
  chaptersData.value.findIndex(c => c.id === props.chapterId)
)

const hasPrevChapter = computed(() => currentChapterIndex.value > 0)

const hasNextChapter = computed(
  () => currentChapterIndex.value >= 0 && currentChapterIndex.value < chaptersData.value.length - 1
)

const pageTitle = computed(() => {
  const chapterTitle = currentChapterInfo.value?.title || '阅读'
  const bookTitle = bookInfo.value?.title || 'Saber-Translator'
  return `${chapterTitle} - ${bookTitle}`
})

const showChapterNav = computed(() => !isLoading.value && imagesData.value.length > 0)
const translatedPageCount = computed(
  () => imagesData.value.filter(page => Boolean(page.translatedUrl)).length
)

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
  const loadId = ++readerLoadSequence
  const bookId = props.bookId
  const chapterId = props.chapterId
  clearFailureRedirectTimer()
  isLoading.value = true
  currentPage.value = 1
  bookInfo.value = null
  chaptersData.value = []
  imagesData.value = []

  try {
    const [bookResult, imagesResult] = await Promise.all([
      getBook(bookId),
      listChapterPages(chapterId, { all: true }),
    ])

    if (!isReaderViewMounted || loadId !== readerLoadSequence) return

    const chapter = bookResult.chapters.find(item => item.id === chapterId)
    if (!chapter) {
      throw new Error('章节不属于当前书籍')
    }
    if (imagesResult.nextCursor !== null) {
      throw new Error('章节页面列表不完整')
    }
    if (imagesResult.items.some(page => page.chapterId !== chapterId)) {
      throw new Error('章节页面归属不一致')
    }

    bookInfo.value = bookResult
    chaptersData.value = bookResult.chapters
    imagesData.value = imagesResult.items

    document.title = pageTitle.value
  } catch (error) {
    if (!isReaderViewMounted || loadId !== readerLoadSequence) return
    toast.error('加载失败: ' + (error instanceof Error ? error.message : '未知错误'))
    scheduleFailureRedirect()
  } finally {
    if (isReaderViewMounted && loadId === readerLoadSequence) {
      isLoading.value = false
    }
  }
}

function setViewMode(mode: 'original' | 'translated') {
  currentViewMode.value = mode
}

function navigateChapter(direction: 'prev' | 'next') {
  const newIndex =
    direction === 'prev' ? currentChapterIndex.value - 1 : currentChapterIndex.value + 1

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
  settingsRequestId.value += 1
}

function handlePageChange(page: number) {
  currentPage.value = page
}

function handleSettingsChange(settings: ReaderSettings) {
  readerSettings.value = { ...settings }
}

onMounted(() => {
  isReaderViewMounted = true
  void loadReaderData()
})

onUnmounted(() => {
  isReaderViewMounted = false
  readerLoadSequence += 1
  clearFailureRedirectTimer()
})

watch(
  () => [props.bookId, props.chapterId],
  () => {
    void loadReaderData()
  }
)
</script>

<template>
  <AppShell
    class="reader-page"
    chrome="fixed"
    header-height="56px"
    content-padding="0 20px"
    content-class="reader-page__content"
  >
    <template #header>
      <ProductPageHeader variant="reader">
        <template #brand>
          <div class="reader-header__left">
            <ProductHeaderAction
              class="reader-header__button"
              title="返回书架"
              label="返回"
              collapse-label-on-mobile
              @click="goBack"
            >
              <template #icon>←</template>
            </ProductHeaderAction>
            <div class="reader-header__book-info">
              <span class="reader-header__book-title">{{ bookInfo?.title || '加载中...' }}</span>
              <span class="reader-header__separator">·</span>
              <span class="reader-header__chapter-title">{{
                currentChapterInfo?.title || '-'
              }}</span>
            </div>
          </div>
        </template>

        <template #meta>
          <span class="reader-header__page-info">{{ currentPage }} / {{ imagesData.length || '-' }}</span>
        </template>

        <template #actions>
          <div class="reader-header__right">
            <div class="reader-header__view-mode-toggle">
              <ProductHeaderAction
                class="reader-header__mode-button"
                :active="currentViewMode === 'original'"
                :pressed="currentViewMode === 'original'"
                data-mode="original"
                title="查看原图"
                label="原图"
                @click="setViewMode('original')"
              />
              <ProductHeaderAction
                class="reader-header__mode-button"
                :active="currentViewMode === 'translated'"
                :pressed="currentViewMode === 'translated'"
                data-mode="translated"
                title="查看翻译"
                label="翻译"
                @click="setViewMode('translated')"
              />
            </div>
            <span class="reader-header__translated-count" aria-label="已翻译页面数量">
              已翻译 {{ translatedPageCount }}/{{ imagesData.length }}
            </span>
            <ProductHeaderAction
              class="reader-header__button"
              title="阅读设置"
              aria-label="阅读设置"
              icon-only
              @click="openSettings"
            >
              <template #icon>⚙️</template>
            </ProductHeaderAction>
            <ProductHeaderAction
              variant="solid"
              class="reader-header__button reader-header__button--primary"
              title="进入翻译页面"
              label="翻译"
              collapse-label-on-mobile
              @click="goToTranslate"
            >
              <template #icon>✏️</template>
            </ProductHeaderAction>
          </div>
        </template>
      </ProductPageHeader>
    </template>

    <ReaderCanvas
      :images="imagesData"
      :view-mode="currentViewMode"
      :is-loading="isLoading"
      :image-width="readerSettings.imageWidth"
      :image-gap="readerSettings.imageGap"
      :background-color="readerSettings.bgColor"
      @page-change="handlePageChange"
      @go-translate="goToTranslate"
    />

    <ReaderControls
      :has-prev-chapter="hasPrevChapter"
      :has-next-chapter="hasNextChapter"
      :show-chapter-nav="showChapterNav"
      :settings-request-id="settingsRequestId"
      @navigate-chapter="navigateChapter"
      @settings-change="handleSettingsChange"
    />
  </AppShell>
</template>

<style scoped>
.reader-page {
  width: 100%;
  margin: 0;
  padding: 0;
  overflow-x: hidden;
  background: var(--color-surface-page);
}

.reader-page__content {
  width: 100%;
}

.reader-header__left,
.reader-header__right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.reader-header__button,
.reader-header__mode-button {
  --product-header-action-min-height: 36px;
  --product-header-action-radius: 8px;
}

.reader-header__button {
  --product-header-action-gap: 4px;
}

.reader-header__button.product-header-action--icon-only {
  width: 44px;
}

.reader-header__book-info {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-inverse);
  font-size: 14px;
}

.reader-header__book-title {
  max-width: 200px;
  overflow: hidden;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.reader-header__separator {
  opacity: 0.6;
}

.reader-header__chapter-title {
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  opacity: 0.9;
}

.reader-header__page-info {
  color: var(--color-text-inverse);
  font-size: 14px;
  opacity: 0.9;
}

.reader-header__view-mode-toggle {
  display: flex;
  gap: 2px;
  overflow: hidden;
  border-radius: 8px;
  background: var(--product-header-action-context-surface, var(--color-overlay-inverse-soft));
}

.reader-header__mode-button {
  border-radius: 8px;
}

.reader-header__translated-count {
  display: inline-flex;
  align-items: center;
  padding: 0;
  color: var(--color-text-inverse);
  font-size: 12px;
  white-space: nowrap;
  opacity: 0.9;
}

@media (--breakpoint-md-down) {
  .reader-header__book-title {
    max-width: 120px;
  }

  .reader-header__chapter-title {
    max-width: 80px;
  }

  .reader-header__mode-button {
    padding: 8px 12px;
    font-size: 12px;
  }

  .reader-header__translated-count {
    display: none;
  }
}

@media (--breakpoint-xs-down) {
  .reader-header__book-info {
    display: none;
  }

  .reader-header__page-info {
    display: none;
  }

  .reader-header__view-mode-toggle {
    gap: 0;
  }
}
</style>
