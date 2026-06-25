<script setup lang="ts">
import { computed, watch, onMounted, onUnmounted } from 'vue'
import type { ChapterImageData } from '@/api/bookshelf'
import UiButton from '@/components/ui/UiButton.vue'

const props = defineProps<{
  images: ChapterImageData[]
  viewMode: 'original' | 'translated'
  isLoading: boolean
}>()

const emit = defineEmits<{
  (e: 'pageChange', page: number): void
  (e: 'goTranslate'): void
}>()

const showEmptyState = computed(() => !props.isLoading && props.images.length === 0)
const showImagesContainer = computed(() => !props.isLoading && props.images.length > 0)

let pageInfoTimer: ReturnType<typeof setTimeout> | null = null

function clearPageInfoTimer() {
  if (pageInfoTimer !== null) {
    clearTimeout(pageInfoTimer)
    pageInfoTimer = null
  }
}

function getImageSource(imageData: ChapterImageData): string {
  if (props.viewMode === 'translated') {
    return imageData.translated || imageData.original
  }
  return imageData.original
}

function updatePageInfo() {
  const images = document.querySelectorAll('.reader-image-wrapper')
  const viewportCenter = window.innerHeight / 2
  let currentPage = 1
  
  images.forEach((img, index) => {
    const rect = img.getBoundingClientRect()
    if (rect.top < viewportCenter && rect.bottom > 0) {
      currentPage = index + 1
    }
  })
  
  emit('pageChange', currentPage)
}

function handleScroll() {
  updatePageInfo()
}

function schedulePageInfoUpdate() {
  clearPageInfoTimer()
  pageInfoTimer = setTimeout(() => {
    pageInfoTimer = null
    updatePageInfo()
  }, 100)
}

function goToTranslate() {
  emit('goTranslate')
}

onMounted(() => {
  window.addEventListener('scroll', handleScroll)
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
  clearPageInfoTimer()
})

watch(
  () => props.images,
  () => {
    schedulePageInfoUpdate()
  },
  { deep: true }
)
</script>

<template>
  <main class="reader-main">
    <div v-if="isLoading" id="loadingState" class="loading-state">
      <div class="loading-spinner"></div>
      <p>正在加载...</p>
    </div>

    <div v-else-if="showEmptyState" id="emptyState" class="reader-empty-state">
      <div class="empty-icon">📖</div>
      <h2>暂无图片</h2>
      <p>该章节还没有图片，点击下方按钮开始翻译</p>
      <UiButton id="goTranslateBtn" class="reader-empty-action" variant="primary" @click="goToTranslate">
        进入翻译
      </UiButton>
    </div>

    <div v-else-if="showImagesContainer" id="imagesContainer" class="images-container">
      <div 
        v-for="(img, index) in images" 
        :key="index" 
        class="reader-image-wrapper"
      >
        <img 
          class="reader-image" 
          :src="getImageSource(img)" 
          :alt="`第 ${index + 1} 页`"
          loading="lazy"
        />
        <div class="image-index">{{ index + 1 }} / {{ images.length }}</div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.reader-main {
  --reader-canvas-page-background: #1a1a2e;
  --reader-canvas-muted-text: rgba(255, 255, 255, .7);
  --reader-canvas-spinner-track: rgba(255, 255, 255, .1);
  --reader-canvas-image-loading-background: rgba(255, 255, 255, .05);
  --reader-canvas-page-index-background: rgba(0, 0, 0, .6);
}

.reader-main {
  min-height: calc(100dvh - 56px);
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--reader-page-background, var(--reader-canvas-page-background));
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-muted-text);
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--reader-canvas-spinner-track);
  border-top-color: var(--color-border-brand-gradient);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.reader-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-muted-text);
  text-align: center;
  padding: 20px;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.reader-empty-state h2 {
  margin: 0 0 8px;
  color: var(--color-text-inverse);
  font-weight: 500;
}

.reader-empty-state p {
  margin: 0 0 24px;
  font-size: 14px;
}

.images-container {
  width: 100%;
  max-width: var(--reader-max-width, 100%);
  padding: 16px 0 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--reader-gap, 8px);
}

.reader-image-wrapper {
  width: var(--reader-image-width, 100%);
  max-width: 1200px;
  position: relative;
}

.reader-image {
  width: 100%;
  height: auto;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

.reader-image.loading {
  min-height: 300px;
  background: var(--reader-canvas-image-loading-background);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-index {
  position: absolute;
  top: 8px;
  left: 8px;
  background: var(--reader-canvas-page-index-background);
  color: var(--color-text-inverse);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  opacity: 0;
  transition: opacity 0.2s;
}

.reader-image-wrapper:hover .image-index {
  opacity: 1;
}

.reader-empty-action {
  padding: 12px 24px;
}

.reader-empty-action:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--shadow-action-brand);
}
</style>
