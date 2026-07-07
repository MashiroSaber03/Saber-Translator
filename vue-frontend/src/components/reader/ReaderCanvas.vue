<script setup lang="ts">
import { computed, watch, onBeforeUpdate, onMounted, onUnmounted, ref } from 'vue'
import type { ChapterImageData } from '@/api/bookshelf'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

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
const imageWrapperRefs = ref<HTMLElement[]>([])

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
  const images = imageWrapperRefs.value
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

function setImageWrapperRef(el: Element | null) {
  if (el instanceof HTMLElement) {
    imageWrapperRefs.value.push(el)
  }
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

onBeforeUpdate(() => {
  imageWrapperRefs.value = []
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
  clearPageInfoTimer()
  imageWrapperRefs.value = []
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
  <main class="reader-canvas">
    <div v-if="isLoading" class="reader-canvas__loading-state">
      <UiSpinner
        size="48px"
        label="正在加载阅读内容"
        :decorative="false"
      />
      <p class="reader-canvas__loading-text">正在加载...</p>
    </div>

    <ProductEmptyState
      v-else-if="showEmptyState"
      class="reader-canvas__empty-state"
      icon-name="book-open"
      title="暂无图片"
      description="该章节还没有图片，点击下方按钮开始翻译"
      variant="inverse"
    >
      <template #actions>
        <UiButton variant="primary" @click="goToTranslate">
          进入翻译
        </UiButton>
      </template>
    </ProductEmptyState>

    <div v-else-if="showImagesContainer" class="reader-canvas__images">
      <div
        v-for="(img, index) in images"
        :key="index"
        :ref="setImageWrapperRef"
        class="reader-canvas__image-wrapper"
      >
        <img
          class="reader-canvas__image"
          :src="getImageSource(img)"
          :alt="`第 ${index + 1} 页`"
          loading="lazy"
        />
        <div class="reader-canvas__image-index">{{ index + 1 }} / {{ images.length }}</div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.reader-canvas {
  --reader-canvas-page-background: var(--color-surface-inverse);
  --reader-canvas-muted-text: color-mix(in srgb, var(--color-text-inverse) 70%, transparent);
  --reader-canvas-page-index-background: color-mix(in srgb, var(--color-surface-inverse-depth) 60%, transparent);

  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: calc(100dvh - 56px);
  background: var(--reader-page-background, var(--reader-canvas-page-background));
}

.reader-canvas__loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-muted-text);
}

.reader-canvas__loading-text {
  margin: 0;
}

.reader-canvas__empty-state {
  --product-empty-state-min-height: calc(100dvh - 56px);
}

.reader-canvas__images {
  width: 100%;
  max-width: var(--reader-max-width, 100%);
  padding: 16px 0 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--reader-gap, 8px);
}

.reader-canvas__image-wrapper {
  width: var(--reader-image-width, 100%);
  max-width: 1200px;
  position: relative;
}

.reader-canvas__image {
  width: 100%;
  height: auto;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

.reader-canvas__image-index {
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

.reader-canvas__image-wrapper:hover .reader-canvas__image-index {
  opacity: 1;
}
</style>
