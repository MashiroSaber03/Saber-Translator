<script setup lang="ts">
/**
 * 阅读器图片画布组件
 * 负责图片显示、滚动监听、原图/翻译图切换、图片懒加载
 */
import { computed, watch, onMounted, onUnmounted } from 'vue'
import type { ChapterImageData } from '@/api/bookshelf'
import UiButton from '@/components/ui/UiButton.vue'

// 组件属性
const props = defineProps<{
  /** 图片数据列表 */
  images: ChapterImageData[]
  /** 当前查看模式 */
  viewMode: 'original' | 'translated'
  /** 是否正在加载 */
  isLoading: boolean
}>()

// 组件事件
const emit = defineEmits<{
  /** 页码变化事件 */
  (e: 'pageChange', page: number): void
  /** 进入翻译页面事件 */
  (e: 'goTranslate'): void
}>()

// ==================== 计算属性 ====================

/**
 * 是否显示空状态
 */
const showEmptyState = computed(() => !props.isLoading && props.images.length === 0)

/**
 * 是否显示图片容器
 */
const showImagesContainer = computed(() => !props.isLoading && props.images.length > 0)

// ==================== 方法 ====================

/**
 * 获取图片源
 * @param imageData 图片数据
 */
function getImageSource(imageData: ChapterImageData): string {
  if (props.viewMode === 'translated') {
    // 优先显示翻译后的图片，如果没有则显示原图
    return imageData.translated || imageData.original
  }
  return imageData.original
}

/**
 * 更新页码信息
 */
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

/**
 * 处理滚动事件
 */
function handleScroll() {
  updatePageInfo()
}

/**
 * 进入翻译页面
 */
function goToTranslate() {
  emit('goTranslate')
}

// ==================== 生命周期 ====================

onMounted(() => {
  window.addEventListener('scroll', handleScroll)
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
})

// 监听图片数据变化，重新计算页码
watch(
  () => props.images,
  () => {
    // 延迟更新页码，等待 DOM 渲染完成
    setTimeout(updatePageInfo, 100)
  },
  { deep: true }
)
</script>

<template>
  <main class="reader-main">
    <!-- 加载状态 -->
    <div v-if="isLoading" id="loadingState" class="loading-state">
      <div class="loading-spinner"></div>
      <p>正在加载...</p>
    </div>

    <!-- 空状态 -->
    <div v-else-if="showEmptyState" id="emptyState" class="reader-empty-state">
      <div class="empty-icon">📖</div>
      <h2>暂无图片</h2>
      <p>该章节还没有图片，点击下方按钮开始翻译</p>
      <UiButton id="goTranslateBtn" class="reader-empty-action" variant="primary" @click="goToTranslate">
        进入翻译
      </UiButton>
    </div>

    <!-- 图片容器 -->
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
  --reader-canvas-border-default: rgba(255, 255, 255, .1);
  --reader-canvas-surface-base: rgba(255, 255, 255, .05);
  --reader-canvas-surface-raised: rgba(0, 0, 0, .6);
  --reader-canvas-text-primary: rgba(255, 255, 255, .7);
}

/* ==================== ReaderCanvas样式 ==================== */

/* 主内容区 */
.reader-main {
  min-height: calc(100dvh - 56px);
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--reader-page-background, var(--reader-view-surface-base));
}

/* 加载状态 */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-text-primary);
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--reader-canvas-border-default);
  border-top-color: var(--color-border-brand-gradient);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* 空状态 */
.reader-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100dvh - 56px);
  color: var(--reader-canvas-text-primary);
  text-align: center;
  padding: 20px;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.reader-empty-state h2 {
  margin: 0 0 8px;
  color: white;
  font-weight: 500;
}

.reader-empty-state p {
  margin: 0 0 24px;
  font-size: 14px;
}

/* 图片容器 */
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
  background: var(--reader-canvas-surface-base);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-index {
  position: absolute;
  top: 8px;
  left: 8px;
  background: var(--reader-canvas-surface-raised);
  color: white;
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
  box-shadow: 0 4px 12px var(--shadow-brand-soft);
}
</style>
