<script setup lang="ts">
/**
 * 阅读器图片画布组件
 * 负责图片显示、滚动监听、原图/翻译图切换、图片懒加载
 */
import { computed, watch, onMounted, onUnmounted } from 'vue'
import type { ChapterImageData } from '@/api/bookshelf'

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
    <div v-else-if="showEmptyState" id="emptyState" class="empty-state">
      <div class="empty-icon">📖</div>
      <h2>暂无图片</h2>
      <p>该章节还没有图片，点击下方按钮开始翻译</p>
      <button id="goTranslateBtn" class="btn btn-primary" @click="goToTranslate">
        进入翻译
      </button>
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
/* ==================== ReaderCanvas 完整样式 - 从 reader.css 迁移 ==================== */

/* 主内容区 */
.reader-main {
  padding-top: 56px;
  min-height: calc(100vh - 56px);
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* 加载状态 */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100vh - 56px);
  color: rgb(255, 255, 255, 0.7);
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 3px solid rgb(255, 255, 255, 0.1);
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* @keyframes spin 已迁移到 global.css */

.loading-state p {
  margin-top: 16px;
  font-size: 14px;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: calc(100vh - 56px);
  color: rgb(255, 255, 255, 0.7);
  text-align: center;
  padding: 20px;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.empty-state h2 {
  margin: 0 0 8px;
  color: white;
  font-weight: 500;
}

.empty-state p {
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
  background: rgb(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-index {
  position: absolute;
  top: 8px;
  left: 8px;
  background: rgb(0, 0, 0, 0.6);
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

/* 按钮样式 */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgb(102, 126, 234, 0.4);
}
</style>
