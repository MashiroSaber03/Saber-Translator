<script setup lang="ts">
/**
 * 右侧缩略图侧边栏组件
 * 显示图片概览列表，固定在页面右侧
 */

import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { useImageStore } from '@/stores/imageStore'

// ============================================================
// Props 和 Emits
// ============================================================

const emit = defineEmits<{
  /** 点击缩略图 */
  (e: 'select', index: number): void
}>()

// ============================================================
// Stores
// ============================================================

const imageStore = useImageStore()

// ============================================================
// 状态定义
// ============================================================

/** 缩略图容器引用 */
const containerRef = ref<HTMLElement | null>(null)

/** 缩略图项引用数组 */
const thumbnailRefs = ref<(HTMLElement | null)[]>([])

// ============================================================
// 计算属性
// ============================================================

/** 图片列表 */
const images = computed(() => imageStore.images)

/** 当前图片索引 */
const currentIndex = computed(() => imageStore.currentImageIndex)

/** 是否有图片 */
const hasImages = computed(() => imageStore.hasImages)

// ============================================================
// 方法
// ============================================================

/**
 * 点击缩略图
 */
function handleClick(index: number) {
  emit('select', index)
}

/**
 * 滚动到当前激活的缩略图
 */
function scrollToActiveThumbnail() {
  nextTick(() => {
    const activeThumb = thumbnailRefs.value[currentIndex.value]
    if (activeThumb && containerRef.value) {
      const container = containerRef.value
      const scrollTop = activeThumb.offsetTop - (container.clientHeight / 2) + (activeThumb.clientHeight / 2)
      container.scrollTo({
        top: Math.max(0, scrollTop),
        behavior: 'smooth'
      })
    }
  })
}

/**
 * 设置缩略图引用
 */
function setThumbnailRef(el: HTMLElement | null, index: number) {
  thumbnailRefs.value[index] = el
}

/**
 * 获取状态指示器类型（复刻原版 ui.js renderThumbnails 逻辑）
 * 优先级：失败 > 手动标注 > 处理中
 */
function getStatusType(image: typeof images.value[0]): 'failed' | 'labeled' | 'processing' | null {
  // 复刻原版：优先显示失败标记
  if (image.translationFailed) return 'failed'
  // 复刻原版：检查 savedManualCoords（原版 imageData.savedManualCoords）
  if (image.savedManualCoords && image.savedManualCoords.length > 0) return 'labeled'
  // 处理中状态
  if (image.translationStatus === 'processing') return 'processing'
  return null
}

/**
 * 获取缩略图项的额外类名（复刻原版）
 */
function getThumbnailClasses(image: typeof images.value[0]): string[] {
  const classes: string[] = []
  // 复刻原版：失败时添加 translation-failed 类
  if (image.translationFailed) {
    classes.push('translation-failed')
  }
  // 复刻原版：有手动标注时添加 has-manual-labels 类
  if (image.savedManualCoords && image.savedManualCoords.length > 0) {
    classes.push('has-manual-labels')
  }
  return classes
}

/**
 * 获取缩略图的 title 提示文本（复刻原版）
 */
function getThumbnailTitle(image: typeof images.value[0]): string {
  // 复刻原版：失败时显示特定提示
  if (image.translationFailed) return '翻译失败，点击可重试'
  // 复刻原版：有手动标注时显示特定提示
  if (image.savedManualCoords && image.savedManualCoords.length > 0) return '包含手动标注框'
  // 默认显示文件名
  return image.fileName || ''
}

// 监听当前索引变化
watch(currentIndex, () => {
  scrollToActiveThumbnail()
})

onMounted(() => {
  if (hasImages.value) {
    scrollToActiveThumbnail()
  }
})
</script>

<template>
  <aside id="thumbnail-sidebar" class="thumbnail-sidebar">
    <div class="card thumbnail-card">
      <h2>图片概览</h2>
      <ul 
        v-if="hasImages"
        ref="containerRef"
        id="thumbnailList"
        class="thumbnail-list"
      >
        <li
          v-for="(image, index) in images"
          :key="image.id"
          :ref="(el) => setThumbnailRef(el as HTMLElement | null, index)"
          class="thumbnail-item"
          :class="[
            { active: index === currentIndex },
            ...getThumbnailClasses(image)
          ]"
          :title="getThumbnailTitle(image)"
          :data-index="index"
          @click="handleClick(index)"
        >
          <!-- 复刻原版：缩略图永远显示原图 -->
          <img 
            v-if="image.originalDataURL"
            :src="image.originalDataURL" 
            :alt="image.fileName"
            class="thumbnail-image"
          >
          <!-- 复刻原版：失败标记（右下角红色感叹号） -->
          <span 
            v-if="getStatusType(image) === 'failed'"
            class="translation-failed-indicator"
          >!</span>
          <!-- 复刻原版：手动标注标记（右下角蓝色铅笔） -->
          <span 
            v-else-if="getStatusType(image) === 'labeled'"
            class="labeled-indicator"
          >✏️</span>
          <!-- 复刻原版：处理中标记（右上角旋转图标） -->
          <div 
            v-if="getStatusType(image) === 'processing'"
            class="thumbnail-processing-indicator"
          >⟳</div>
        </li>
      </ul>
      <div v-else class="empty-state">
        <p>暂无图片</p>
      </div>
    </div>
  </aside>
</template>

<style scoped>
/* ===================================
   缩略图侧边栏样式 - 复刻自 thumbnail.css
   =================================== */

#thumbnail-sidebar {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 200px;
  height: calc(100vh - 40px);
  overflow-y: auto;
  padding-top: 20px;
  box-sizing: border-box;
  margin-left: 0;
  order: 1;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e0 #f8fafc;
}

#thumbnail-sidebar::-webkit-scrollbar {
  width: 8px;
}

#thumbnail-sidebar::-webkit-scrollbar-track {
  background: #f8fafc;
  border-radius: 8px;
}

#thumbnail-sidebar::-webkit-scrollbar-thumb {
  background-color: #cbd5e0;
  border-radius: 8px;
  border: 2px solid #f8fafc;
}

#thumbnail-sidebar .thumbnail-card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  transition: transform 0.2s, box-shadow 0.2s;
}

#thumbnail-sidebar .thumbnail-card:hover {
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

#thumbnail-sidebar .thumbnail-card h2 {
  border-bottom: 2px solid #f0f0f0;
  padding-bottom: 12px;
  margin-bottom: 20px;
  color: #2c3e50;
  font-size: 1.4em;
  text-align: center;
}

#thumbnail-sidebar ul#thumbnailList {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

/* 缩略图项基础样式 */
#thumbnail-sidebar .thumbnail-item {
  margin-bottom: 10px;
  cursor: pointer;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  padding: 5px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

/* 激活和悬停状态 */
#thumbnail-sidebar .thumbnail-item.active,
#thumbnail-sidebar .thumbnail-item:hover {
  border-color: #3498db;
  box-shadow: 0 0 8px rgba(52, 152, 219, 0.5);
  transform: translateY(-2px);
}

/* 激活状态的左上角圆点标记 */
#thumbnail-sidebar .thumbnail-item.active:before {
  content: '●';
  position: absolute;
  top: 5px;
  left: 5px;
  color: #3498db;
  font-size: 18px;
  z-index: 10;
  text-shadow: 0 0 3px white;
  font-weight: bold;
}

/* 激活状态的边框叠加 */
#thumbnail-sidebar .thumbnail-item.active:after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: 3px solid #3498db;
  border-radius: 4px;
  box-sizing: border-box;
  pointer-events: none;
}

/* 缩略图图片 */
#thumbnail-sidebar .thumbnail-image {
  max-width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
}

/* 处理中指示器（右上角旋转图标） */
.thumbnail-processing-indicator {
  position: absolute;
  top: 5px;
  right: 5px;
  background-color: rgba(53, 152, 219, 0.8);
  color: white;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  z-index: 9;
  animation: pulse 1.5s infinite;
}

/* 翻译失败标记（右下角红色感叹号） */
.translation-failed-indicator {
  position: absolute;
  bottom: 3px;
  right: 3px;
  background-color: rgba(255, 0, 0, 0.8);
  color: white;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  z-index: 11;
  box-shadow: 0 0 3px black;
}

/* 手动标注指示器（右下角蓝色铅笔） */
.labeled-indicator {
  position: absolute;
  bottom: 3px;
  right: 3px;
  background-color: rgba(0, 123, 255, 0.8);
  color: white;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  z-index: 10;
  box-shadow: 0 0 3px black;
}

/* 手动标注项的左侧蓝条 */
#thumbnail-sidebar .thumbnail-item.has-manual-labels {
  border-left: 4px solid #007bff;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}
</style>
