<script setup lang="ts">
/**
 * 缩略图列表组件
 * 显示已上传图片的缩略图列表，支持点击切换、状态指示和自动滚动
 * 
 * 功能：
 * - 缩略图列表显示
 * - 点击切换图片
 * - 状态指示器（失败、处理中、完成、手动标注）
 * - 当前图片高亮
 * - 自动滚动到当前激活的缩略图
 */

import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { useImageStore } from '@/stores/imageStore'

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 是否显示状态指示器 */
  showStatus?: boolean
  /** 缩略图大小（像素） */
  thumbnailSize?: number
}>()

const emit = defineEmits<{
  /** 点击缩略图 */
  (e: 'select', index: number): void
  /** 双击缩略图 */
  (e: 'dblclick', index: number): void
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

/** 缩略图大小样式 */
const thumbnailStyle = computed(() => {
  const size = props.thumbnailSize || 80
  return {
    width: `${size}px`,
    height: `${size}px`,
  }
})

// ============================================================
// 方法
// ============================================================

/**
 * 点击缩略图
 * @param index 图片索引
 */
function handleClick(index: number) {
  imageStore.setCurrentImageIndex(index)
  emit('select', index)
}

/**
 * 双击缩略图
 * @param index 图片索引
 */
function handleDblClick(index: number) {
  emit('dblclick', index)
}

/**
 * 滚动到当前激活的缩略图
 */
function scrollToActiveThumbnail() {
  nextTick(() => {
    const activeThumb = thumbnailRefs.value[currentIndex.value]
    if (activeThumb && containerRef.value) {
      // 计算滚动位置，使激活的缩略图居中显示
      const container = containerRef.value
      const thumbRect = activeThumb.getBoundingClientRect()
      const containerRect = container.getBoundingClientRect()
      
      // 判断是水平还是垂直滚动
      const isHorizontal = container.scrollWidth > container.clientWidth
      
      if (isHorizontal) {
        // 水平滚动
        const scrollLeft = activeThumb.offsetLeft - (containerRect.width / 2) + (thumbRect.width / 2)
        container.scrollTo({
          left: Math.max(0, scrollLeft),
          behavior: 'smooth'
        })
      } else {
        // 垂直滚动
        const scrollTop = activeThumb.offsetTop - (containerRect.height / 2) + (thumbRect.height / 2)
        container.scrollTo({
          top: Math.max(0, scrollTop),
          behavior: 'smooth'
        })
      }
    }
  })
}

/**
 * 获取图片状态类名
 * @param image 图片数据
 */
function getStatusClass(image: typeof images.value[0]) {
  const classes: string[] = []
  
  if (image.translationFailed) {
    classes.push('failed')
  }
  
  if (image.translationStatus === 'processing') {
    classes.push('processing')
  }
  
  if (image.translationStatus === 'completed') {
    classes.push('completed')
  }
  
  if (image.isManualAnnotation) {
    classes.push('annotated')
  }
  
  return classes.join(' ')
}

/**
 * 获取状态图标
 * @param image 图片数据
 */
function getStatusIcon(image: typeof images.value[0]): string | null {
  if (image.translationFailed) {
    return '❌'
  }
  if (image.translationStatus === 'processing') {
    return '⏳'
  }
  if (image.translationStatus === 'completed' && !image.isManualAnnotation) {
    return '✓'
  }
  if (image.isManualAnnotation) {
    return '✏️'
  }
  return null
}

/**
 * 设置缩略图引用
 * @param el 元素
 * @param index 索引
 */
function setThumbnailRef(el: HTMLElement | null, index: number) {
  thumbnailRefs.value[index] = el
}

// ============================================================
// 生命周期和监听
// ============================================================

// 监听当前索引变化，自动滚动到激活的缩略图
watch(currentIndex, () => {
  scrollToActiveThumbnail()
})

// 组件挂载时滚动到当前激活的缩略图
onMounted(() => {
  if (hasImages.value) {
    scrollToActiveThumbnail()
  }
})

// 暴露方法供父组件调用
defineExpose({
  scrollToActiveThumbnail,
})
</script>

<template>
  <div 
    v-if="hasImages"
    ref="containerRef"
    class="thumbnail-list"
    id="uploadThumbnailList"
  >
    <div 
      v-for="(image, index) in images" 
      :key="image.id"
      :ref="(el) => setThumbnailRef(el as HTMLElement | null, index)"
      class="thumbnail-item"
      :class="[
        { active: index === currentIndex },
        getStatusClass(image)
      ]"
      :style="thumbnailStyle"
      :title="image.fileName"
      @click="handleClick(index)"
      @dblclick="handleDblClick(index)"
    >
      <!-- 缩略图图片 -->
      <img 
        v-if="image.originalDataURL"
        :src="image.originalDataURL" 
        :alt="image.fileName"
        class="thumbnail-image"
      >
      
      <!-- 序号标签 -->
      <span class="thumbnail-index">{{ index + 1 }}</span>
      
      <!-- 状态图标 -->
      <span 
        v-if="showStatus !== false && getStatusIcon(image)"
        class="status-icon"
        :class="getStatusClass(image)"
      >
        {{ getStatusIcon(image) }}
      </span>
      
      <!-- 处理中动画遮罩 -->
      <div 
        v-if="image.translationStatus === 'processing'"
        class="processing-overlay"
      >
        <div class="processing-spinner"></div>
      </div>
    </div>
  </div>
  
  <!-- 空状态 -->
  <div v-else class="thumbnail-empty">
    <span class="empty-text">暂无图片</span>
  </div>
</template>

<style scoped>
/* 缩略图列表容器 */
.thumbnail-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px;
  max-height: 300px;
  overflow-y: auto;
  background: var(--secondary-bg, #f5f5f5);
  border-radius: 6px;
}

/* 缩略图项 */
.thumbnail-item {
  position: relative;
  cursor: pointer;
  border: 2px solid transparent;
  border-radius: 4px;
  overflow: hidden;
  transition: all 0.2s ease;
  background: var(--card-bg, #fff);
  flex-shrink: 0;
}

.thumbnail-item:hover {
  border-color: var(--primary-color, #4a90d9);
  transform: scale(1.02);
}

/* 当前激活状态 */
.thumbnail-item.active {
  border-color: var(--primary-color, #4a90d9);
  box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.3);
}

/* 失败状态 */
.thumbnail-item.failed {
  border-color: var(--error-color, #e74c3c);
}

.thumbnail-item.failed:hover {
  border-color: var(--error-color, #e74c3c);
}

/* 处理中状态 */
.thumbnail-item.processing {
  border-color: var(--warning-color, #f39c12);
}

/* 完成状态 */
.thumbnail-item.completed {
  border-color: var(--success-color, #27ae60);
}

/* 手动标注状态 */
.thumbnail-item.annotated {
  border-color: var(--info-color, #3498db);
}

/* 缩略图图片 */
.thumbnail-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* 序号标签 */
.thumbnail-index {
  position: absolute;
  bottom: 2px;
  left: 2px;
  font-size: 10px;
  font-weight: 500;
  color: #fff;
  background: rgba(0, 0, 0, 0.6);
  padding: 1px 4px;
  border-radius: 2px;
  line-height: 1.2;
}

/* 状态图标 */
.status-icon {
  position: absolute;
  top: 2px;
  right: 2px;
  font-size: 12px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 50%;
  padding: 2px;
  line-height: 1;
  min-width: 16px;
  min-height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.status-icon.failed {
  color: var(--error-color, #e74c3c);
}

.status-icon.processing {
  animation: pulse 1s infinite;
}

.status-icon.completed {
  color: var(--success-color, #27ae60);
}

.status-icon.annotated {
  color: var(--info-color, #3498db);
}

/* 处理中动画遮罩 */
.processing-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
}

.processing-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* 空状态 */
.thumbnail-empty {
  padding: 20px;
  text-align: center;
  color: var(--text-secondary, #666);
  background: var(--secondary-bg, #f5f5f5);
  border-radius: 6px;
}

.empty-text {
  font-size: 14px;
}

/* 动画 */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 响应式：小屏幕时水平滚动 */
@media (max-width: 768px) {
  .thumbnail-list {
    flex-wrap: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    max-height: none;
  }
}
</style>
