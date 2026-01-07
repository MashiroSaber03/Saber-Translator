<script setup lang="ts">
/**
 * 右侧缩略图侧边栏组件
 * 显示图片概览列表，固定在页面右侧
 * 
 * 支持两种模式：
 * - 扁平模式：普通的图片列表
 * - 文件夹模式：面包屑导航 + 扁平列表（无缩进）
 */

import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useFolderTree } from '@/composables/useFolderTree'
import type { ImageData } from '@/types/image'
import type { FolderNode } from '@/types/folder'

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
// 文件夹树逻辑（使用 Composable）
// ============================================================

const { 
  useTreeMode, 
  breadcrumbs,
  currentSubfolders,
  currentImages,
  currentFolderPath,
  enterFolder,
  goUp,
  navigateTo,
  getFolderImageCount,
  folderTree,
  resetToRoot
} = useFolderTree(images)

// 当图片列表变化时，重置到根目录
watch(() => images.value.length, (newLen, oldLen) => {
  if (newLen === 0 || (oldLen === 0 && newLen > 0)) {
    resetToRoot()
  }
})

// ============================================================
// 方法
// ============================================================

/**
 * 获取图片在全局列表中的索引
 */
function getImageGlobalIndex(image: ImageData): number {
  return images.value.findIndex(img => img.id === image.id)
}

/**
 * 获取状态指示器类型
 */
function getStatusType(image: ImageData): 'failed' | 'labeled' | 'processing' | null {
  if (image.translationFailed) return 'failed'
  if (image.isManuallyAnnotated) return 'labeled'
  if (image.translationStatus === 'processing') return 'processing'
  return null
}

/**
 * 点击缩略图
 */
function handleClick(index: number) {
  emit('select', index)
}

/**
 * 点击文件夹
 */
function handleFolderClick(folder: FolderNode) {
  enterFolder(folder.path)
}

/**
 * 点击面包屑
 */
function handleBreadcrumbClick(path: string) {
  navigateTo(path)
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
 * 获取缩略图项的额外类名
 */
function getThumbnailClasses(image: ImageData): string[] {
  const classes: string[] = []
  if (image.translationFailed) {
    classes.push('translation-failed')
  }
  if (image.isManuallyAnnotated) {
    classes.push('has-manual-labels')
  }
  return classes
}

/**
 * 获取缩略图的 title 提示文本
 */
function getThumbnailTitle(image: ImageData): string {
  if (image.translationFailed) return '翻译失败，点击可重试'
  if (image.isManuallyAnnotated) return '包含手动标注'
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
      
      <!-- 文件夹模式：面包屑导航 + 扁平列表 -->
      <template v-if="hasImages && useTreeMode && folderTree">
        <!-- 面包屑导航 -->
        <div class="breadcrumb-nav">
          <template v-for="(crumb, idx) in breadcrumbs" :key="crumb.path">
            <span 
              class="breadcrumb-item"
              :class="{ active: idx === breadcrumbs.length - 1 }"
              @click="idx < breadcrumbs.length - 1 && handleBreadcrumbClick(crumb.path)"
            >
              {{ idx === 0 ? '📁' : '' }}{{ crumb.name }}
            </span>
            <span v-if="idx < breadcrumbs.length - 1" class="breadcrumb-sep">/</span>
          </template>
        </div>

        <!-- 返回上级按钮 -->
        <div 
          v-if="currentFolderPath" 
          class="folder-back-btn"
          @click="goUp"
        >
          <span class="back-icon">⬅️</span>
          <span>返回上级</span>
        </div>

        <!-- 内容列表 -->
        <div ref="containerRef" class="folder-content-list">
          <!-- 子文件夹列表 -->
          <div
            v-for="subfolder in currentSubfolders"
            :key="subfolder.path"
            class="folder-item"
            @click="handleFolderClick(subfolder)"
          >
            <span class="folder-icon">📁</span>
            <div class="folder-info">
              <span class="folder-name" :title="subfolder.name">{{ subfolder.name }}</span>
              <span class="folder-count">({{ getFolderImageCount(subfolder) }})</span>
            </div>
          </div>
          
          <!-- 当前文件夹的图片 -->
          <div
            v-for="image in currentImages"
            :key="image.id"
            :ref="(el) => setThumbnailRef(el as HTMLElement | null, getImageGlobalIndex(image))"
            class="thumbnail-item"
            :class="[
              { active: getImageGlobalIndex(image) === currentIndex },
              ...getThumbnailClasses(image)
            ]"
            :title="getThumbnailTitle(image)"
            @click="handleClick(getImageGlobalIndex(image))"
          >
            <img 
              v-if="image.originalDataURL"
              :src="image.originalDataURL" 
              :alt="image.fileName"
              class="thumbnail-image"
            >
            <span v-if="getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
            <span v-else-if="getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
            <div v-if="getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
          </div>

          <!-- 空文件夹提示 -->
          <div 
            v-if="currentSubfolders.length === 0 && currentImages.length === 0" 
            class="empty-folder"
          >
            <p>此文件夹为空</p>
          </div>
        </div>
      </template>
      
      <!-- 扁平模式（保留原有完整代码） -->
      <ul 
        v-else-if="hasImages"
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
          <img 
            v-if="image.originalDataURL"
            :src="image.originalDataURL" 
            :alt="image.fileName"
            class="thumbnail-image"
          >
          <span 
            v-if="getStatusType(image) === 'failed'"
            class="translation-failed-indicator"
          >!</span>
          <span 
            v-else-if="getStatusType(image) === 'labeled'"
            class="labeled-indicator"
          >✏️</span>
          <div 
            v-if="getStatusType(image) === 'processing'"
            class="thumbnail-processing-indicator"
          >
            ⟳
          </div>
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
  width: 230px;
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
  margin-bottom: 15px;
  color: #2c3e50;
  font-size: 1.4em;
  text-align: center;
}

/* ===================================
   面包屑导航样式
   =================================== */

.breadcrumb-nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2px;
  padding: 8px 10px;
  background: #f8fafc;
  border-radius: 6px;
  margin-bottom: 10px;
  font-size: 12px;
  line-height: 1.4;
}

.breadcrumb-item {
  color: #3498db;
  cursor: pointer;
  word-break: break-word;
}

.breadcrumb-item:hover:not(.active) {
  text-decoration: underline;
}

.breadcrumb-item.active {
  color: #2c3e50;
  font-weight: 500;
  cursor: default;
}

.breadcrumb-sep {
  color: #94a3b8;
  margin: 0 2px;
}

/* ===================================
   返回上级按钮
   =================================== */

.folder-back-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: linear-gradient(135deg, #e8f4fd 0%, #d4e8f8 100%);
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 12px;
  font-size: 13px;
  color: #3498db;
  transition: all 0.2s;
}

.folder-back-btn:hover {
  background: linear-gradient(135deg, #d4e8f8 0%, #c0dcf0 100%);
  transform: translateX(-2px);
}

.back-icon {
  font-size: 14px;
}

/* ===================================
   文件夹内容列表
   =================================== */

.folder-content-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: calc(100vh - 280px);
  overflow-y: auto;
}

/* 文件夹项样式 */
.folder-item {
  position: relative;
  padding: 10px 12px;
  padding-left: 10px; /* 不需要为图标留空间了 */
  background: linear-gradient(135deg, #fff8e6 0%, #fff3d4 100%);
  border: 1px solid #f0d78c;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  flex-shrink: 0;
}

.folder-item:hover {
  background: linear-gradient(135deg, #fff3d4 0%, #ffe8b8 100%);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(240, 215, 140, 0.4);
}

/* 文件夹图标作为右上角角标 */
.folder-item .folder-icon {
  position: absolute;
  top: -4px;
  right: -4px;
  font-size: 14px;
  background: #fff;
  border-radius: 4px;
  padding: 1px 2px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  z-index: 1;
}

.folder-item .folder-info {
  width: 100%;
}

.folder-item .folder-name {
  font-size: 13px;
  font-weight: 500;
  color: #5a4a00;
  word-break: break-word;
  line-height: 1.4;
}

.folder-item .folder-count {
  font-size: 11px;
  color: #8a7a30;
  margin-left: 4px;
}

/* 空文件夹提示 */
.empty-folder {
  text-align: center;
  padding: 20px;
  color: #94a3b8;
  font-size: 13px;
}

/* ===================================
   缩略图列表（扁平模式）
   =================================== */

#thumbnail-sidebar ul#thumbnailList {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

/* ===================================
   缩略图项基础样式（两种模式通用）
   =================================== */

#thumbnail-sidebar .thumbnail-item,
.folder-content-list .thumbnail-item {
  margin-bottom: 0;
  cursor: pointer;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  padding: 5px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  flex-shrink: 0;
}

/* 激活和悬停状态 */
#thumbnail-sidebar .thumbnail-item.active,
#thumbnail-sidebar .thumbnail-item:hover,
.folder-content-list .thumbnail-item.active,
.folder-content-list .thumbnail-item:hover {
  border-color: #3498db;
  box-shadow: 0 0 8px rgba(52, 152, 219, 0.5);
  transform: translateY(-2px);
}

/* 激活状态的左上角圆点标记 */
#thumbnail-sidebar .thumbnail-item.active::before,
.folder-content-list .thumbnail-item.active::before {
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
#thumbnail-sidebar .thumbnail-item.active::after,
.folder-content-list .thumbnail-item.active::after {
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
#thumbnail-sidebar .thumbnail-image,
.folder-content-list .thumbnail-image {
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

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 20px;
  color: #94a3b8;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}
</style>
