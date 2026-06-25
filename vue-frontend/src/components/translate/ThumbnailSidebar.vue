<script setup lang="ts">

/**
 * 右侧缩略图侧边栏组件
 * 显示图片概览列表，所在区域由页面 shell 编排
 *
 * 支持两种模式：
 * - 扁平模式：普通的图片列表
 * - 文件夹模式：面包屑导航 + 扁平列表（无缩进）
 */

import { ref, computed, watch, nextTick, onMounted, type ComponentPublicInstance } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useImageStore } from '@/stores/imageStore'
import { useFolderTree } from '@/composables/useFolderTree'
import { useThumbnailSelection } from '@/composables/useThumbnailSelection'
import type { FolderNode } from '@/types/folder'

type TemplateElementRef = Element | ComponentPublicInstance | null

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 组件是否可见（用于 v-show 状态监听） */
  isVisible?: boolean
}>()

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

/** 侧边栏容器引用（真正的滚动容器） */
const sidebarRef = ref<HTMLElement | null>(null)

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

const {
  getImageGlobalIndex,
  getStatusType,
  isTranslated,
  getThumbnailTitle,
} = useThumbnailSelection(images)

// 当图片列表变化时，重置到根目录
watch(() => images.value.length, (newLen, previousLen) => {
  if (newLen === 0 || (previousLen === 0 && newLen > 0)) {
    resetToRoot()
  }
})

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

    // 智能选择滚动容器：
    // - 文件夹模式：使用 containerRef (.folder-content-list)
    // - 扁平模式：使用 sidebarRef (aside)
    const scrollContainer = (useTreeMode.value && containerRef.value)
      ? containerRef.value
      : sidebarRef.value

    if (activeThumb && scrollContainer) {
      // 计算缩略图相对于滚动容器的位置
      const thumbRect = activeThumb.getBoundingClientRect()
      const containerRect = scrollContainer.getBoundingClientRect()

      // 计算需要滚动的距离，使缩略图居中显示
      const thumbCenter = thumbRect.top + thumbRect.height / 2
      const containerCenter = containerRect.top + containerRect.height / 2
      const scrollOffset = thumbCenter - containerCenter

      scrollContainer.scrollTo({
        top: scrollContainer.scrollTop + scrollOffset,
        behavior: 'smooth'
      })
    }
  })
}

/**
 * 设置缩略图引用
 */
function resolveTemplateElement(el: TemplateElementRef): HTMLElement | null {
  if (!el) return null
  if (el instanceof HTMLElement) return el

  const componentElement = el.$el
  return componentElement instanceof HTMLElement ? componentElement : null
}

function setThumbnailRef(el: TemplateElementRef, index: number) {
  thumbnailRefs.value[index] = resolveTemplateElement(el)
}

// 监听当前索引变化
watch(currentIndex, () => {
  scrollToActiveThumbnail()
})

// 监听可见性变化，当从隐藏变为显示时重新定位
watch(() => props.isVisible, (newVisible, previousVisible) => {
  if (newVisible && !previousVisible) {
    // 组件从隐藏变为显示，触发滚动定位
    scrollToActiveThumbnail()
  }
})

onMounted(() => {
  if (hasImages.value) {
    scrollToActiveThumbnail()
  }
})
</script>

<template>
  <aside ref="sidebarRef" id="thumbnail-sidebar" class="thumbnail-sidebar">
    <div class="ui-panel-card thumbnail-card">
      <h2>图片概览</h2>

      <!-- 文件夹模式：面包屑导航 + 扁平列表 -->
      <template v-if="hasImages && useTreeMode && folderTree">
        <!-- 面包屑导航 -->
        <div class="breadcrumb-nav">
          <template v-for="(crumb, idx) in breadcrumbs" :key="crumb.path">
            <span
              v-if="idx === breadcrumbs.length - 1"
              class="breadcrumb-item active"
              aria-current="page"
            >
              {{ idx === 0 ? '📁' : '' }}{{ crumb.name }}
            </span>
            <UiButton
              v-else
              variant="toolbar"
              class="breadcrumb-item"
              :aria-label="`打开${crumb.name}`"
              @click="handleBreadcrumbClick(crumb.path)"
            >
              {{ idx === 0 ? '📁' : '' }}{{ crumb.name }}
            </UiButton>
            <span v-if="idx < breadcrumbs.length - 1" class="breadcrumb-sep">/</span>
          </template>
        </div>

        <!-- 返回上级按钮 -->
        <UiButton
          v-if="currentFolderPath"
          variant="toolbar"
          class="folder-back-btn"
          @click="goUp"
        >
          <span class="back-icon">⬅️</span>
          <span>返回上级</span>
        </UiButton>

        <!-- 内容列表 -->
        <div ref="containerRef" class="folder-content-list">
          <!-- 子文件夹列表 -->
          <UiButton
            v-for="subfolder in currentSubfolders"
            :key="subfolder.path"
            variant="toolbar"
            class="folder-item"
            :aria-label="`打开文件夹 ${subfolder.name}`"
            @click="handleFolderClick(subfolder)"
          >
            <span class="folder-icon">📁</span>
            <div class="folder-info">
              <span class="folder-name" :title="subfolder.name">{{ subfolder.name }}</span>
              <span class="folder-count">({{ getFolderImageCount(subfolder) }})</span>
            </div>
          </UiButton>

          <!-- 当前文件夹的图片 -->
          <UiButton
            v-for="image in currentImages"
            :key="image.id"
            :ref="(el) => setThumbnailRef(el, getImageGlobalIndex(image))"
            variant="toolbar"
            class="thumbnail-sidebar__item"
            :class="{ active: getImageGlobalIndex(image) === currentIndex }"
            :title="getThumbnailTitle(image)"
            :aria-current="getImageGlobalIndex(image) === currentIndex ? 'true' : undefined"
            :aria-label="`选择图片 ${getImageGlobalIndex(image) + 1}: ${image.fileName}`"
            @click="handleClick(getImageGlobalIndex(image))"
          >
            <img
              v-if="image.originalDataURL"
              :src="image.originalDataURL"
              :alt="image.fileName"
              class="thumbnail-image"
            >
            <!-- 页码角标（左下角） -->
            <span class="page-number-indicator">{{ getImageGlobalIndex(image) + 1 }}</span>
            <!-- 已翻译角标（左上角） -->
            <span v-if="isTranslated(image)" class="translated-indicator">✓</span>
            <!-- 状态角标 -->
            <span v-if="getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
            <span v-else-if="getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
            <div v-if="getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
          </UiButton>

          <!-- 空文件夹提示 -->
          <div
            v-if="currentSubfolders.length === 0 && currentImages.length === 0"
            class="empty-folder"
          >
            <p>此文件夹为空</p>
          </div>
        </div>
      </template>

      <!-- 扁平模式：按当前图片列表顺序展示缩略图 -->
      <ul
        v-else-if="hasImages"
        ref="containerRef"
        id="thumbnailList"
        class="thumbnail-list"
      >
        <li
          v-for="(image, index) in images"
          :key="image.id"
          class="thumbnail-list__entry"
          :data-index="index"
        >
          <UiButton
            :ref="(el) => setThumbnailRef(el, index)"
            variant="toolbar"
            class="thumbnail-sidebar__item"
            :class="{ active: index === currentIndex }"
            :title="getThumbnailTitle(image)"
            :aria-current="index === currentIndex ? 'true' : undefined"
            :aria-label="`选择图片 ${index + 1}: ${image.fileName}`"
            @click="handleClick(index)"
          >
            <img
              v-if="image.originalDataURL"
              :src="image.originalDataURL"
              :alt="image.fileName"
              class="thumbnail-image"
            >
            <!-- 页码角标（左下角） -->
            <span class="page-number-indicator">{{ index + 1 }}</span>
            <!-- 已翻译角标（左上角） -->
            <span v-if="isTranslated(image)" class="translated-indicator">✓</span>
            <!-- 状态角标 -->
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
          </UiButton>
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
   缩略图侧边栏样式 - 缩略图列表
   =================================== */

.thumbnail-sidebar {
  /* owner tokens: thumbnail-sidebar */
  --thumbnail-sidebar-scrollbar-thumb: #cbd5e0;
  --thumbnail-sidebar-scrollbar-track: #f8fafc;
  --thumbnail-sidebar-title-divider: #f0f0f0;
  --thumbnail-sidebar-folder-border: #f0d78c;
  --thumbnail-sidebar-folder-hover-shadow: rgba(240, 215, 140, .4);
  --thumbnail-sidebar-folder-icon-shadow: rgba(0, 0, 0, .1);
  --thumbnail-sidebar-thumbnail-active-shadow: rgba(52, 152, 219, .5);
  --thumbnail-sidebar-badge-shadow: rgba(0, 0, 0, .3);
  --thumbnail-sidebar-folder-back-background-start: #e8f4fd;
  --thumbnail-sidebar-folder-back-background-end: #d4e8f8;
  --thumbnail-sidebar-folder-back-hover-background-end: #c0dcf0;
  --thumbnail-sidebar-folder-hover-background-end: #ffe8b8;
  --thumbnail-sidebar-processing-badge-background: rgba(53, 152, 219, .8);
  --thumbnail-sidebar-failed-badge-background: rgba(255, 0, 0, .8);
  --thumbnail-sidebar-labeled-badge-background: rgba(0, 123, 255, .8);
  --thumbnail-sidebar-page-number-badge-background: rgba(0, 0, 0, .6);
  --thumbnail-sidebar-translated-badge-background: rgba(34, 197, 94, .9);
  --thumbnail-sidebar-muted-text: #94a3b8;
  --thumbnail-sidebar-folder-name-text: #5a4a00;
  --thumbnail-sidebar-folder-count-text: #8a7a30;

  width: 100%;
  height: 100%;
  overflow-y: auto;
  padding: 20px;
  margin-left: 0;
  order: 1;
  scrollbar-width: thin;
  scrollbar-color: var(--thumbnail-sidebar-scrollbar-thumb) var(--thumbnail-sidebar-scrollbar-track);
}

.thumbnail-sidebar::-webkit-scrollbar {
  width: 8px;
}

.thumbnail-sidebar::-webkit-scrollbar-track {
  background: var(--color-surface-quiet);
  border-radius: 8px;
}

.thumbnail-sidebar::-webkit-scrollbar-thumb {
  background-color: var(--thumbnail-sidebar-scrollbar-thumb);
  border-radius: 8px;
  border: 2px solid var(--thumbnail-sidebar-scrollbar-track);
}

.thumbnail-sidebar .thumbnail-card {
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
  background-color: var(--color-surface-card);
  border-radius: 12px;
  box-shadow: 0 4px 12px var(--shadow-soft);
  padding: 25px;
  transition: box-shadow 0.2s;
}

.thumbnail-sidebar .thumbnail-card:hover {
  box-shadow: 0 6px 16px var(--shadow-medium);
}

.thumbnail-sidebar .thumbnail-card h2 {
  border-bottom: 2px solid var(--thumbnail-sidebar-title-divider);
  padding-bottom: 12px;
  margin-bottom: 15px;
  color: var(--color-text-heading);
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
  background: var(--color-surface-quiet);
  border-radius: 6px;
  margin-bottom: 10px;
  font-size: 12px;
  line-height: 1.4;
}

.breadcrumb-item {
  color: var(--color-text-link);
  cursor: pointer;
  font: inherit;
  word-break: break-word;
}

.breadcrumb-item:hover:not(.active) {
  text-decoration: underline;
}

.breadcrumb-item.active {
  color: var(--color-text-heading);
  font-weight: 500;
  cursor: default;
}

.breadcrumb-sep {
  color: var(--thumbnail-sidebar-muted-text);
  margin: 0 2px;
}

/* ===================================
   返回上级按钮
   =================================== */

.folder-back-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 8px 12px;
  border: 0;
  background: linear-gradient(135deg, var(--thumbnail-sidebar-folder-back-background-start) 0%, var(--thumbnail-sidebar-folder-back-background-end) 100%);
  border-radius: 8px;
  cursor: pointer;
  font: inherit;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--color-text-link);
  text-align: left;
  transition: all 0.2s;
}

.folder-back-btn:hover {
  background: linear-gradient(135deg, var(--thumbnail-sidebar-folder-back-background-end) 0%, var(--thumbnail-sidebar-folder-back-hover-background-end) 100%);
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
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
}

/* 文件夹项样式 */
.folder-item {
  position: relative;
  display: block;
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--thumbnail-sidebar-folder-border);
  background: linear-gradient(135deg, var(--color-status-warning-surface-soft) 0%, var(--color-status-warning-surface-raised) 100%);
  border-radius: 8px;
  cursor: pointer;
  font: inherit;
  text-align: left;
  transition: all 0.2s;
  flex-shrink: 0;
}

.folder-item:hover {
  background: linear-gradient(135deg, var(--color-status-warning-surface-raised) 0%, var(--thumbnail-sidebar-folder-hover-background-end) 100%);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px var(--thumbnail-sidebar-folder-hover-shadow);
}

/* 文件夹图标作为右上角角标 */
.folder-item .folder-icon {
  position: absolute;
  top: -4px;
  right: -4px;
  font-size: 14px;
  background: var(--color-surface-base);
  border-radius: 4px;
  padding: 1px 2px;
  box-shadow: 0 1px 3px var(--thumbnail-sidebar-folder-icon-shadow);
  z-index: var(--z-local);
}

.folder-item .folder-info {
  width: 100%;
}

.folder-item .folder-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--thumbnail-sidebar-folder-name-text);
  word-break: break-word;
  line-height: 1.4;
}

.folder-item .folder-count {
  font-size: 11px;
  color: var(--thumbnail-sidebar-folder-count-text);
  margin-left: 4px;
}

/* 空文件夹提示 */
.empty-folder {
  text-align: center;
  padding: 20px;
  color: var(--thumbnail-sidebar-muted-text);
  font-size: 13px;
}

/* ===================================
   缩略图列表（扁平模式）
   =================================== */

.thumbnail-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.thumbnail-list__entry {
  margin: 0;
}

/* ===================================
   缩略图项基础样式（两种模式通用）
   =================================== */

.thumbnail-sidebar .thumbnail-sidebar__item,
.folder-content-list .thumbnail-sidebar__item {
  display: block;
  width: 100%;
  margin-bottom: 0;
  cursor: pointer;
  border: 2px solid var(--color-border-muted);
  border-radius: 8px;
  padding: 5px;
  background: transparent;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  flex-shrink: 0;
}

/* 激活和悬停状态 */
.thumbnail-sidebar .thumbnail-sidebar__item.active,
.thumbnail-sidebar .thumbnail-sidebar__item:hover,
.folder-content-list .thumbnail-sidebar__item.active,
.folder-content-list .thumbnail-sidebar__item:hover {
  border-color: var(--color-border-accent);
  box-shadow: 0 0 8px var(--thumbnail-sidebar-thumbnail-active-shadow);
  transform: translateY(-2px);
}

/* 缩略图图片 */
.thumbnail-sidebar .thumbnail-image,
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
  background-color: var(--thumbnail-sidebar-processing-badge-background);
  color: var(--color-text-inverse);
  width: 15px;
  height: 15px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  z-index: var(--z-local-control);
  animation: pulse 1.5s infinite;
}

/* 翻译失败标记（右下角红色感叹号） */
.translation-failed-indicator {
  position: absolute;
  bottom: 3px;
  right: 3px;
  background-color: var(--thumbnail-sidebar-failed-badge-background);
  color: var(--color-text-inverse);
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  z-index: var(--z-local-toolbar-raised);
  box-shadow: 0 0 3px var(--thumbnail-sidebar-badge-shadow);
}

/* 手动标注指示器（右下角蓝色铅笔） */
.labeled-indicator {
  position: absolute;
  bottom: 3px;
  right: 3px;
  background-color: var(--thumbnail-sidebar-labeled-badge-background);
  color: var(--color-text-inverse);
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  z-index: var(--z-local-toolbar);
  box-shadow: 0 0 3px var(--thumbnail-sidebar-badge-shadow);
}

/* 页码角标（左下角灰色小角标） */
.page-number-indicator {
  position: absolute;
  bottom: 3px;
  left: 3px;
  background-color: var(--thumbnail-sidebar-page-number-badge-background);
  color: var(--color-text-inverse);
  min-width: 18px;
  height: 18px;
  padding: 0 4px;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 500;
  z-index: var(--z-local-raised);
  box-shadow: 0 1px 3px var(--thumbnail-sidebar-badge-shadow);
  font-family: var(--font-sans);
}

/* 已翻译角标（左上角绿色对勾） */
.translated-indicator {
  position: absolute;
  top: 3px;
  left: 3px;
  background-color: var(--thumbnail-sidebar-translated-badge-background);
  color: var(--color-text-inverse);
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  z-index: var(--z-local-control);
  box-shadow: 0 1px 3px var(--thumbnail-sidebar-badge-shadow);
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 20px;
  color: var(--thumbnail-sidebar-muted-text);
}

@media (--breakpoint-md-down) {
  .thumbnail-sidebar {
    order: 3;
    width: 100%;
    height: auto;
    max-height: none;
    overflow: visible;
    padding: 0;
  }

  .thumbnail-sidebar .thumbnail-card {
    padding: 16px;
  }

  .thumbnail-sidebar .thumbnail-card h2 {
    font-size: 1.2em;
  }

  .thumbnail-sidebar .thumbnail-image {
    width: min(100%, 290px);
    margin: 0 auto;
  }
}
</style>
