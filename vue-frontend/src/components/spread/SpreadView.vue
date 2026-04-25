<script setup lang="ts">
/**
 * 拼页漫画视图组件
 * 实现左右并排展示两张图片的拼页阅读模式
 * 
 * 功能：
 * - 左右并排展示两张图片，零间隙拼页效果
 * - 支持单独封面模式（第一张单独显示）
 * - 统一的缩放控制
 * - 每张图片独立的编辑和原图预览功能
 * - 翻页导航
 */

import { ref, computed, watch, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { showToast } from '@/utils/toast'
import SpreadImagePair from './SpreadImagePair.vue'
import type { ImageData } from '@/types/image'

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 是否激活拼页视图 */
  isActive: boolean
  /** 初始拼页索引（用于从编辑返回时恢复位置） */
  initialSpreadIndex?: number
}>()

const emit = defineEmits<{
  /** 退出拼页视图 */
  (e: 'exit'): void
  /** 编辑指定图片，携带当前拼页索引用于恢复 */
  (e: 'edit', imageIndex: number, spreadIndex: number): void
}>()

// ============================================================
// Store 引用
// ============================================================

const imageStore = useImageStore()
const settingsStore = useSettingsStore()

const { images, imageCount } = storeToRefs(imageStore)

// ============================================================
// 状态定义
// ============================================================

/** 当前拼页索引（从0开始，表示当前显示的是第几对图片） */
const currentSpreadIndex = ref(0)

/** 是否启用单独封面模式 */
const isCoverMode = ref(false)

/** 图片缩放比例（50% - 200%） */
const imageScale = ref(100)

/** 是否显示缩略图面板 */
const showThumbnails = ref(false)

/** 组件是否已挂载 */
const isMounted = ref(false)

/** 翻页方向：'ltr' = 从左到右（西漫），'rtl' = 从右到左（日漫） */
const pageDirection = ref<'ltr' | 'rtl'>('ltr')

// ============================================================
// 计算属性
// ============================================================

/** 总拼页数 */
const totalSpreads = computed(() => {
  if (imageCount.value === 0) return 0
  
  if (isCoverMode.value) {
    // 封面模式：第1张单独显示，剩余图片两两组合
    const remainingImages = imageCount.value - 1
    if (remainingImages <= 0) return 1
    return 1 + Math.ceil(remainingImages / 2)
  } else {
    // 普通模式：所有图片两两组合
    return Math.ceil(imageCount.value / 2)
  }
})

/** 当前页码显示文本 */
const pageIndicatorText = computed(() => {
  if (imageCount.value === 0) return '0 / 0'
  
  const currentPage = currentSpreadIndex.value + 1
  const total = totalSpreads.value
  return `${currentPage} / ${total}`
})

/** 当前显示的图片对 */
const currentImagePair = computed(() => {
  const pair: { left: ImageData | null; right: ImageData | null; isSingle: boolean; direction: 'ltr' | 'rtl' } = {
    left: null,
    right: null,
    isSingle: false,
    direction: pageDirection.value
  }
  
  if (imageCount.value === 0) return pair
  
  if (isCoverMode.value && currentSpreadIndex.value === 0) {
    // 封面模式的第一页：只显示第一张图片
    pair.left = images.value[0] || null
    pair.isSingle = true
    return pair
  }
  
  // 计算实际图片索引
  let firstIndex: number
  let secondIndex: number
  
  if (isCoverMode.value) {
    // 封面模式：第1张单独显示，剩余从索引1开始两两组合
    firstIndex = 1 + (currentSpreadIndex.value - 1) * 2
    secondIndex = firstIndex + 1
  } else {
    // 普通模式：从索引0开始两两组合
    firstIndex = currentSpreadIndex.value * 2
    secondIndex = firstIndex + 1
  }
  
  // 根据翻页方向决定左右排布
  // LTR（从左到右）：先读左页，再读右页 -> 左页是first，右页是second
  // RTL（从右到左）：先读右页，再读左页 -> 右页是first，左页是second
  if (pageDirection.value === 'ltr') {
    pair.left = images.value[firstIndex] || null
    pair.right = images.value[secondIndex] || null
  } else {
    pair.right = images.value[firstIndex] || null
    pair.left = images.value[secondIndex] || null
  }
  
  pair.isSingle = !images.value[secondIndex]
  
  return pair
})

/** 当前图片对的索引 */
const currentPairIndices = computed(() => {
  if (imageCount.value === 0) return { left: -1, right: -1 }
  
  if (isCoverMode.value && currentSpreadIndex.value === 0) {
    return { left: 0, right: -1 }
  }
  
  let firstIndex: number
  let secondIndex: number
  
  if (isCoverMode.value) {
    firstIndex = 1 + (currentSpreadIndex.value - 1) * 2
    secondIndex = firstIndex + 1
  } else {
    firstIndex = currentSpreadIndex.value * 2
    secondIndex = firstIndex + 1
  }
  
  // 根据翻页方向决定左右索引
  if (pageDirection.value === 'ltr') {
    return {
      left: firstIndex < imageCount.value ? firstIndex : -1,
      right: secondIndex < imageCount.value ? secondIndex : -1
    }
  } else {
    return {
      right: firstIndex < imageCount.value ? firstIndex : -1,
      left: secondIndex < imageCount.value ? secondIndex : -1
    }
  }
})

/** 是否可以翻到上一页 */
const canGoPrevious = computed(() => currentSpreadIndex.value > 0)

/** 是否可以翻到下一页 */
const canGoNext = computed(() => currentSpreadIndex.value < totalSpreads.value - 1)

/** 图片缩放样式 */
const imageScaleStyle = computed(() => ({
  transform: `scale(${imageScale.value / 100})`,
  transformOrigin: 'center center'
}))

// ============================================================
// 方法
// ============================================================

/** 切换到上一页 */
function goToPrevious(): void {
  if (canGoPrevious.value) {
    currentSpreadIndex.value--
  }
}

/** 切换到下一页 */
function goToNext(): void {
  if (canGoNext.value) {
    currentSpreadIndex.value++
  }
}

/** 跳转到指定拼页 */
function goToSpread(index: number): void {
  if (index >= 0 && index < totalSpreads.value) {
    currentSpreadIndex.value = index
  }
}

/** 切换封面模式 */
function toggleCoverMode(): void {
  isCoverMode.value = !isCoverMode.value
  // 重置到第一页
  currentSpreadIndex.value = 0
  showToast(isCoverMode.value ? '已启用单独封面模式' : '已关闭单独封面模式', 'info')
}

/** 更新缩放比例 */
function updateScale(event: Event): void {
  const input = event.target as HTMLInputElement
  imageScale.value = parseInt(input.value, 10)
}

/** 放大 */
function zoomIn(): void {
  imageScale.value = Math.min(200, imageScale.value + 10)
}

/** 缩小 */
function zoomOut(): void {
  imageScale.value = Math.max(50, imageScale.value - 10)
}

/** 重置缩放 */
function resetZoom(): void {
  imageScale.value = 100
}

/** 切换缩略图显示 */
function toggleThumbnails(): void {
  showThumbnails.value = !showThumbnails.value
}

/** 切换翻页方向 */
function togglePageDirection(): void {
  pageDirection.value = pageDirection.value === 'ltr' ? 'rtl' : 'ltr'
  showToast(pageDirection.value === 'ltr' ? '翻页方向：从左到右' : '翻页方向：从右到左', 'info')
}

/** 处理编辑请求 */
function handleEdit(imageIndex: number): void {
  emit('edit', imageIndex, currentSpreadIndex.value)
}

/** 处理原图预览切换 */
function handleToggleOriginal(imageIndex: number, showOriginal: boolean): void {
  const image = images.value[imageIndex]
  if (image) {
    imageStore.updateImageByIndex(imageIndex, { showOriginal })
  }
}

/** 退出拼页视图 */
function exitSpreadView(): void {
  emit('exit')
}

/** 处理键盘导航 */
function handleKeyDown(event: KeyboardEvent): void {
  if (!props.isActive) return
  
  // 在输入框中不处理
  const target = event.target as HTMLElement
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
  
  switch (event.key) {
    case 'ArrowLeft':
    case 'a':
    case 'A':
      // 根据翻页方向决定左右键行为
      // LTR: 左键 = 上一页, 右键 = 下一页
      // RTL: 左键 = 下一页, 右键 = 上一页（符合日漫阅读习惯）
      if (pageDirection.value === 'ltr') {
        goToPrevious()
      } else {
        goToNext()
      }
      event.preventDefault()
      break
    case 'ArrowRight':
    case 'd':
    case 'D':
      if (pageDirection.value === 'ltr') {
        goToNext()
      } else {
        goToPrevious()
      }
      event.preventDefault()
      break
    case 'Escape':
      exitSpreadView()
      event.preventDefault()
      break
    case '+':
    case '=':
      zoomIn()
      event.preventDefault()
      break
    case '-':
      zoomOut()
      event.preventDefault()
      break
    case '0':
      resetZoom()
      event.preventDefault()
      break
  }
}

/** 获取缩略图对应的拼页索引 */
function getSpreadIndexForThumbnail(thumbIndex: number): number {
  if (isCoverMode.value) {
    if (thumbIndex === 0) return 0
    return 1 + Math.floor((thumbIndex - 1) / 2)
  }
  return Math.floor(thumbIndex / 2)
}

/** 判断缩略图是否属于当前拼页 */
function isThumbnailInCurrentSpread(thumbIndex: number): boolean {
  const spreadIndex = getSpreadIndexForThumbnail(thumbIndex)
  return spreadIndex === currentSpreadIndex.value
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  isMounted.value = true
  document.addEventListener('keydown', handleKeyDown)
  
  // 加载保存的封面模式设置
  try {
    const savedCoverMode = localStorage.getItem('spreadView_coverMode')
    if (savedCoverMode !== null) {
      isCoverMode.value = savedCoverMode === 'true'
    }
  } catch (e) {
    console.warn('加载封面模式设置失败:', e)
  }
  
  // 加载保存的翻页方向设置
  try {
    const savedDirection = localStorage.getItem('spreadView_pageDirection')
    if (savedDirection === 'ltr' || savedDirection === 'rtl') {
      pageDirection.value = savedDirection
    }
  } catch (e) {
    console.warn('加载翻页方向设置失败:', e)
  }
})

// 监听封面模式变化并保存
watch(isCoverMode, (newValue) => {
  try {
    localStorage.setItem('spreadView_coverMode', String(newValue))
  } catch (e) {
    console.warn('保存封面模式设置失败:', e)
  }
})

// 监听翻页方向变化并保存
watch(pageDirection, (newValue) => {
  try {
    localStorage.setItem('spreadView_pageDirection', newValue)
  } catch (e) {
    console.warn('保存翻页方向设置失败:', e)
  }
})

// 监听激活状态
watch(() => props.isActive, (active) => {
  if (active) {
    // 如果有指定的初始拼页索引（从编辑返回），使用它
    if (props.initialSpreadIndex !== undefined && props.initialSpreadIndex >= 0) {
      currentSpreadIndex.value = props.initialSpreadIndex
    } else {
      // 否则同步当前图片索引到拼页索引
      const currentIndex = imageStore.currentImageIndex
      if (currentIndex >= 0) {
        if (isCoverMode.value) {
          if (currentIndex === 0) {
            currentSpreadIndex.value = 0
          } else {
            currentSpreadIndex.value = 1 + Math.floor((currentIndex - 1) / 2)
          }
        } else {
          currentSpreadIndex.value = Math.floor(currentIndex / 2)
        }
      }
    }
  }
})

// 清理事件监听
watch(() => props.isActive, (active) => {
  if (!active) {
    document.removeEventListener('keydown', handleKeyDown)
  } else {
    document.addEventListener('keydown', handleKeyDown)
  }
})
</script>

<template>
  <div
    v-if="isActive"
    class="spread-view"
    tabindex="0"
  >
    <!-- 顶部工具栏 -->
    <div class="spread-toolbar">
      <div class="toolbar-left">
        <!-- 返回按钮 -->
        <button class="toolbar-btn back-btn" @click="exitSpreadView" title="退出拼页视图 (Esc)">
          <span class="icon">←</span>
          <span>返回</span>
        </button>
        
        <!-- 封面模式切换 -->
        <button 
          class="toolbar-btn cover-mode-btn" 
          :class="{ active: isCoverMode }"
          @click="toggleCoverMode"
          title="切换单独封面模式"
        >
          <span class="icon">📖</span>
          <span>{{ isCoverMode ? '封面模式开' : '封面模式关' }}</span>
        </button>
        
        <!-- 缩略图切换 -->
        <button 
          class="toolbar-btn thumbnail-btn" 
          :class="{ active: showThumbnails }"
          @click="toggleThumbnails"
          title="显示/隐藏缩略图"
        >
          <span class="icon">🖼️</span>
          <span>缩略图</span>
        </button>
      </div>
      
      <div class="toolbar-center">
        <!-- 翻页导航 -->
        <div class="page-navigator">
          <button 
            class="nav-btn prev-btn" 
            :disabled="!canGoPrevious"
            @click="goToPrevious"
            title="上一页 (← 或 A)"
          >
            <span class="icon">◀</span>
          </button>
          
          <span class="page-indicator">{{ pageIndicatorText }}</span>
          
          <button 
            class="nav-btn next-btn" 
            :disabled="!canGoNext"
            @click="goToNext"
            title="下一页 (→ 或 D)"
          >
            <span class="icon">▶</span>
          </button>
        </div>
      </div>
      
      <div class="toolbar-right">
        <!-- 翻页方向切换按钮 -->
        <button 
          class="direction-btn" 
          :class="pageDirection"
          @click="togglePageDirection"
          :title="pageDirection === 'ltr' ? '从左到右翻页（西漫）' : '从右到左翻页（日漫）'"
        >
          <span class="direction-icon">{{ pageDirection === 'ltr' ? '➡️' : '⬅️' }}</span>
          <span class="direction-text">{{ pageDirection === 'ltr' ? 'LTR' : 'RTL' }}</span>
        </button>
        
        <!-- 缩放控制 -->
        <div class="zoom-controls">
          <button class="zoom-btn" @click="zoomOut" title="缩小 (-)">−</button>
          <div class="zoom-slider-container">
            <input
              type="range"
              class="zoom-slider"
              min="50"
              max="200"
              :value="imageScale"
              @input="updateScale"
            />
            <span class="zoom-value">{{ imageScale }}%</span>
          </div>
          <button class="zoom-btn" @click="zoomIn" title="放大 (+)">+</button>
          <button class="zoom-btn reset-btn" @click="resetZoom" title="重置缩放 (0)">⟲</button>
        </div>
      </div>
    </div>
    
    <!-- 缩略图面板 -->
    <div v-if="showThumbnails" class="spread-thumbnails-panel">
      <div class="thumbnails-scroll">
        <div
          v-for="(image, index) in images"
          :key="image.id"
          class="spread-thumbnail-item"
          :class="{ 
            active: isThumbnailInCurrentSpread(index),
            'first-in-spread': isCoverMode ? index === 0 : index % 2 === 0
          }"
          @click="goToSpread(getSpreadIndexForThumbnail(index))"
        >
          <img :src="image.originalDataURL" :alt="`缩略图 ${index + 1}`" />
          <span class="thumb-index">{{ index + 1 }}</span>
        </div>
      </div>
    </div>
    
    <!-- 主内容区域 -->
    <div class="spread-content">
      <div class="spread-container" :style="imageScaleStyle">
        <SpreadImagePair
          :left-image="currentImagePair.left"
          :right-image="currentImagePair.right"
          :left-index="currentPairIndices.left"
          :right-index="currentPairIndices.right"
          :is-single="currentImagePair.isSingle"
          :scale="imageScale"
          :direction="pageDirection"
          @edit="handleEdit"
          @toggle-original="handleToggleOriginal"
        />
      </div>
    </div>
    
    <!-- 底部提示 -->
    <div class="spread-hints">
      <span class="hint">← → 或 A D : 翻页</span>
      <span class="hint">+ - 0 : 缩放</span>
      <span class="hint">Esc : 退出</span>
    </div>
  </div>
</template>

<style scoped>
/* ============ 拼页视图主容器 ============ */
.spread-view {
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  z-index: var(--z-overlay, 1000);
  overflow: hidden;
}

/* ============ 顶部工具栏 ============ */
.spread-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
  gap: 20px;
}

.toolbar-left,
.toolbar-center,
.toolbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.toolbar-center {
  flex: 1;
  justify-content: center;
}

.toolbar-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.toolbar-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-1px);
}

.toolbar-btn.active {
  background: rgba(102, 126, 234, 0.5);
}

.toolbar-btn .icon {
  font-size: 14px;
}

.back-btn {
  background: rgba(231, 76, 60, 0.2);
}

.back-btn:hover {
  background: rgba(231, 76, 60, 0.4);
}

/* ============ 翻页导航 ============ */
.page-navigator {
  display: flex;
  align-items: center;
  gap: 15px;
}

.nav-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 10px;
  background: rgba(102, 126, 234, 0.3);
  color: #fff;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
}

.nav-btn:hover:not(:disabled) {
  background: rgba(102, 126, 234, 0.5);
  transform: scale(1.05);
}

.nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.page-indicator {
  min-width: 80px;
  text-align: center;
  padding: 8px 16px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  font-family: var(--font-mono, monospace);
}

/* ============ 翻页方向按钮 ============ */
.direction-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  transition: all 0.2s ease;
  margin-right: 10px;
}

.direction-btn.ltr {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: #fff;
}

.direction-btn.rtl {
  background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
  color: #fff;
}

.direction-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.direction-icon {
  font-size: 14px;
}

.direction-text {
  font-size: 12px;
  letter-spacing: 0.5px;
}

/* ============ 缩放控制 ============ */
.zoom-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.zoom-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  font-size: 16px;
  font-weight: 600;
  transition: all 0.2s ease;
}

.zoom-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.zoom-slider-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.zoom-slider {
  width: 100px;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}

.zoom-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 14px;
  height: 14px;
  background: #667eea;
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.2s ease;
}

.zoom-slider::-webkit-slider-thumb:hover {
  transform: scale(1.2);
  box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
}

.zoom-value {
  min-width: 45px;
  color: #fff;
  font-size: 13px;
  font-weight: 500;
}

/* ============ 缩略图面板 ============ */
.spread-thumbnails-panel {
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding: 10px 15px;
  flex-shrink: 0;
}

.thumbnails-scroll {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 5px 0;
}

.thumbnails-scroll::-webkit-scrollbar {
  height: 6px;
}

.thumbnails-scroll::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.thumbnails-scroll::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}

.spread-thumbnail-item {
  flex-shrink: 0;
  width: 60px;
  height: 80px;
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.2s ease;
  position: relative;
}

.spread-thumbnail-item:hover {
  border-color: rgba(255, 255, 255, 0.5);
  transform: scale(1.05);
}

.spread-thumbnail-item.active {
  border-color: #667eea;
  box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
}

.spread-thumbnail-item.first-in-spread {
  margin-left: 12px;
}

.spread-thumbnail-item.first-in-spread::before {
  content: '';
  position: absolute;
  left: -10px;
  top: 50%;
  transform: translateY(-50%);
  width: 2px;
  height: 60%;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 1px;
}

.spread-thumbnail-item:first-child.first-in-spread::before,
.spread-thumbnail-item:nth-child(2).first-in-spread::before {
  display: none;
}

.spread-thumbnail-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb-index {
  position: absolute;
  bottom: 2px;
  right: 2px;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 3px;
}

/* ============ 主内容区域 ============ */
.spread-content {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  overflow: auto;
  background: #0d1b2a;
}

.spread-container {
  display: flex;
  align-items: center;
  justify-content: center;
  max-width: 100%;
  max-height: 100%;
  transition: transform 0.3s ease;
}

/* ============ 底部提示 ============ */
.spread-hints {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  padding: 10px 20px;
  background: rgba(0, 0, 0, 0.3);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.hint {
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
}

/* ============ 响应式适配 ============ */
@media (max-width: 1024px) {
  .spread-toolbar {
    flex-wrap: wrap;
    padding: 10px 15px;
  }
  
  .toolbar-left,
  .toolbar-right {
    flex: 1;
  }
  
  .toolbar-center {
    order: -1;
    width: 100%;
    margin-bottom: 10px;
  }
  
  .zoom-slider {
    width: 80px;
  }
}

@media (max-width: 768px) {
  .spread-toolbar {
    flex-direction: column;
    gap: 10px;
  }
  
  .toolbar-left,
  .toolbar-center,
  .toolbar-right {
    width: 100%;
    justify-content: center;
  }
  
  .toolbar-center {
    order: 0;
  }
  
  .spread-hints {
    display: none;
  }
}
</style>
