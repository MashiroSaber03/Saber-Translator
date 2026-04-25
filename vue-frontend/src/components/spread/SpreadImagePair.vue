<script setup lang="ts">
/**
 * 拼页图片对组件
 * 展示左右两张图片，支持独立的编辑和原图预览功能
 */

import { computed } from 'vue'
import type { ImageData } from '@/types/image'

// ============================================================
// Props 和 Emits
// ============================================================

const props = defineProps<{
  /** 左侧图片 */
  leftImage: ImageData | null
  /** 右侧图片 */
  rightImage: ImageData | null
  /** 左侧图片索引 */
  leftIndex: number
  /** 右侧图片索引 */
  rightIndex: number
  /** 是否为单张图片模式（封面模式的第一页） */
  isSingle: boolean
  /** 缩放比例 */
  scale: number
  /** 翻页方向：'ltr' = 从左到右，'rtl' = 从右到左 */
  direction: 'ltr' | 'rtl'
}>()

const emit = defineEmits<{
  /** 编辑图片 */
  (e: 'edit', imageIndex: number): void
  /** 切换原图预览 */
  (e: 'toggle-original', imageIndex: number, showOriginal: boolean): void
}>()

// ============================================================
// 计算属性
// ============================================================

/** 左侧图片显示URL */
const leftImageUrl = computed(() => {
  if (!props.leftImage) return ''
  if (props.leftImage.showOriginal || !props.leftImage.translatedDataURL) {
    return props.leftImage.originalDataURL
  }
  return props.leftImage.translatedDataURL
})

/** 右侧图片显示URL */
const rightImageUrl = computed(() => {
  if (!props.rightImage) return ''
  if (props.rightImage.showOriginal || !props.rightImage.translatedDataURL) {
    return props.rightImage.originalDataURL
  }
  return props.rightImage.translatedDataURL
})

/** 左侧图片是否显示原图 */
const leftShowOriginal = computed(() => props.leftImage?.showOriginal ?? false)

/** 右侧图片是否显示原图 */
const rightShowOriginal = computed(() => props.rightImage?.showOriginal ?? false)

/** 左侧图片是否有翻译结果 */
const leftHasTranslated = computed(() => !!props.leftImage?.translatedDataURL)

/** 右侧图片是否有翻译结果 */
const rightHasTranslated = computed(() => !!props.rightImage?.translatedDataURL)

/** 左侧图片文件名 */
const leftFileName = computed(() => props.leftImage?.fileName || '')

/** 右侧图片文件名 */
const rightFileName = computed(() => props.rightImage?.fileName || '')

/** 容器样式 */
const containerClass = computed(() => ({
  'image-pair-container': true,
  'single-mode': props.isSingle,
  'double-mode': !props.isSingle,
  'direction-ltr': props.direction === 'ltr',
  'direction-rtl': props.direction === 'rtl'
}))

// ============================================================
// 方法
// ============================================================

/** 处理编辑 */
function handleEdit(side: 'left' | 'right'): void {
  const index = side === 'left' ? props.leftIndex : props.rightIndex
  if (index >= 0) {
    emit('edit', index)
  }
}

/** 处理原图预览切换 */
function handleToggleOriginal(side: 'left' | 'right'): void {
  const index = side === 'left' ? props.leftIndex : props.rightIndex
  const image = side === 'left' ? props.leftImage : props.rightImage
  
  if (index >= 0 && image) {
    const newValue = !image.showOriginal
    emit('toggle-original', index, newValue)
  }
}

/** 处理图片加载错误 */
function handleImageError(side: 'left' | 'right'): void {
  console.error(`拼页视图: ${side} 侧图片加载失败`)
}
</script>

<template>
  <div :class="containerClass">
    <!-- 左侧图片 -->
    <div 
      v-if="leftImage" 
      class="spread-image-wrapper left-image"
      :class="{ 'single-image': isSingle }"
    >
      <div class="image-header">
        <!-- 状态标签放在左侧 -->
        <div class="header-status">
          <span v-if="leftShowOriginal" class="status-badge original">
            原图预览
          </span>
          <span 
            v-else-if="leftImage?.translationStatus === 'completed'" 
            class="status-badge translated"
          >
            已翻译
          </span>
          <span 
            v-else-if="leftImage?.translationStatus === 'processing'" 
            class="status-badge processing"
          >
            翻译中...
          </span>
          <span v-else class="status-badge pending">待翻译</span>
        </div>
        <!-- 文件名放在右侧 -->
        <span class="image-name" :title="leftFileName">{{ leftFileName }}</span>
      </div>
      
      <div class="image-content">
        <img
          :src="leftImageUrl"
          :alt="leftFileName"
          class="spread-image"
          @error="handleImageError('left')"
        />
      </div>
      
      <div class="image-actions">
        <button 
          class="action-btn edit-btn"
          @click="handleEdit('left')"
          title="编辑此图片"
        >
          <span class="btn-icon">✏️</span>
          <span class="btn-text">编辑</span>
        </button>
        
        <button 
          v-if="leftHasTranslated"
          class="action-btn toggle-btn"
          :class="{ 'showing-original': leftShowOriginal }"
          @click="handleToggleOriginal('left')"
          :title="leftShowOriginal ? '显示翻译图' : '显示原图'"
        >
          <span class="btn-icon">{{ leftShowOriginal ? '📄' : '🖼️' }}</span>
          <span class="btn-text">{{ leftShowOriginal ? '翻译图' : '原图' }}</span>
        </button>
      </div>
    </div>
    
    <!-- 空状态占位（左侧无图片时） -->
    <div v-else class="spread-image-wrapper empty-wrapper">
      <div class="empty-content">
        <span class="empty-icon">📷</span>
        <span class="empty-text">无图片</span>
      </div>
    </div>
    

    
    <!-- 右侧图片 -->
    <div 
      v-if="rightImage && !isSingle" 
      class="spread-image-wrapper right-image"
    >
      <div class="image-header">
        <!-- 状态标签放在左侧 -->
        <div class="header-status">
          <span v-if="rightShowOriginal" class="status-badge original">
            原图预览
          </span>
          <span 
            v-else-if="rightImage?.translationStatus === 'completed'" 
            class="status-badge translated"
          >
            已翻译
          </span>
          <span 
            v-else-if="rightImage?.translationStatus === 'processing'" 
            class="status-badge processing"
          >
            翻译中...
          </span>
          <span v-else class="status-badge pending">待翻译</span>
        </div>
        <!-- 文件名放在右侧 -->
        <span class="image-name" :title="rightFileName">{{ rightFileName }}</span>
      </div>
      
      <div class="image-content">
        <img
          :src="rightImageUrl"
          :alt="rightFileName"
          class="spread-image"
          @error="handleImageError('right')"
        />
      </div>
      
      <div class="image-actions">
        <button 
          class="action-btn edit-btn"
          @click="handleEdit('right')"
          title="编辑此图片"
        >
          <span class="btn-icon">✏️</span>
          <span class="btn-text">编辑</span>
        </button>
        
        <button 
          v-if="rightHasTranslated"
          class="action-btn toggle-btn"
          :class="{ 'showing-original': rightShowOriginal }"
          @click="handleToggleOriginal('right')"
          :title="rightShowOriginal ? '显示翻译图' : '显示原图'"
        >
          <span class="btn-icon">{{ rightShowOriginal ? '📄' : '🖼️' }}</span>
          <span class="btn-text">{{ rightShowOriginal ? '翻译图' : '原图' }}</span>
        </button>
      </div>
    </div>
    
    <!-- 空状态占位（右侧无图片时，双页模式显示） -->
    <div v-else-if="!isSingle" class="spread-image-wrapper empty-wrapper">
      <div class="empty-content">
        <span class="empty-icon">📷</span>
        <span class="empty-text">无图片</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ============ 图片对容器 ============ */
.image-pair-container {
  display: flex;
  align-items: stretch;
  justify-content: center;
  gap: 0;
  max-width: 100%;
  max-height: 100%;
  background: #0d1b2a;
}

.image-pair-container.double-mode {
  /* 双页模式：图片紧密排列 */
}

.image-pair-container.single-mode {
  /* 单页模式：图片居中显示 */
  max-width: 50%;
}

/* ============ 图片包装器 ============ */
.spread-image-wrapper {
  display: flex;
  flex-direction: column;
  background: #16213e;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  flex: 1;
  max-width: 50%;
  min-width: 300px;
}

.spread-image-wrapper.single-image {
  max-width: 100%;
}

.spread-image-wrapper.left-image {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
  border-right: none;
}

.spread-image-wrapper.right-image {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
  border-left: none;
}

/* ============ 图片头部 ============ */
.image-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 15px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  gap: 10px;
}

/* 状态标签容器 - 左侧 */
.header-status {
  flex-shrink: 0;
  min-width: 70px;
}

/* 状态徽章 */
.status-badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.status-badge.original {
  background: rgba(231, 76, 60, 0.9);
  color: #fff;
}

.status-badge.translated {
  background: rgba(39, 174, 96, 0.9);
  color: #fff;
}

.status-badge.processing {
  background: rgba(243, 156, 18, 0.9);
  color: #fff;
  animation: pulse 1.5s ease-in-out infinite;
}

.status-badge.pending {
  background: rgba(149, 165, 166, 0.6);
  color: #fff;
}

/* 文件名 - 右侧 */
.image-name {
  flex: 1;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding-left: 10px;
}

/* ============ 图片内容区域 ============ */
.image-content {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  background: #0d1b2a;
  overflow: hidden;
  min-height: 300px;
}

.spread-image {
  max-width: 100%;
  max-height: calc(100vh - 280px);
  object-fit: contain;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

/* ============ 图片操作按钮 ============ */
.image-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 12px 15px;
  background: rgba(0, 0, 0, 0.3);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.edit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
}

.edit-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.toggle-btn {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.toggle-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.toggle-btn.showing-original {
  background: rgba(231, 76, 60, 0.3);
  border-color: rgba(231, 76, 60, 0.5);
}

.btn-icon {
  font-size: 14px;
}

.btn-text {
  font-size: 12px;
}

/* ============ 空状态 ============ */
.empty-wrapper {
  background: rgba(0, 0, 0, 0.2);
  border: 2px dashed rgba(255, 255, 255, 0.1);
}

.empty-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px;
}

.empty-icon {
  font-size: 48px;
  opacity: 0.3;
}

.empty-text {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.4);
}

/* ============ 响应式适配 ============ */
@media (max-width: 1200px) {
  .spread-image-wrapper {
    min-width: 250px;
  }
  
  .spread-image {
    max-height: calc(100vh - 260px);
  }
}

@media (max-width: 900px) {
  .image-pair-container.double-mode {
    flex-direction: column;
    gap: 15px;
  }
  
  .spread-image-wrapper {
    max-width: 100%;
    min-width: auto;
    border-radius: 8px !important;
  }
  
  .book-spine {
    display: none;
  }
  
  .spread-image {
    max-height: calc(50vh - 150px);
  }
  
  .image-name {
    max-width: 200px;
  }
}

@media (max-width: 600px) {
  .action-btn {
    padding: 6px 12px;
  }
  
  .btn-text {
    display: none;
  }
  
  .image-header {
    padding: 8px 12px;
  }
  
  .image-actions {
    padding: 10px 12px;
  }
}
</style>
