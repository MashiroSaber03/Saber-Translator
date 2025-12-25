<script setup lang="ts">
/**
 * 图片预览模态框组件
 * 支持点击缩略图放大查看、键盘导航、缩放等功能
 * 可在漫画分析页面的多个位置复用
 */

import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'

// ============================================================
// Props 和 Events
// ============================================================

const props = defineProps<{
  /** 是否显示 */
  visible: boolean
  /** 当前页码 */
  pageNum: number
  /** 总页数 */
  totalPages: number
  /** 书籍ID（可选，默认使用 store 中的） */
  bookId?: string
}>()

const emit = defineEmits<{
  /** 关闭事件 */
  (e: 'close'): void
  /** 页码变更事件 */
  (e: 'change', pageNum: number): void
}>()

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 模态框根元素引用 */
const modalRef = ref<HTMLElement | null>(null)

/** 当前缩放比例 */
const scale = ref(1)

/** 最小缩放比例 */
const minScale = 0.5

/** 最大缩放比例 */
const maxScale = 3

/** 缩放步进 */
const scaleStep = 0.25

/** 图片位置偏移 */
const offset = ref({ x: 0, y: 0 })

/** 是否正在拖拽 */
const isDragging = ref(false)

/** 拖拽起始位置 */
const dragStart = ref({ x: 0, y: 0 })

/** 是否正在加载图片 */
const isLoading = ref(true)

/** 图片加载错误 */
const loadError = ref(false)

// ============================================================
// 计算属性
// ============================================================

/** 实际使用的书籍ID */
const actualBookId = computed(() => props.bookId || insightStore.currentBookId || '')

/** 图片URL */
const imageUrl = computed(() => {
  if (!actualBookId.value || !props.pageNum) return ''
  return insightApi.getPageImageUrl(actualBookId.value, props.pageNum)
})

/** 是否有上一页 */
const hasPrevPage = computed(() => props.pageNum > 1)

/** 是否有下一页 */
const hasNextPage = computed(() => props.pageNum < props.totalPages)

/** 缩放百分比显示 */
const scalePercent = computed(() => Math.round(scale.value * 100) + '%')

/** 图片样式 */
const imageStyle = computed(() => ({
  transform: `scale(${scale.value}) translate(${offset.value.x}px, ${offset.value.y}px)`,
  cursor: isDragging.value ? 'grabbing' : (scale.value > 1 ? 'grab' : 'default')
}))

// ============================================================
// 方法
// ============================================================

/**
 * 关闭模态框
 */
function close(): void {
  resetView()
  emit('close')
}

/**
 * 导航到上一页
 */
function navigatePrev(): void {
  if (hasPrevPage.value) {
    resetView()
    emit('change', props.pageNum - 1)
  }
}

/**
 * 导航到下一页
 */
function navigateNext(): void {
  if (hasNextPage.value) {
    resetView()
    emit('change', props.pageNum + 1)
  }
}

/**
 * 放大
 */
function zoomIn(): void {
  if (scale.value < maxScale) {
    scale.value = Math.min(scale.value + scaleStep, maxScale)
  }
}

/**
 * 缩小
 */
function zoomOut(): void {
  if (scale.value > minScale) {
    scale.value = Math.max(scale.value - scaleStep, minScale)
    // 缩小时重置偏移
    if (scale.value <= 1) {
      offset.value = { x: 0, y: 0 }
    }
  }
}

/**
 * 重置视图
 */
function resetView(): void {
  scale.value = 1
  offset.value = { x: 0, y: 0 }
}

/**
 * 处理滚轮缩放
 */
function handleWheel(event: WheelEvent): void {
  event.preventDefault()
  if (event.deltaY < 0) {
    zoomIn()
  } else {
    zoomOut()
  }
}

/**
 * 开始拖拽
 */
function startDrag(event: MouseEvent): void {
  if (scale.value <= 1) return
  isDragging.value = true
  dragStart.value = {
    x: event.clientX - offset.value.x,
    y: event.clientY - offset.value.y
  }
}

/**
 * 拖拽中
 */
function onDrag(event: MouseEvent): void {
  if (!isDragging.value) return
  offset.value = {
    x: event.clientX - dragStart.value.x,
    y: event.clientY - dragStart.value.y
  }
}

/**
 * 结束拖拽
 */
function endDrag(): void {
  isDragging.value = false
}

/**
 * 处理键盘事件
 */
function handleKeydown(event: KeyboardEvent): void {
  if (!props.visible) return
  
  switch (event.key) {
    case 'Escape':
      close()
      break
    case 'ArrowLeft':
      navigatePrev()
      break
    case 'ArrowRight':
      navigateNext()
      break
    case '+':
    case '=':
      zoomIn()
      break
    case '-':
      zoomOut()
      break
    case '0':
      resetView()
      break
  }
}

/**
 * 图片加载完成
 */
function onImageLoad(): void {
  isLoading.value = false
  loadError.value = false
}

/**
 * 图片加载失败
 */
function onImageError(): void {
  isLoading.value = false
  loadError.value = true
}

// ============================================================
// 生命周期
// ============================================================

// 监听显示状态
watch(() => props.visible, (visible) => {
  if (visible) {
    isLoading.value = true
    loadError.value = false
    nextTick(() => {
      modalRef.value?.focus()
    })
  }
})

// 监听页码变化
watch(() => props.pageNum, () => {
  isLoading.value = true
  loadError.value = false
})

onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <Teleport to="body">
    <div 
      v-if="visible"
      ref="modalRef"
      class="image-preview-modal"
      tabindex="0"
      @click="close"
      @wheel="handleWheel"
    >
      <!-- 关闭按钮 -->
      <button class="preview-close" title="关闭 (Esc)" @click.stop="close">
        &times;
      </button>
      
      <!-- 图片容器 -->
      <div 
        class="preview-image-container" 
        @click.stop
        @mousedown="startDrag"
        @mousemove="onDrag"
        @mouseup="endDrag"
        @mouseleave="endDrag"
      >
        <!-- 加载中 -->
        <div v-if="isLoading" class="preview-loading">
          <div class="loading-spinner"></div>
          <p>加载中...</p>
        </div>
        
        <!-- 加载失败 -->
        <div v-else-if="loadError" class="preview-error">
          <div class="error-icon">❌</div>
          <p>图片加载失败</p>
        </div>
        
        <!-- 图片 -->
        <img 
          v-show="!isLoading && !loadError"
          :src="imageUrl" 
          :alt="`第${pageNum}页`"
          :style="imageStyle"
          draggable="false"
          @load="onImageLoad"
          @error="onImageError"
        >
      </div>
      
      <!-- 底部工具栏 -->
      <div class="preview-toolbar" @click.stop>
        <!-- 导航按钮 -->
        <button 
          class="toolbar-btn nav-btn"
          :disabled="!hasPrevPage"
          title="上一页 (←)"
          @click="navigatePrev"
        >
          ◀
        </button>
        
        <!-- 页码信息 -->
        <span class="page-info">{{ pageNum }} / {{ totalPages }}</span>
        
        <!-- 导航按钮 -->
        <button 
          class="toolbar-btn nav-btn"
          :disabled="!hasNextPage"
          title="下一页 (→)"
          @click="navigateNext"
        >
          ▶
        </button>
        
        <!-- 分隔线 -->
        <div class="toolbar-divider"></div>
        
        <!-- 缩放控制 -->
        <button 
          class="toolbar-btn"
          :disabled="scale <= minScale"
          title="缩小 (-)"
          @click="zoomOut"
        >
          ➖
        </button>
        
        <span class="scale-info">{{ scalePercent }}</span>
        
        <button 
          class="toolbar-btn"
          :disabled="scale >= maxScale"
          title="放大 (+)"
          @click="zoomIn"
        >
          ➕
        </button>
        
        <button 
          class="toolbar-btn"
          title="重置 (0)"
          @click="resetView"
        >
          🔄
        </button>
      </div>
      
      <!-- 快捷键提示 -->
      <div class="preview-hints">
        <span>← → 切换页面</span>
        <span>+ - 缩放</span>
        <span>滚轮缩放</span>
        <span>Esc 关闭</span>
      </div>
    </div>
  </Teleport>
</template>


<style scoped>
/* 模态框背景 */
.image-preview-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.95);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  outline: none;
}

/* 关闭按钮 */
.preview-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 40px;
  height: 40px;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 24px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-close:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: scale(1.1);
}

/* 图片容器 */
.preview-image-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  overflow: hidden;
  padding: 60px 20px 100px;
}

.preview-image-container img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  transition: transform 0.1s ease-out;
  user-select: none;
}

/* 加载状态 */
.preview-loading,
.preview-error {
  text-align: center;
  color: white;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

/* 底部工具栏 */
.preview-toolbar {
  position: absolute;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(0, 0, 0, 0.7);
  padding: 10px 20px;
  border-radius: 30px;
  backdrop-filter: blur(10px);
}

.toolbar-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.toolbar-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.2);
}

.toolbar-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.nav-btn {
  font-size: 14px;
}

.page-info,
.scale-info {
  color: white;
  font-size: 14px;
  min-width: 60px;
  text-align: center;
}

.toolbar-divider {
  width: 1px;
  height: 24px;
  background: rgba(255, 255, 255, 0.2);
  margin: 0 4px;
}

/* 快捷键提示 */
.preview-hints {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 16px;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.5);
}

/* 响应式 */
@media (max-width: 768px) {
  .preview-toolbar {
    padding: 8px 12px;
    gap: 8px;
  }
  
  .toolbar-btn {
    width: 32px;
    height: 32px;
    font-size: 14px;
  }
  
  .page-info,
  .scale-info {
    font-size: 12px;
    min-width: 50px;
  }
  
  .preview-hints {
    display: none;
  }
}
</style>
