<template>
  <div
    ref="viewportRef"
    class="image-viewer"
    :class="{ 'is-dragging': isDragging }"
    @wheel.prevent="handleWheel"
    @mousedown="handleMouseDown"
    @mousemove="handleMouseMove"
    @mouseup="handleMouseUp"
    @mouseleave="handleMouseUp"
    @dblclick="handleDoubleClick"
  >
    <div ref="contentRef" class="image-viewer-content" :style="transformStyle">
      <slot>
        <img
          v-if="src"
          ref="imageRef"
          :src="src"
          :alt="alt"
          class="image-viewer-image"
          @load="handleImageLoad"
        />
      </slot>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'

/**
 * 图片查看器组件
 * 支持缩放、平移、双击重置等功能
 */

// Props 定义
interface Props {
  /** 图片地址 */
  src?: string
  /** 图片描述 */
  alt?: string
  /** 最小缩放比例 */
  minScale?: number
  /** 最大缩放比例 */
  maxScale?: number
  /** 缩放速度 */
  zoomSpeed?: number
  /** 初始缩放比例 */
  initialScale?: number
  /** 是否启用缩放 */
  enableZoom?: boolean
  /** 是否启用平移 */
  enablePan?: boolean
  /** 是否启用键盘控制 */
  enableKeyboard?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  src: '',
  alt: '',
  minScale: 0.1,
  maxScale: 5,
  zoomSpeed: 0.1,
  initialScale: 1,
  enableZoom: true,
  enablePan: true,
  enableKeyboard: true
})

// Emits 定义
const emit = defineEmits<{
  /** 缩放比例变化 */
  scaleChange: [scale: number]
  /** 变换变化（缩放和平移） */
  transformChange: [transform: { scale: number; translateX: number; translateY: number }]
  /** 图片加载完成 */
  imageLoad: [event: Event]
}>()

// 模板引用
const viewportRef = ref<HTMLElement | null>(null)
const contentRef = ref<HTMLElement | null>(null)
const imageRef = ref<HTMLImageElement | null>(null)

// 状态
const scale = ref(props.initialScale)
const translateX = ref(0)
const translateY = ref(0)
const isDragging = ref(false)
const lastX = ref(0)
const lastY = ref(0)

// 计算变换样式
const transformStyle = computed(() => ({
  transform: `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`,
  transformOrigin: '0 0'
}))

/**
 * 在指定点缩放
 */
const zoomAt = (x: number, y: number, factor: number): void => {
  if (!props.enableZoom) return

  const newScale = Math.min(Math.max(scale.value * factor, props.minScale), props.maxScale)
  const scaleChange = newScale / scale.value

  // 以指定位置为中心缩放
  translateX.value = x - (x - translateX.value) * scaleChange
  translateY.value = y - (y - translateY.value) * scaleChange
  scale.value = newScale

  emitChanges()
}

/**
 * 以视口中心缩放
 */
const zoom = (factor: number): void => {
  if (!viewportRef.value) return
  const rect = viewportRef.value.getBoundingClientRect()
  zoomAt(rect.width / 2, rect.height / 2, factor)
}

/**
 * 设置缩放比例
 */
const setScale = (newScale: number): void => {
  if (!viewportRef.value) return
  const rect = viewportRef.value.getBoundingClientRect()
  const centerX = rect.width / 2
  const centerY = rect.height / 2
  const factor = newScale / scale.value
  zoomAt(centerX, centerY, factor)
}

/**
 * 重置变换
 */
const reset = (): void => {
  scale.value = props.initialScale
  translateX.value = 0
  translateY.value = 0
  emitChanges()
}

/**
 * 适应视口大小
 */
const fitToViewport = (): void => {
  if (!viewportRef.value || !imageRef.value) return

  const img = imageRef.value
  if (!img.naturalWidth || !img.naturalHeight) return

  const viewportRect = viewportRef.value.getBoundingClientRect()
  const scaleX = viewportRect.width / img.naturalWidth
  const scaleY = viewportRect.height / img.naturalHeight
  scale.value = Math.min(scaleX, scaleY) * 0.95 // 留5%边距

  // 居中
  translateX.value = (viewportRect.width - img.naturalWidth * scale.value) / 2
  translateY.value = (viewportRect.height - img.naturalHeight * scale.value) / 2

  emitChanges()
}

/**
 * 发送变化事件
 */
const emitChanges = (): void => {
  emit('scaleChange', scale.value)
  emit('transformChange', {
    scale: scale.value,
    translateX: translateX.value,
    translateY: translateY.value
  })
}

/**
 * 处理滚轮事件
 */
const handleWheel = (e: WheelEvent): void => {
  if (!props.enableZoom || !viewportRef.value) return

  const rect = viewportRef.value.getBoundingClientRect()
  const mouseX = e.clientX - rect.left
  const mouseY = e.clientY - rect.top

  const delta = e.deltaY > 0 ? 1 - props.zoomSpeed : 1 + props.zoomSpeed
  zoomAt(mouseX, mouseY, delta)
}

/**
 * 处理鼠标按下事件
 */
const handleMouseDown = (e: MouseEvent): void => {
  if (!props.enablePan || e.button !== 0) return

  isDragging.value = true
  lastX.value = e.clientX
  lastY.value = e.clientY
}

/**
 * 处理鼠标移动事件
 */
const handleMouseMove = (e: MouseEvent): void => {
  if (!isDragging.value || !props.enablePan) return

  const deltaX = e.clientX - lastX.value
  const deltaY = e.clientY - lastY.value

  translateX.value += deltaX
  translateY.value += deltaY

  lastX.value = e.clientX
  lastY.value = e.clientY

  emitChanges()
}

/**
 * 处理鼠标释放事件
 */
const handleMouseUp = (): void => {
  isDragging.value = false
}

/**
 * 处理双击事件
 */
const handleDoubleClick = (): void => {
  fitToViewport()
}

/**
 * 处理图片加载完成
 */
const handleImageLoad = (e: Event): void => {
  emit('imageLoad', e)
  // 图片加载完成后自动适应视口
  fitToViewport()
}

/**
 * 处理键盘事件
 */
const handleKeydown = (e: KeyboardEvent): void => {
  if (!props.enableKeyboard) return

  const step = 50
  let handled = true

  switch (e.key) {
    case 'ArrowUp':
      translateY.value += step
      break
    case 'ArrowDown':
      translateY.value -= step
      break
    case 'ArrowLeft':
      translateX.value += step
      break
    case 'ArrowRight':
      translateX.value -= step
      break
    case '+':
    case '=':
      zoom(1.2)
      return
    case '-':
      zoom(0.8)
      return
    case '0':
      reset()
      return
    default:
      handled = false
  }

  if (handled) {
    e.preventDefault()
    emitChanges()
  }
}

// 监听 src 变化
watch(
  () => props.src,
  () => {
    // 图片变化时重置变换
    reset()
  }
)

// 生命周期
onMounted(() => {
  if (props.enableKeyboard) {
    document.addEventListener('keydown', handleKeydown)
  }
})

onUnmounted(() => {
  if (props.enableKeyboard) {
    document.removeEventListener('keydown', handleKeydown)
  }
})

// 暴露方法供外部使用
defineExpose({
  scale,
  translateX,
  translateY,
  zoom,
  zoomAt,
  setScale,
  reset,
  fitToViewport
})
</script>

<style scoped>
.image-viewer {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--bg-secondary, #f5f5f5);
  cursor: grab;
}

.image-viewer.is-dragging {
  cursor: grabbing;
}

.image-viewer-content {
  position: absolute;
  top: 0;
  left: 0;
  will-change: transform;
}

.image-viewer-image {
  display: block;
  max-width: none;
  user-select: none;
  -webkit-user-drag: none;
}
</style>
