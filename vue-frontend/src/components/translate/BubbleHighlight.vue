<script setup lang="ts">
/**
 * 气泡高亮框组件
 * 在图片上显示气泡检测框，支持矩形和多边形坐标
 * 
 * 功能：
 * - 气泡高亮框渲染（updateBubbleHighlight）
 * - 支持多边形和矩形坐标
 * - 点击高亮框选中气泡
 * - 高亮框跟随图片缩放
 * - 支持气泡旋转角度显示
 */

import { computed } from 'vue'
import type { BubbleCoords, PolygonCoords } from '@/types/bubble'

// Props 定义
interface Props {
  /** 气泡坐标数组 */
  bubbleCoords: BubbleCoords[]
  /** 气泡多边形数组（可选） */
  bubblePolygons?: (PolygonCoords | null)[]
  /** 气泡旋转角度数组（可选，单位：度） */
  bubbleRotations?: number[]
  /** 当前选中的气泡索引 */
  selectedIndex?: number
  /** 多选的气泡索引数组（可选） */
  selectedIndices?: number[]
  /** 图片显示指标 */
  imageMetrics: ImageDisplayMetrics | null
  /** 是否显示高亮框 */
  visible?: boolean
  /** 是否显示气泡索引标签 */
  showIndex?: boolean
}

/** 图片显示指标接口 */
interface ImageDisplayMetrics {
  /** 图像内容在屏幕上的实际渲染宽度 */
  visualContentWidth: number
  /** 图像内容在屏幕上的实际渲染高度 */
  visualContentHeight: number
  /** 图像内容左上角相对于容器的X轴偏移 */
  visualContentOffsetX: number
  /** 图像内容左上角相对于容器的Y轴偏移 */
  visualContentOffsetY: number
  /** 水平缩放比例 */
  scaleX: number
  /** 垂直缩放比例 */
  scaleY: number
  /** 图像的原始宽度 */
  naturalWidth: number
  /** 图像的原始高度 */
  naturalHeight: number
}

const props = withDefaults(defineProps<Props>(), {
  bubblePolygons: () => [],
  bubbleRotations: () => [],
  selectedIndex: -1,
  selectedIndices: () => [],
  visible: true,
  showIndex: true
})

// Emits 定义
const emit = defineEmits<{
  /** 点击气泡高亮框 */
  (e: 'select', index: number): void
}>()

// ============================================================
// 计算属性
// ============================================================

/** 是否有多边形数据 */
const hasPolygons = computed(() => {
  return props.bubblePolygons && props.bubblePolygons.some(p => p !== null)
})

// ============================================================
// 方法
// ============================================================

/**
 * 获取矩形高亮框样式
 * @param coords - 气泡坐标 [x1, y1, x2, y2]
 * @param index - 气泡索引，用于获取旋转角度
 * @returns CSS 样式对象
 */
function getHighlightStyle(coords: BubbleCoords, index: number): Record<string, string> {
  if (!props.imageMetrics) return { display: 'none' }

  const [x1, y1, x2, y2] = coords
  const metrics = props.imageMetrics
  const rotation = props.bubbleRotations[index] || 0
  
  const width = (x2 - x1) * metrics.scaleX
  const height = (y2 - y1) * metrics.scaleY
  const left = metrics.visualContentOffsetX + x1 * metrics.scaleX
  const top = metrics.visualContentOffsetY + y1 * metrics.scaleY

  const style: Record<string, string> = {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`
  }
  
  // 如果有旋转角度，添加 transform
  if (rotation !== 0) {
    style.transformOrigin = 'center center'
    style.transform = `rotate(${rotation}deg)`
  }
  
  return style
}

/**
 * 获取多边形高亮框的 SVG 路径
 * @param polygon - 多边形坐标数组 [[x1, y1], [x2, y2], ...]
 * @returns SVG 路径字符串
 */
function getPolygonPath(polygon: PolygonCoords): string {
  if (!polygon || polygon.length < 3 || !props.imageMetrics) return ''
  
  const metrics = props.imageMetrics
  const points = polygon.map(([x, y]) => {
    const screenX = metrics.visualContentOffsetX + x * metrics.scaleX
    const screenY = metrics.visualContentOffsetY + y * metrics.scaleY
    return `${screenX},${screenY}`
  })
  
  return `M ${points.join(' L ')} Z`
}

/**
 * 检查是否应该显示多边形（而不是矩形）
 * @param index - 气泡索引
 * @returns 是否有多边形数据
 */
function hasPolygonAt(index: number): boolean {
  return !!(props.bubblePolygons && props.bubblePolygons[index])
}

/**
 * 获取指定索引的多边形
 * @param index - 气泡索引
 * @returns 多边形坐标或 null
 */
function getPolygonAt(index: number): PolygonCoords | null {
  return props.bubblePolygons?.[index] || null
}

/**
 * 点击高亮框处理
 * @param index - 气泡索引
 * @param event - 鼠标事件
 */
function handleClick(index: number, event: MouseEvent): void {
  event.preventDefault()
  event.stopPropagation()
  emit('select', index)
}

/**
 * 判断气泡是否被选中（单选或多选）
 * @param index - 气泡索引
 * @returns 是否选中
 */
function isSelected(index: number): boolean {
  // 检查单选
  if (index === props.selectedIndex) return true
  // 检查多选
  if (props.selectedIndices && props.selectedIndices.includes(index)) return true
  return false
}
</script>

<template>
  <div v-if="visible && imageMetrics" class="bubble-highlight-container">
    <!-- 矩形高亮框 -->
    <template v-for="(coords, index) in bubbleCoords" :key="`rect-${index}`">
      <div
        v-if="!hasPolygonAt(index)"
        class="highlight-bubble"
        :class="{ selected: isSelected(index) }"
        :style="getHighlightStyle(coords, index)"
        :data-bubble-index="index"
        :data-rotation="bubbleRotations[index] || 0"
        @click="handleClick(index, $event)"
      >
        <span v-if="showIndex" class="bubble-index">#{{ index + 1 }}</span>
      </div>
    </template>
    
    <!-- 多边形高亮框（SVG） -->
    <svg 
      v-if="hasPolygons"
      class="polygon-overlay"
    >
      <template v-for="(polygon, index) in bubblePolygons" :key="`polygon-${index}`">
        <g v-if="polygon" @click="handleClick(index, $event)">
          <path
            :d="getPolygonPath(polygon)"
            class="polygon-highlight"
            :class="{ selected: isSelected(index) }"
            :data-bubble-index="index"
          />
          <!-- 多边形中心显示索引 -->
          <text
            v-if="showIndex && polygon.length > 0"
            :x="imageMetrics.visualContentOffsetX + (polygon.reduce((sum, p) => sum + p[0], 0) / polygon.length) * imageMetrics.scaleX"
            :y="imageMetrics.visualContentOffsetY + (polygon.reduce((sum, p) => sum + p[1], 0) / polygon.length) * imageMetrics.scaleY"
            class="polygon-index"
            text-anchor="middle"
            dominant-baseline="middle"
          >
            #{{ index + 1 }}
          </text>
        </g>
      </template>
    </svg>
  </div>
</template>

<style scoped>
/* 高亮框容器 */
.bubble-highlight-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* 矩形气泡高亮框 */
.highlight-bubble {
  position: absolute;
  border: 2px solid var(--primary-color, #4a90d9);
  background: rgba(74, 144, 217, 0.1);
  cursor: pointer;
  transition: all 0.2s;
  box-sizing: border-box;
  pointer-events: auto;
}

.highlight-bubble:hover {
  border-color: var(--primary-hover, #3a7bc8);
  background: rgba(74, 144, 217, 0.2);
}

.highlight-bubble.selected {
  border-color: var(--success-color, #27ae60);
  background: rgba(39, 174, 96, 0.2);
  border-width: 3px;
}

/* 气泡索引标签 */
.bubble-index {
  position: absolute;
  top: -8px;
  left: -8px;
  width: 20px;
  height: 20px;
  background: var(--primary-color, #4a90d9);
  color: white;
  font-size: 11px;
  font-weight: bold;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}

.highlight-bubble.selected .bubble-index {
  background: var(--success-color, #27ae60);
}

/* 多边形 SVG 覆盖层 */
.polygon-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: visible;
}

/* 多边形高亮框 */
.polygon-highlight {
  fill: rgba(74, 144, 217, 0.1);
  stroke: var(--primary-color, #4a90d9);
  stroke-width: 2;
  cursor: pointer;
  pointer-events: auto;
  transition: all 0.2s;
}

.polygon-highlight:hover {
  fill: rgba(74, 144, 217, 0.2);
  stroke: var(--primary-hover, #3a7bc8);
}

.polygon-highlight.selected {
  fill: rgba(39, 174, 96, 0.2);
  stroke: var(--success-color, #27ae60);
  stroke-width: 3;
}

/* 多边形索引文字 */
.polygon-index {
  fill: var(--primary-color, #4a90d9);
  font-size: 14px;
  font-weight: bold;
  pointer-events: none;
}

.polygon-highlight.selected + .polygon-index,
g:has(.polygon-highlight.selected) .polygon-index {
  fill: var(--success-color, #27ae60);
}
</style>
