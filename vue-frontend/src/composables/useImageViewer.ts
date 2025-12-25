/**
 * 图片查看器组合式函数
 * 提供缩放、平移、视口同步等功能
 * 对应原 image_viewer.js 中的 ImageViewer 类
 */

import { ref, computed } from 'vue'

// ============================================================
// 类型定义
// ============================================================

/** 变换状态 */
export interface TransformState {
  scale: number
  translateX: number
  translateY: number
}

/** 图片查看器配置 */
export interface ImageViewerOptions {
  /** 最小缩放比例 */
  minScale?: number
  /** 最大缩放比例 */
  maxScale?: number
  /** 缩放速度 */
  zoomSpeed?: number
  /** 缩放变化回调 */
  onScaleChange?: (scale: number) => void
  /** 变换变化回调 */
  onTransformChange?: (transform: TransformState) => void
}

// ============================================================
// 默认配置
// ============================================================

const DEFAULT_OPTIONS: Required<Omit<ImageViewerOptions, 'onScaleChange' | 'onTransformChange'>> = {
  minScale: 0.1,
  maxScale: 5,
  zoomSpeed: 0.1
}

// ============================================================
// 组合式函数
// ============================================================

/**
 * 图片查看器组合式函数
 * @param options - 配置选项
 */
export function useImageViewer(options: ImageViewerOptions = {}) {
  const config = { ...DEFAULT_OPTIONS, ...options }

  // ============================================================
  // 状态
  // ============================================================

  /** 缩放比例 */
  const scale = ref(1)

  /** X 轴平移量 */
  const translateX = ref(0)

  /** Y 轴平移量 */
  const translateY = ref(0)


  /** 是否正在拖拽 */
  const isDragging = ref(false)

  /** 上一次鼠标位置 */
  const lastX = ref(0)
  const lastY = ref(0)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 变换样式 */
  const transformStyle = computed(() => ({
    transform: `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`
  }))

  // ============================================================
  // 缩放方法
  // ============================================================

  /**
   * 在指定点缩放
   * @param x - 鼠标 X 坐标（相对于视口）
   * @param y - 鼠标 Y 坐标（相对于视口）
   * @param factor - 缩放因子
   */
  function zoomAt(x: number, y: number, factor: number): void {
    const newScale = Math.min(
      Math.max(scale.value * factor, config.minScale),
      config.maxScale
    )
    const scaleChange = newScale / scale.value

    // 以鼠标位置为中心缩放
    translateX.value = x - (x - translateX.value) * scaleChange
    translateY.value = y - (y - translateY.value) * scaleChange
    scale.value = newScale

    // 触发回调
    options.onScaleChange?.(scale.value)
    options.onTransformChange?.(getTransform())
  }

  /**
   * 以视口中心缩放
   * @param factor - 缩放因子
   * @param viewportWidth - 视口宽度
   * @param viewportHeight - 视口高度
   */
  function zoom(factor: number, viewportWidth = 800, viewportHeight = 600): void {
    zoomAt(viewportWidth / 2, viewportHeight / 2, factor)
  }

  /**
   * 放大
   */
  function zoomIn(): void {
    zoom(1.2)
  }

  /**
   * 缩小
   */
  function zoomOut(): void {
    zoom(0.8)
  }

  /**
   * 设置缩放比例
   * @param newScale - 新的缩放比例
   * @param viewportWidth - 视口宽度
   * @param viewportHeight - 视口高度
   */
  function setScale(newScale: number, viewportWidth = 800, viewportHeight = 600): void {
    const factor = newScale / scale.value
    zoomAt(viewportWidth / 2, viewportHeight / 2, factor)
  }

  // ============================================================
  // 平移方法
  // ============================================================

  /**
   * 开始拖拽
   * @param x - 鼠标 X 坐标
   * @param y - 鼠标 Y 坐标
   */
  function startDrag(x: number, y: number): void {
    isDragging.value = true
    lastX.value = x
    lastY.value = y
  }

  /**
   * 拖拽移动
   * @param x - 鼠标 X 坐标
   * @param y - 鼠标 Y 坐标
   */
  function drag(x: number, y: number): void {
    if (!isDragging.value) return

    const dx = x - lastX.value
    const dy = y - lastY.value
    translateX.value += dx
    translateY.value += dy
    lastX.value = x
    lastY.value = y

    options.onTransformChange?.(getTransform())
  }

  /**
   * 结束拖拽
   */
  function endDrag(): void {
    isDragging.value = false
  }

  /**
   * 平移
   * @param dx - X 方向偏移
   * @param dy - Y 方向偏移
   */
  function pan(dx: number, dy: number): void {
    translateX.value += dx
    translateY.value += dy
    options.onTransformChange?.(getTransform())
  }


  // ============================================================
  // 重置方法
  // ============================================================

  /**
   * 重置变换
   */
  function reset(): void {
    scale.value = 1
    translateX.value = 0
    translateY.value = 0
    options.onScaleChange?.(scale.value)
    options.onTransformChange?.(getTransform())
  }

  /**
   * 重置缩放（别名）
   */
  function resetZoom(): void {
    reset()
  }

  /**
   * 重置变换（用于切换图片前）
   */
  function resetTransform(): void {
    scale.value = 1
    translateX.value = 0
    translateY.value = 0
  }

  /**
   * 适应屏幕
   * @param imageWidth - 图片宽度
   * @param imageHeight - 图片高度
   * @param viewportWidth - 视口宽度
   * @param viewportHeight - 视口高度
   */
  function fitToScreen(
    imageWidth: number,
    imageHeight: number,
    viewportWidth: number,
    viewportHeight: number
  ): void {
    if (!imageWidth || !imageHeight) return

    const scaleX = viewportWidth / imageWidth
    const scaleY = viewportHeight / imageHeight
    scale.value = Math.min(scaleX, scaleY) * 0.95 // 留5%边距

    // 居中
    translateX.value = (viewportWidth - imageWidth * scale.value) / 2
    translateY.value = (viewportHeight - imageHeight * scale.value) / 2

    options.onScaleChange?.(scale.value)
    options.onTransformChange?.(getTransform())
  }

  // ============================================================
  // 状态获取/设置方法
  // ============================================================

  /**
   * 获取当前变换状态
   */
  function getTransform(): TransformState {
    return {
      scale: scale.value,
      translateX: translateX.value,
      translateY: translateY.value
    }
  }

  /**
   * 设置变换状态
   * @param transform - 变换状态
   */
  function setTransform(transform: Partial<TransformState>): void {
    if (transform.scale !== undefined) {
      scale.value = transform.scale
    }
    if (transform.translateX !== undefined) {
      translateX.value = transform.translateX
    }
    if (transform.translateY !== undefined) {
      translateY.value = transform.translateY
    }
    options.onTransformChange?.(getTransform())
  }

  /**
   * 同步另一个查看器的状态
   * @param otherTransform - 另一个查看器的变换状态
   */
  function syncWith(otherTransform: TransformState): void {
    scale.value = otherTransform.scale
    translateX.value = otherTransform.translateX
    translateY.value = otherTransform.translateY
  }

  // ============================================================
  // 滚动到指定位置
  // ============================================================

  /**
   * 滚动到指定气泡位置
   * @param bubbleCoords - 气泡坐标 [x1, y1, x2, y2]
   * @param viewportWidth - 视口宽度
   * @param viewportHeight - 视口高度
   */
  function scrollToBubble(
    bubbleCoords: [number, number, number, number],
    viewportWidth: number,
    viewportHeight: number
  ): void {
    if (!bubbleCoords || bubbleCoords.length < 4) return

    const [x1, y1, x2, y2] = bubbleCoords
    const bubbleCenterX = (x1 + x2) / 2
    const bubbleCenterY = (y1 + y2) / 2

    const viewportCenterX = viewportWidth / 2
    const viewportCenterY = viewportHeight / 2

    // 计算需要的平移量，使气泡居中
    translateX.value = viewportCenterX - bubbleCenterX * scale.value
    translateY.value = viewportCenterY - bubbleCenterY * scale.value

    options.onTransformChange?.(getTransform())
  }

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 状态
    scale,
    translateX,
    translateY,
    isDragging,
    transformStyle,

    // 缩放方法
    zoomAt,
    zoom,
    zoomIn,
    zoomOut,
    setScale,

    // 平移方法
    startDrag,
    drag,
    endDrag,
    pan,

    // 重置方法
    reset,
    resetZoom,
    resetTransform,
    fitToScreen,

    // 状态获取/设置
    getTransform,
    setTransform,
    syncWith,

    // 滚动
    scrollToBubble
  }
}
