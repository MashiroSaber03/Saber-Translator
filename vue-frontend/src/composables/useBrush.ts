/**
 * 笔刷工具组合式函数
 * 管理笔刷模式的状态和操作，包括修复笔刷和还原笔刷
 * 对应原 edit_mode.js 中的笔刷功能
 */

import { ref, computed, onUnmounted } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { BRUSH_MIN_SIZE, BRUSH_MAX_SIZE, BRUSH_DEFAULT_SIZE } from '@/constants'
import { inpaintSingleBubble } from '@/api/translate'
import { showToast } from '@/utils/toast'
import type { BubbleCoords, InpaintMethod } from '@/types/bubble'
import { addErasureToUserMask, addRestorationToUserMask } from '@/utils/maskMerger'

// ============================================================
// 类型定义
// ============================================================

/** 笔刷模式类型 */
export type BrushMode = 'repair' | 'restore' | null

/** 笔刷位置 */
export interface BrushPosition {
  x: number
  y: number
  scale: number
}

/** 笔刷边界 */
export interface BrushBounds {
  x1: number
  y1: number
  x2: number
  y2: number
  path: BrushPosition[]
  radius: number
}

// ============================================================
// 类型定义 - 回调
// ============================================================

/** 当前修复设置 */
export interface CurrentRepairSettings {
  /** 修复方式 */
  inpaintMethod: InpaintMethod
  /** 填充颜色 */
  fillColor: string
}

export interface BrushCallbacks {
  /** 笔刷操作完成后触发重新渲染 */
  onBrushComplete?: () => void
  /** 【复刻原版】获取当前编辑面板的修复设置，对应原版 $('#bubbleInpaintMethodNew').val() 等 */
  getCurrentRepairSettings?: () => CurrentRepairSettings
}

// ============================================================
// 组合式函数
// ============================================================

export function useBrush(callbacks?: BrushCallbacks) {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  // ============================================================
  // 状态定义
  // ============================================================

  /** 当前笔刷模式 */
  const brushMode = ref<BrushMode>(null)

  /** 笔刷大小 */
  const brushSize = ref(BRUSH_DEFAULT_SIZE)

  /** 笔刷快捷键是否按下 */
  const isBrushKeyDown = ref(false)

  /** 是否正在涂抹 */
  const isBrushPainting = ref(false)

  /** 笔刷路径 */
  const brushPath = ref<BrushPosition[]>([])

  /** 鼠标位置 */
  const mouseX = ref(0)
  const mouseY = ref(0)

  /** 当前操作的视口元素 */
  let activeViewport: HTMLElement | null = null

  /** 笔刷临时画布 */
  let brushCanvas: HTMLCanvasElement | null = null
  let brushCtx: CanvasRenderingContext2D | null = null

  // ============================================================
  // 计算属性
  // ============================================================

  /** 笔刷是否激活 */
  const isActive = computed(() => brushMode.value !== null)

  /** 笔刷颜色配置 */
  const brushColor = computed(() => {
    if (brushMode.value === 'repair') {
      return {
        fill: 'rgba(76, 175, 80, 0.4)',
        border: '#4CAF50'
      }
    } else if (brushMode.value === 'restore') {
      return {
        fill: 'rgba(33, 150, 243, 0.4)',
        border: '#2196F3'
      }
    }
    return { fill: 'transparent', border: 'transparent' }
  })

  // ============================================================
  // 笔刷模式控制
  // ============================================================

  /**
   * 进入笔刷模式
   * @param mode - 笔刷模式: 'repair' 修复笔刷 | 'restore' 还原笔刷
   */
  function enterBrushMode(mode: 'repair' | 'restore'): void {
    if (brushMode.value === mode) return

    brushMode.value = mode
    isBrushKeyDown.value = true
    brushPath.value = []

    console.log(`进入${mode === 'repair' ? '修复' : '还原'}笔刷模式，笔刷大小: ${brushSize.value}px`)
  }

  /**
   * 退出笔刷模式
   */
  function exitBrushMode(): void {
    // 如果正在涂抹，先完成涂抹
    if (isBrushPainting.value) {
      finishBrushPainting()
    }

    // 重置所有笔刷相关变量
    const wasActive = brushMode.value !== null
    brushMode.value = null
    isBrushKeyDown.value = false
    isBrushPainting.value = false
    brushPath.value = []

    // 清理画布
    removeBrushCanvas()

    if (wasActive) {
      console.log('退出笔刷模式')
    }
  }

  /**
   * 切换笔刷模式
   */
  function toggleBrushMode(mode: 'repair' | 'restore'): void {
    if (brushMode.value === mode) {
      exitBrushMode()
    } else {
      enterBrushMode(mode)
    }
  }

  // ============================================================
  // 笔刷大小控制
  // ============================================================

  /**
   * 设置笔刷大小
   */
  function setBrushSize(size: number): void {
    brushSize.value = Math.max(BRUSH_MIN_SIZE, Math.min(BRUSH_MAX_SIZE, size))
  }

  /**
   * 调整笔刷大小
   * @param delta - 调整量（正数增大，负数减小）
   */
  function adjustBrushSize(delta: number): void {
    setBrushSize(brushSize.value + delta)
  }

  /**
   * 处理滚轮调整笔刷大小
   */
  function handleBrushWheel(event: WheelEvent): void {
    if (!isActive.value) return

    event.preventDefault()
    event.stopPropagation()

    const delta = event.deltaY > 0 ? -5 : 5
    adjustBrushSize(delta)
  }

  // ============================================================
  // 笔刷涂抹操作
  // ============================================================

  /**
   * 开始涂抹
   */
  function startBrushPainting(event: MouseEvent, viewport: HTMLElement): void {
    if (!isActive.value || event.button !== 0) return

    event.preventDefault()
    event.stopPropagation()

    isBrushPainting.value = true
    activeViewport = viewport
    brushPath.value = []

    // 获取相对于图片的坐标
    const pos = getBrushPositionInImage(event, viewport)
    if (pos) {
      brushPath.value.push(pos)
      // 创建临时画布
      createBrushCanvas(viewport)
      drawBrushStroke(pos)
    }

    console.log('开始笔刷涂抹')
  }

  /**
   * 继续涂抹
   */
  function continueBrushPainting(event: MouseEvent): void {
    // 更新鼠标位置（用于光标显示）
    mouseX.value = event.clientX
    mouseY.value = event.clientY

    if (!isBrushPainting.value || !isActive.value) return

    const pos = getBrushPositionInImage(event, activeViewport)
    if (pos) {
      brushPath.value.push(pos)
      drawBrushStroke(pos)
    }
  }

  /**
   * 结束涂抹
   */
  function finishBrushPainting(): void {
    if (!isBrushPainting.value) return

    isBrushPainting.value = false

    const currentImage = imageStore.currentImage
    if (!currentImage || brushPath.value.length === 0) {
      removeBrushCanvas()
      brushPath.value = []
      return
    }

    // 获取涂抹区域边界
    const bounds = getBrushPathBounds()
    if (!bounds) {
      removeBrushCanvas()
      brushPath.value = []
      return
    }

    const mode = brushMode.value

    // 执行笔刷操作（异步）
    const executeAndRender = async () => {
      if (mode === 'restore') {
        await restoreBrushArea(currentImage, bounds)
      } else if (mode === 'repair') {
        await repairBrushArea(currentImage, bounds)
      }

      // 触发重新渲染回调
      callbacks?.onBrushComplete?.()
    }

    executeAndRender()

    removeBrushCanvas()
    brushPath.value = []
  }

  // ============================================================
  // 坐标计算
  // ============================================================

  /**
   * 获取笔刷在图片中的位置
   */
  function getBrushPositionInImage(event: MouseEvent, viewport: HTMLElement | null): BrushPosition | null {
    if (!viewport) return null

    const wrapper = viewport.querySelector('.image-canvas-wrapper') as HTMLElement
    const img = wrapper?.querySelector('img') as HTMLImageElement

    if (!img || !img.naturalWidth) return null

    const rect = wrapper.getBoundingClientRect()
    const transform = window.getComputedStyle(wrapper).transform
    let scale = 1

    if (transform && transform !== 'none') {
      const matrix = new DOMMatrix(transform)
      scale = matrix.a
    }

    // 计算相对于图片的坐标
    const imgX = (event.clientX - rect.left) / scale
    const imgY = (event.clientY - rect.top) / scale

    // 确保坐标在图片范围内
    const imgWidth = img.naturalWidth
    const imgHeight = img.naturalHeight

    if (imgX < 0 || imgY < 0 || imgX > imgWidth || imgY > imgHeight) {
      return null
    }

    return { x: imgX, y: imgY, scale }
  }

  /**
   * 获取笔刷路径边界
   */
  function getBrushPathBounds(): BrushBounds | null {
    if (brushPath.value.length === 0) return null

    const firstPoint = brushPath.value[0]
    const scale = firstPoint?.scale || 1
    const radius = brushSize.value / 2 / scale

    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity

    for (const pos of brushPath.value) {
      minX = Math.min(minX, pos.x - radius)
      minY = Math.min(minY, pos.y - radius)
      maxX = Math.max(maxX, pos.x + radius)
      maxY = Math.max(maxY, pos.y + radius)
    }

    return {
      x1: Math.max(0, Math.floor(minX)),
      y1: Math.max(0, Math.floor(minY)),
      x2: Math.ceil(maxX),
      y2: Math.ceil(maxY),
      path: [...brushPath.value],
      radius
    }
  }

  // ============================================================
  // 画布操作
  // ============================================================

  /**
   * 创建笔刷临时画布
   */
  function createBrushCanvas(viewport: HTMLElement): void {
    // 移除旧画布
    removeBrushCanvas()

    const wrapper = viewport.querySelector('.image-canvas-wrapper') as HTMLElement
    const img = wrapper?.querySelector('img') as HTMLImageElement

    if (!img) return

    const canvas = document.createElement('canvas')
    canvas.id = 'brushOverlayCanvas'
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    canvas.style.cssText = 'position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 100;'

    wrapper.appendChild(canvas)
    brushCanvas = canvas
    brushCtx = canvas.getContext('2d')
  }

  /**
   * 移除笔刷画布
   */
  function removeBrushCanvas(): void {
    if (brushCanvas && brushCanvas.parentNode) {
      brushCanvas.parentNode.removeChild(brushCanvas)
    }
    brushCanvas = null
    brushCtx = null
  }

  /**
   * 绘制笔刷笔画
   */
  function drawBrushStroke(pos: BrushPosition): void {
    if (!brushCtx || !pos) return

    const color = brushColor.value.fill
    const radius = brushSize.value / 2 / (pos.scale || 1)

    brushCtx.beginPath()
    brushCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2)
    brushCtx.fillStyle = color
    brushCtx.fill()
  }

  // ============================================================
  // 笔刷效果应用
  // ============================================================

  /**
   * 还原笔刷区域（恢复为原图）
   */
  async function restoreBrushArea(currentImage: any, bounds: BrushBounds): Promise<void> {
    if (!currentImage.originalDataURL) return

    // 获取当前干净背景
    let cleanSrc: string
    if (currentImage.cleanImageData) {
      cleanSrc = 'data:image/png;base64,' + currentImage.cleanImageData
    } else {
      cleanSrc = currentImage.originalDataURL
    }

    return new Promise((resolve) => {
      const cleanImg = new Image()
      const originalImg = new Image()
      let loadedCount = 0

      const onLoad = async () => {
        loadedCount++
        if (loadedCount < 2) return

        const canvas = document.createElement('canvas')
        canvas.width = cleanImg.naturalWidth
        canvas.height = cleanImg.naturalHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          resolve()
          return
        }

        // 先绘制当前干净背景
        ctx.drawImage(cleanImg, 0, 0)

        // 创建笔刷蒙版
        const maskCanvas = document.createElement('canvas')
        maskCanvas.width = canvas.width
        maskCanvas.height = canvas.height
        const maskCtx = maskCanvas.getContext('2d')
        if (!maskCtx) {
          resolve()
          return
        }

        // 绘制笔刷路径作为蒙版
        maskCtx.fillStyle = 'white'
        for (const pos of bounds.path) {
          maskCtx.beginPath()
          maskCtx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2)
          maskCtx.fill()
        }

        // 使用蒙版从原图恢复
        ctx.globalCompositeOperation = 'destination-out'
        ctx.drawImage(maskCanvas, 0, 0)
        ctx.globalCompositeOperation = 'destination-over'
        ctx.drawImage(originalImg, 0, 0)
        ctx.globalCompositeOperation = 'source-over'

        // ✅ 更新 userMask（记录用户还原意图）
        const newUserMask = await addRestorationToUserMask(
          currentImage.userMask,
          canvas.width,
          canvas.height,
          bounds.path,
          bounds.radius
        )

        // 更新 cleanImageData 和 userMask
        const newCleanImageData = canvas.toDataURL('image/png').split(',')[1]
        imageStore.updateCurrentImage({
          cleanImageData: newCleanImageData,
          userMask: newUserMask
        })
        console.log('还原笔刷区域完成，userMask 已更新')
        resolve()
      }

      cleanImg.onload = onLoad
      cleanImg.onerror = () => resolve()
      originalImg.onload = onLoad
      originalImg.onerror = () => resolve()

      cleanImg.src = cleanSrc
      originalImg.src = currentImage.originalDataURL
    })
  }

  /**
   * 修复笔刷区域（使用填充色或LAMA）
   * 【复刻原版】从编辑面板下拉框读取修复方式，不依赖气泡选中状态
   */
  async function repairBrushArea(currentImage: any, bounds: BrushBounds): Promise<void> {
    // 【复刻原版】从编辑面板获取修复方式，对应原版 $('#bubbleInpaintMethodNew').val()
    // 通过回调获取，不依赖气泡选中状态
    const settings = callbacks?.getCurrentRepairSettings?.() || {
      inpaintMethod: 'solid' as InpaintMethod,
      fillColor: '#FFFFFF'
    }
    const inpaintMethod = settings.inpaintMethod

    // 判断是否使用 LAMA（lama_mpe 或 litelama）
    const isLamaMethod = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama'
    if (isLamaMethod) {
      // 使用 LAMA 修复
      await repairBrushAreaWithLama(currentImage, bounds, inpaintMethod)
    } else {
      // 使用纯色填充
      await repairBrushAreaWithColor(currentImage, bounds, settings.fillColor)
    }
  }

  /**
   * 使用 LAMA 修复笔刷区域
   * 【复刻原版】支持精确掩膜：根据用户的笔刷路径生成掩膜，而非使用外接矩形
   */
  async function repairBrushAreaWithLama(
    currentImage: any,
    bounds: BrushBounds,
    inpaintMethod: 'lama_mpe' | 'litelama' = 'lama_mpe'
  ): Promise<void> {
    // 获取当前干净背景或原图
    let baseImageData: string
    let baseImageSrc: string
    if (currentImage.cleanImageData) {
      baseImageData = currentImage.cleanImageData
      baseImageSrc = 'data:image/png;base64,' + currentImage.cleanImageData
    } else if (currentImage.originalDataURL) {
      baseImageData = currentImage.originalDataURL.split(',')[1]
      baseImageSrc = currentImage.originalDataURL
    } else {
      console.error('无法获取基础图像用于 LAMA 修复')
      return
    }

    // 【复刻原版】通过加载图像获取实际尺寸
    return new Promise((resolve) => {
      const img = new Image()
      img.onload = async () => {
        const imgWidth = img.naturalWidth
        const imgHeight = img.naturalHeight

        // 生成精确的笔刷掩膜（白色=需要修复的区域，黑色=保留的区域）
        const maskCanvas = document.createElement('canvas')
        maskCanvas.width = imgWidth
        maskCanvas.height = imgHeight
        const maskCtx = maskCanvas.getContext('2d')

        if (!maskCtx) {
          console.error('无法创建掩膜画布上下文')
          resolve()
          return
        }

        // 填充黑色背景（保留区域）
        maskCtx.fillStyle = 'black'
        maskCtx.fillRect(0, 0, imgWidth, imgHeight)

        // 用白色绘制笔刷路径（需要修复的区域）
        maskCtx.fillStyle = 'white'
        for (const pos of bounds.path) {
          maskCtx.beginPath()
          maskCtx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2)
          maskCtx.fill()
        }

        // 将掩膜转换为 base64
        const maskDataUrl = maskCanvas.toDataURL('image/png')
        const maskBase64 = maskDataUrl.split(',')[1]

        // 将笔刷路径边界转换为矩形坐标（用于边界检查）
        const coords: BubbleCoords = [bounds.x1, bounds.y1, bounds.x2, bounds.y2]

        // 确定 LAMA 模型类型
        const lamaModel = inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe'

        try {
          showToast('LAMA 修复中...', 'info')

          const response = await inpaintSingleBubble(baseImageData, coords, {
            method: 'lama',
            lamaModel: lamaModel,
            maskData: maskBase64
          })

          if (response.success && response.inpainted_image) {
            // ✅ 更新 userMask（记录用户消除意图）
            const newUserMask = await addErasureToUserMask(
              currentImage.userMask,
              imgWidth,
              imgHeight,
              bounds.path,
              bounds.radius
            )

            // 更新 cleanImageData 和 userMask
            imageStore.updateCurrentImage({
              cleanImageData: response.inpainted_image,
              userMask: newUserMask
            })
            console.log('修复笔刷区域完成（LAMA 修复，精确掩膜），userMask 已更新')
            showToast('LAMA 修复完成', 'success')
          } else {
            throw new Error(response.error || 'LAMA 修复返回无效数据')
          }
        } catch (error) {
          console.error('LAMA 修复失败，回退到纯色填充:', error)
          showToast('LAMA 修复失败，使用纯色填充', 'warning')
          // 【复刻原版】回退到纯色填充时，重新获取填充颜色
          const fallbackSettings = callbacks?.getCurrentRepairSettings?.()
          await repairBrushAreaWithColor(currentImage, bounds, fallbackSettings?.fillColor)
        }
        resolve()
      }
      img.onerror = () => {
        console.error('加载图像失败，无法进行 LAMA 修复')
        resolve()
      }
      img.src = baseImageSrc
    })
  }

  /**
   * 使用纯色填充修复笔刷区域
   * 【复刻原版】从编辑面板读取填充颜色，不依赖气泡选中状态
   */
  async function repairBrushAreaWithColor(currentImage: any, bounds: BrushBounds, fillColor?: string): Promise<void> {
    // 【复刻原版】使用传入的填充色，对应原版 $('#fillColorNew').val()
    const color = fillColor || settingsStore.settings.textStyle.fillColor || '#FFFFFF'

    // 获取当前干净背景
    let cleanSrc: string
    if (currentImage.cleanImageData) {
      cleanSrc = 'data:image/png;base64,' + currentImage.cleanImageData
    } else if (currentImage.originalDataURL) {
      cleanSrc = currentImage.originalDataURL
    } else {
      return
    }

    return new Promise((resolve) => {
      const img = new Image()
      img.onload = async () => {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) {
          resolve()
          return
        }

        // 绘制当前背景
        ctx.drawImage(img, 0, 0)

        // 用填充色绘制笔刷路径
        ctx.fillStyle = color
        for (const pos of bounds.path) {
          ctx.beginPath()
          ctx.arc(pos.x, pos.y, bounds.radius, 0, Math.PI * 2)
          ctx.fill()
        }


        // ✅ 更新 userMask（记录用户消除意图）
        const newUserMask = await addErasureToUserMask(
          currentImage.userMask,
          img.naturalWidth,
          img.naturalHeight,
          bounds.path,
          bounds.radius
        )

        // 更新 cleanImageData 和 userMask
        const newCleanImageData = canvas.toDataURL('image/png').split(',')[1]
        imageStore.updateCurrentImage({
          cleanImageData: newCleanImageData,
          userMask: newUserMask
        })
        console.log('修复笔刷区域完成（纯色填充），userMask 已更新')
        resolve()
      }
      img.onerror = () => resolve()
      img.src = cleanSrc
    })
  }

  // ============================================================
  // 干净背景管理
  // ============================================================

  /**
   * 确保有干净的背景图
   * 如果当前图片没有干净背景，则尝试生成（仅用于纯色填充模式）
   * @returns Promise<boolean> 是否成功确保干净背景存在
   */
  async function ensureCleanBackground(): Promise<boolean> {
    const currentImage = imageStore.currentImage

    // 如果已有干净背景，直接返回
    if (!currentImage) {
      console.warn('ensureCleanBackground: 没有当前图片')
      return false
    }

    if (currentImage.cleanImageData) {
      console.log('ensureCleanBackground: 已有干净背景')
      return true
    }

    // 【复刻原版】从回调获取当前设置，回退到settingsStore
    const settings = callbacks?.getCurrentRepairSettings?.()
    const inpaintMethod = settings?.inpaintMethod || settingsStore.settings.textStyle.inpaintMethod

    // 检查修复方式，只有纯色填充模式才需要创建临时干净背景
    if (inpaintMethod !== 'solid') {
      console.log('ensureCleanBackground: 非纯色填充模式，跳过')
      return false
    }

    // 检查是否有必要的数据
    if (!currentImage.bubbleStates || currentImage.bubbleStates.length === 0) {
      console.warn('ensureCleanBackground: 没有气泡数据')
      return false
    }

    if (!currentImage.translatedDataURL && !currentImage.originalDataURL) {
      console.warn('ensureCleanBackground: 没有图片数据')
      return false
    }

    console.log('ensureCleanBackground: 尝试为纯色填充模式创建临时干净背景')

    // 使用翻译后的图片或原图作为基础
    const baseSrc = currentImage.translatedDataURL || currentImage.originalDataURL
    const fillColor = settings?.fillColor || settingsStore.settings.textStyle.fillColor || '#FFFFFF'

    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          const ctx = canvas.getContext('2d')

          if (!ctx) {
            console.error('ensureCleanBackground: 无法获取 Canvas 上下文')
            resolve(false)
            return
          }

          // 绘制基础图片
          ctx.drawImage(img, 0, 0)

          // 用填充色覆盖所有气泡区域
          ctx.fillStyle = fillColor
          for (const bubble of currentImage.bubbleStates!) {
            const [x1, y1, x2, y2] = bubble.coords
            ctx.fillRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
          }

          // 提取 Base64 数据（不含前缀）
          const tempCleanImage = canvas.toDataURL('image/png').split(',')[1]

          // 更新图片数据
          imageStore.updateCurrentImage({ cleanImageData: tempCleanImage })

          console.log('ensureCleanBackground: 成功创建临时干净背景（纯色填充）')
          resolve(true)
        } catch (e) {
          console.error('ensureCleanBackground: Canvas 操作失败:', e)
          resolve(false)
        }
      }
      img.onerror = () => {
        console.error('ensureCleanBackground: 加载图像失败')
        resolve(false)
      }
      img.src = baseSrc
    })
  }

  // ============================================================
  // 清理
  // ============================================================

  onUnmounted(() => {
    exitBrushMode()
  })

  // ============================================================
  // 返回接口
  // ============================================================

  return {
    // 状态
    brushMode,
    brushSize,
    isBrushKeyDown,
    isBrushPainting,
    mouseX,
    mouseY,

    // 计算属性
    isActive,
    brushColor,

    // 常量
    BRUSH_MIN_SIZE,
    BRUSH_MAX_SIZE,

    // 笔刷模式控制
    enterBrushMode,
    exitBrushMode,
    toggleBrushMode,

    // 笔刷大小控制
    setBrushSize,
    adjustBrushSize,
    handleBrushWheel,

    // 笔刷涂抹操作
    startBrushPainting,
    continueBrushPainting,
    finishBrushPainting,

    // 坐标计算
    getBrushPositionInImage,
    getBrushPathBounds,

    // 干净背景管理
    ensureCleanBackground
  }
}
