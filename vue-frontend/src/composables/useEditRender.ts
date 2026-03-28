/**
 * 编辑模式渲染组合式函数
 * 处理编辑模式下的图像重新渲染逻辑
 * 对应原版 edit_mode.js 中的 reRenderFullImage 函数
 */

import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { getEffectiveDirection } from '@/types/bubble'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { reRenderImage } from '@/api/translate'
import { DEFAULT_FONT_FAMILY } from '@/constants'
import { getPureBase64FromImageSource } from '@/utils/imageBase64'

// ============================================================
// 类型定义
// ============================================================

export interface EditRenderCallbacks {
  /** 渲染开始 */
  onRenderStart?: () => void
  /** 渲染成功 */
  onRenderSuccess?: (translatedDataURL: string) => void
  /** 渲染失败 */
  onRenderError?: (error: string) => void
  /** 渲染结束（无论成功失败） */
  onRenderEnd?: () => void
}

// ============================================================
// 组合式函数
// ============================================================

export function useEditRender(callbacks?: EditRenderCallbacks) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()

  const { bubbles } = storeToRefs(bubbleStore)
  const { currentImage } = storeToRefs(imageStore)

  // ============================================================
  // 状态
  // ============================================================

  /** 是否正在渲染 */
  const isRendering = ref(false)

  /** 渲染错误信息 */
  const renderError = ref('')

  /** 每张图片各自的渲染 token（避免切图后把结果写到错误图片上） */
  const renderTokenByImageId: Record<string, symbol> = {}
  let activeRenderCount = 0

  // ============================================================
  // 辅助函数
  // ============================================================

  /**
   * 获取干净背景图像的Base64数据
   */
  async function getCleanImageBase64(): Promise<string | null> {
    const image = currentImage.value
    if (!image) return null

    // 优先使用cleanImageData
    if (image.cleanImageData) {
      return await getPureBase64FromImageSource(image.cleanImageData)
    }

    return await getPureBase64FromImageSource(image.originalDataURL)
  }

  // ============================================================
  // 主要功能
  // ============================================================

  /**
   * 重新渲染整个图像
   * @param silentMode 是否静默模式（不触发回调）
   * @returns Promise<boolean> 是否成功
   */
  async function reRenderFullImage(silentMode = false): Promise<boolean> {
    const image = currentImage.value
    if (!image) {
      console.warn('reRenderFullImage: 没有当前图片')
      return false
    }
    const imageId = image.id

    // 检查是否有气泡
    // 【复刻原版逻辑】没有气泡坐标时跳过后端渲染
    // 【Vue适配】将cleanImageData作为translatedDataURL显示，确保修复笔刷效果可见
    if (bubbles.value.length === 0) {
      console.log('reRenderFullImage: 没有气泡，跳过后端渲染')

      // 将cleanImageData作为翻译图显示（修复笔刷场景）
      const cleanBase64 = await getCleanImageBase64()
      if (cleanBase64) {
        const translatedDataURL = `data:image/png;base64,${cleanBase64}`
        image.translatedDataURL = translatedDataURL
        if (!silentMode) callbacks?.onRenderSuccess?.(translatedDataURL)
      }
      return true
    }

    // 获取必要的图像数据
    const cleanBase64 = await getCleanImageBase64()

    if (!cleanBase64) {
      console.error('reRenderFullImage: 缺少图像数据')
      renderError.value = '缺少图像数据'
      if (!silentMode) callbacks?.onRenderError?.('缺少图像数据')
      return false
    }

    // 创建新的渲染token（按图片维度）
    const renderToken = Symbol('render')
    renderTokenByImageId[imageId] = renderToken

    // 设置渲染状态
    activeRenderCount += 1
    isRendering.value = activeRenderCount > 0
    renderError.value = ''
    if (!silentMode) callbacks?.onRenderStart?.()

    try {
      console.log('reRenderFullImage: 开始重新渲染，气泡数量:', bubbles.value.length)

      // 构建后端API需要的参数格式（参考原版edit_mode.js）
      const bubbleStates = bubbles.value
      const bubbleTexts = bubbleStates.map(s => s.translatedText || '')
      // 【复刻原版】确保坐标为整数（后端PIL的paste方法需要整数坐标）
      const bubbleCoords = bubbleStates.map(s => s.coords.map(c => Math.round(c)) as [number, number, number, number])

      // 构建每个气泡的样式状态（确保数值类型正确）
      const bubbleStatesForApi = bubbleStates.map((s) => ({
        translatedText: s.translatedText || '',
        coords: s.coords,  // 后端会用 bubble_coords 覆盖，这里无需整数化
        fontSize: Number(s.fontSize) || 24,
        fontFamily: s.fontFamily || DEFAULT_FONT_FAMILY,
        textDirection: getEffectiveDirection(s),
        textAlign: s.textAlign || 'center',
        textColor: s.textColor || '#231816',
        rotationAngle: Math.round(Number(s.rotationAngle) || 0),
        position: s.position ? { x: Math.round(s.position.x || 0), y: Math.round(s.position.y || 0) } : { x: 0, y: 0 },
        strokeEnabled: s.strokeEnabled !== undefined ? s.strokeEnabled : true,
        strokeColor: s.strokeColor || '#FFFFFF',
        strokeWidth: Number(s.strokeWidth) || 3,
      }))

      // 调用后端API（使用正确的参数格式，确保数值类型）
      const response = await reRenderImage({
        clean_image: cleanBase64,
        bubble_texts: bubbleTexts,
        bubble_coords: bubbleCoords,
        bubble_states: bubbleStatesForApi,
        fontSize: Number(bubbleStates[0]?.fontSize) || 24,
        fontFamily: bubbleStates[0]?.fontFamily || DEFAULT_FONT_FAMILY,
        textDirection: bubbleStates[0] ? getEffectiveDirection(bubbleStates[0]) : 'vertical',
        textColor: bubbleStates[0]?.textColor || '#231816',
        strokeEnabled: bubbleStates[0]?.strokeEnabled !== undefined ? bubbleStates[0].strokeEnabled : true,
        strokeColor: bubbleStates[0]?.strokeColor || '#FFFFFF',
        strokeWidth: Number(bubbleStates[0]?.strokeWidth) || 3,
        use_individual_styles: true,
        use_inpainting: false,
        use_lama: false,
        is_font_style_change: true
      } as any)

      // 检查token是否过期（被新的渲染请求取代）
      if (renderTokenByImageId[imageId] !== renderToken) {
        console.log('reRenderFullImage: 渲染结果已过期，忽略')
        return false
      }

      // 后端返回的是 rendered_image 而不是 translated_image
      if (response.rendered_image) {
        // 更新翻译图
        const translatedDataURL = `data:image/png;base64,${response.rendered_image}`
        image.translatedDataURL = translatedDataURL

        console.log('reRenderFullImage: 渲染成功')
        if (!silentMode) callbacks?.onRenderSuccess?.(translatedDataURL)
        return true
      } else {
        const errorMsg = response.error || '渲染失败'
        console.error('reRenderFullImage: 渲染失败 -', errorMsg)
        renderError.value = errorMsg
        if (!silentMode) callbacks?.onRenderError?.(errorMsg)
        return false
      }
    } catch (error) {
      // 检查token是否过期
      if (renderTokenByImageId[imageId] !== renderToken) {
        return false
      }

      const errorMsg = error instanceof Error ? error.message : '渲染请求失败'
      console.error('reRenderFullImage: 渲染出错 -', error)
      renderError.value = errorMsg
      if (!silentMode) callbacks?.onRenderError?.(errorMsg)
      return false
    } finally {
      if (renderTokenByImageId[imageId] === renderToken) {
        delete renderTokenByImageId[imageId]
      }
      activeRenderCount = Math.max(0, activeRenderCount - 1)
      isRendering.value = activeRenderCount > 0
      if (!silentMode) callbacks?.onRenderEnd?.()
    }
  }

  /**
   * 取消当前渲染
   */
  function cancelRender(): void {
    for (const k of Object.keys(renderTokenByImageId)) {
      delete renderTokenByImageId[k]
    }
    activeRenderCount = 0
    isRendering.value = false
  }

  // ============================================================
  // 返回接口
  // ============================================================

  return {
    // 状态
    isRendering,
    renderError,

    // 方法
    reRenderFullImage,
    cancelRender
  }
}
