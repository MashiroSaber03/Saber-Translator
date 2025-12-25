/**
 * 编辑模式组合式函数
 * 简化版：仅提供编辑模式状态管理
 * 具体的编辑功能由 EditWorkspace.vue 和相关 composables 处理
 */

import { ref } from 'vue'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'

// ============================================================
// 类型定义
// ============================================================

export type ViewMode = 'dual' | 'original' | 'translated'
export type LayoutMode = 'horizontal' | 'vertical'

// ============================================================
// 组合式函数
// ============================================================

export function useEditMode() {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()

  // ============================================================
  // 状态定义
  // ============================================================

  /** 编辑模式是否激活 */
  const isActive = ref(false)

  // ============================================================
  // 编辑模式控制
  // ============================================================

  /**
   * 退出编辑模式但不触发重渲染
   * 仅被 useTranslateInit 使用
   */
  function exitEditModeWithoutRender(): void {
    if (!isActive.value) return

    const currentImage = imageStore.currentImage
    
    // 【修复】与原版一致：保存气泡状态，包括处理用户删除所有气泡的情况
    if (bubbleStore.bubbleCount > 0) {
      // 有气泡，保存当前状态
      imageStore.updateCurrentBubbleStates([...bubbleStore.bubbles])
    } else if (currentImage && Array.isArray(currentImage.bubbleStates) && currentImage.bubbleStates.length > 0) {
      // 用户删除了所有气泡，需要同步更新为空数组（保持"已处理过"的语义）
      imageStore.updateCurrentBubbleStates([])
      console.log('退出编辑模式（无重渲染），用户已删除所有气泡')
    }

    // 清理状态
    isActive.value = false
    bubbleStore.clearSelection()

    console.log('已退出编辑模式（无重渲染）')
  }

  // ============================================================
  // 返回接口
  // ============================================================

  return {
    // 状态
    isActive,

    // 编辑模式控制
    exitEditModeWithoutRender
  }
}
